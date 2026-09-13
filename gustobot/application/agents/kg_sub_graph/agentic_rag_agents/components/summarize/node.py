"""
This code is based on content found in the LangGraph documentation: https://python.langchain.com/docs/tutorials/graph/#advanced-implementation-with-langgraph
"""

from loguru import logger

from typing import Any, Callable, Coroutine, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.state import OverallState
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.summarize.prompts import create_summarization_prompt_template
from gustobot.infrastructure.core.logger import get_logger

logger = get_logger(service="summarize-node")

generate_summary_prompt = create_summarization_prompt_template()


async def _milvus_fallback_context(question: str) -> str:
    """图谱(Neo4j)没命中时，直接查 Milvus 取上下文。

    背景：graphrag-query 默认用 Cypher 查 Neo4j，其数据源是 recipe.json；
    但 8 本古籍的译文只灌进了 Milvus。实测问「随园食单的萝卜汤圆怎么做」时
    agent 选 predefined_cypher → 图谱没这道菜 → 直接回"暂未找到相关菜谱信息"，
    而 Milvus 里该条目（classics_随园食单_321）相似度 0.72 排第一。
    这里补一次直接向量检索，把古籍内容捞回来。
    """
    if not (question or "").strip():
        return ""
    try:
        from gustobot.infrastructure.knowledge.knowledge_service import KnowledgeService

        docs = await KnowledgeService().search(question, top_k=5) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[summarize] Milvus 兜底检索失败: {exc}")
        return ""
    parts = []
    for d in docs:
        content = (d.get("content") or "").strip()
        if not content:
            continue
        meta = d.get("metadata") or {}
        label = meta.get("name") or d.get("id") or ""
        parts.append(f"《{label}》：{content}" if label else content)
    return (chr(10) + chr(10)).join(parts).strip()


def create_summarization_node(
    llm: BaseChatModel,
) -> Callable[[OverallState], Coroutine[Any, Any, dict[str, Any]]]:
    """
    Create a Summarization node for a LangGraph workflow.

    Parameters
    ----------
    llm : BaseChatModel
        The LLM do perform processing.

    Returns
    -------
    Callable[[OverallState], OutputState]
        The LangGraph node.
    """

    generate_summary = generate_summary_prompt | llm | StrOutputParser()

    async def summarize(state: OverallState, *, config: RunnableConfig) -> Dict[str, Any]:
        """
        Summarize results of the performed Cypher queries.
        """
        tasks = state.get("tasks", [])
        cypher_entries = state.get("cyphers", [])

        narrative_sections: List[str] = []
        metric_sections: List[str] = []
        error_sections: List[str] = []

        def _format_rows(rows: List[dict[str, Any]]) -> str:
            if not rows:
                return ""
            if len(rows) == 1:
                row = rows[0]
                if len(row) == 1:
                    key, value = next(iter(row.items()))
                    return f"{key}：{value}"
                return "; ".join(f"{key}：{value}" for key, value in row.items())
            lines = []

            # 检查是否是烹饪步骤（包含"步骤序号"和"步骤说明"）
            is_cooking_steps = all("步骤序号" in row and "步骤说明" in row for row in rows if isinstance(row, dict))

            # 检查是否是食材列表（包含"食材"和"用量"）
            is_ingredients = all("食材" in row and "用量" in row for row in rows if isinstance(row, dict))

            for idx, row in enumerate(rows, 1):
                if is_cooking_steps and isinstance(row, dict):
                    # 烹饪步骤：只显示步骤说明
                    step_num = row.get("步骤序号", idx)
                    step_desc = row.get("步骤说明", "")
                    lines.append(f"{step_num}. {step_desc}")
                elif is_ingredients and isinstance(row, dict):
                    # 食材列表：只显示食材名和用量，隐藏关系类型
                    ingredient = row.get("食材", "")
                    amount = row.get("用量", "")
                    relation = row.get("关系类型", "")
                    # 主料用 ★ 标记
                    marker = "★ " if "MAIN" in relation else "  "
                    lines.append(f"{marker}{ingredient}：{amount}")
                else:
                    # 其他数据：显示所有字段
                    row_desc = ", ".join(f"{key}：{value}" for key, value in row.items())
                    lines.append(f"{idx}. {row_desc}")
            return "\n".join(lines)

        for idx, cypher in enumerate(cypher_entries):
            if hasattr(cypher, "model_dump"):
                data = cypher.model_dump()
            elif isinstance(cypher, dict):
                data = cypher
            else:
                data = {}

            task_label = ""
            if idx < len(tasks):
                task_label = tasks[idx].question
            else:
                task_label = data.get("task") or ""

            records = data.get("records") or {}
            errors = data.get("errors") or []

            if errors:
                error_sections.append(f"{task_label}：{'；'.join(errors)}" if task_label else "；".join(errors))
                continue

            if not records:
                continue

            if isinstance(records, dict):
                if isinstance(records.get("result"), str) and records["result"].strip():
                    narrative_sections.append(records["result"].strip())

                answer = records.get("answer")
                if answer:
                    metric_sections.append(f"{task_label}：{answer}".strip())

                rows = records.get("rows")
                if isinstance(rows, list) and rows:
                    formatted_rows = _format_rows(rows)
                    if formatted_rows:
                        metric_sections.append(
                            f"{task_label}：\n{formatted_rows}".rstrip()
                            if task_label
                            else formatted_rows
                        )
            elif isinstance(records, list):
                # 如果 records 是列表，友好格式化输出
                formatted_rows = _format_rows(records)
                if formatted_rows:
                    metric_sections.append(
                        f"{task_label}：\n{formatted_rows}".rstrip()
                        if task_label
                        else formatted_rows
                    )
            else:
                metric_sections.append(str(records))

        # 构建原始数据摘要，作为 LLM 的输入上下文
        raw_context_parts: List[str] = []
        if narrative_sections:
            raw_context_parts.append("### 川菜概览\n" + "\n\n".join(narrative_sections))
        if metric_sections:
            raw_context_parts.append("### 数据统计\n" + "\n\n".join(metric_sections))
        if error_sections:
            raw_context_parts.append("### 查询提示\n" + "\n".join(f"- {msg}" for msg in error_sections))

        raw_context = "\n\n".join(part for part in raw_context_parts if part).strip()

        # 判断"真的没查到"。注意：**不能只看 raw_context 是否为空** —— Cypher 查询失败时
        # 会返回 {"error": "I couldn't find any relevant information ..."}，它会被当成
        # "### 数据统计" 塞进 raw_context，于是 raw_context 非空但内容其实是个错误提示，
        # 模型据此回答"暂未找到相关菜谱信息"。所以要连这个特征串一起判。
        _NO_INFO_MARKS = (
            "couldn't find any relevant information",
            "could not find any relevant information",
            "no relevant information",
        )
        no_hit = (not raw_context) or any(k in raw_context.lower() for k in _NO_INFO_MARKS)
        logger.info("[summarize] raw_context 长度={} no_hit={} 前80字={!r}",
                    len(raw_context), no_hit, raw_context[:80])

        if no_hit:
            # 兜底：直接查 Milvus（8 本古籍译文只灌进了这里，Neo4j 图谱里没有）
            fb = await _milvus_fallback_context(state.get("question", ""))
            if fb:
                raw_context = fb
                logger.info("[summarize] Neo4j 未命中 -> Milvus 兜底命中, 长度={}", len(fb))
            elif not raw_context:
                return {"summary": "抱歉，暂未找到相关菜谱信息，请尝试换一种方式提问。",
                        "steps": ["summarize"]}

        # 获取用户问题
        question = state.get("question", "")

        # 调用 LLM 生成友好的回答
        try:
            summary = await generate_summary.ainvoke({
                "results": raw_context,
                "question": question,
            }, config=config)
        except Exception as e:
            # 如果 LLM 调用失败，返回原始数据作为兜底
            print(f"LLM summarization failed, using raw context: {e}")
            summary = raw_context
            print(f"LLM summarization failed, using raw context: {raw_context}")

        return {"summary": summary, "steps": ["summarize"]}

    return summarize
