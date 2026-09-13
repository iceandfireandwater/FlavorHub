"""
Answer formatting node.
"""

import json
from typing import Any, Callable, Coroutine, Dict, List

from gustobot.infrastructure.core.logger import get_logger

logger = get_logger(service="text2sql.answer_formatter")


def create_answer_formatter_node() -> Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]:
    """Build a LangGraph node that composes the final answer for the user."""

    async def format_answer(state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("-----格式化最终回答-----")

        execution_error = state.get("execution_error")
        sql_statement = state.get("sql_statement", "")
        results = state.get("execution_results") or []
        analysis_text = state.get("analysis_text") or ""
        visualization = state.get("visualization") or {}
        question = state.get("question", "")

        if execution_error:
            answer = f"抱歉，执行 SQL 时出现错误：{execution_error}"
        else:
            answer_lines: List[str] = []
            answer_lines.append(f"### 查询结果摘要\n问题：{question or '（未提供）'}")
            if analysis_text:
                answer_lines.append("\n---\n")
                answer_lines.append(analysis_text)
            if results:
                answer_lines.append("\n---\n### 结果预览")
                preview = json.dumps(results[:5], ensure_ascii=False, indent=2)
                answer_lines.append(f"```json\n{preview}\n```")
            if visualization:
                answer_lines.append("\n---\n### 可视化建议")
                answer_lines.append(
                    f"- 类型：{visualization.get('chart_type', 'table')}\n"
                    f"- 标题：{visualization.get('title', '查询结果')}"
                )
                config = visualization.get("config") if isinstance(visualization, dict) else None
                if config:
                    answer_lines.append("```json")
                    answer_lines.append(json.dumps(config, ensure_ascii=False, indent=2))
                    answer_lines.append("```")
            answer = "\n".join(answer_lines).strip()

        viz_config = None
        if isinstance(visualization, dict):
            viz_config = visualization.get("config")

        logger.info("最终回答：{}", answer)
        # ### 查询结果摘要
        # 问题：统计所有菜品的数量

        # ---

        # ## SQL 命令分析报告

        # ### 1. 查询意图
        # 统计所有菜品的数量

        # ### 2. 涉及的表
        # - recipes

        # ### 3. 关键字段
        # - COUNT(*) AS total_recipes

        # ### 6. 聚合需求
        # COUNT(*) AS total_recipes

        # ---
        # ### 结果预览
        # ```json
        # [
        # {
        #     "total_recipes": 70
        # }
        # ]
        # ```

        # ---
        # ### 可视化建议
        # - 类型：table
        # - 标题：菜品数量统计
        logger.info("SQL 语句：{}", sql_statement)
        # SELECT COUNT(*) AS total_recipes FROM recipes;
        logger.info("执行结果：{}", results)
        # [{'total_recipes': 70}]
        logger.info("可视化配置：{}", viz_config)
        # None

        return {
            "answer": answer,
            "sql_statement": sql_statement,
            "execution_results": results,
            "visualization": visualization if isinstance(visualization, dict) else None,
            "visualization_config": viz_config,
            "steps": ["format_answer"],
        }

    return format_answer
