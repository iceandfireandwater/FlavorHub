import asyncio
from typing import Any, Dict, List, Optional, Literal

from operator import add

import aiohttp

try:  # pragma: no cover - prefer typing_extensions for Pydantic compatibility
    from typing_extensions import Annotated, TypedDict  # type: ignore
except ImportError:  # pragma: no cover - minimal stdlib fallback
    from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_neo4j import Neo4jGraph
from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

# 导入输入输出状态定义
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.state import (
    InputState,
    OutputState,
    OverallState,
)
# 导入guardrails逻辑
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.guardrails.node import create_guardrails_node
# 导入分解节点
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.planner import create_planner_node
# 导入工具选择节点
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.tool_selection import create_tool_selection_node
# 导入 text2cypher 节点
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.cypher_tools import create_cypher_query_node
# 导入Cypher示例检索器基类
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.base import BaseCypherExampleRetriever
# 导入预定义Cypher节点
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.predefined_cypher import create_predefined_cypher_node
# 导入自定义工具函数节点
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.customer_tools import create_graphrag_query_node
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.text2cypher.text2sql_tool import create_text2sql_tool_node

from gustobot.config import settings
from gustobot.infrastructure.core.logger import get_logger
from gustobot.infrastructure.knowledge import KnowledgeService


from ...components.errors import create_error_tool_selection_node
from ...components.final_answer import create_final_answer_node



from ...components.summarize import create_summarization_node


# 获取日志记录器
kb_logger = get_logger(service="kb-multi-tool")

from .edges import (
    guardrails_conditional_edge,
    map_reduce_planner_to_tool_selection,
)

from dataclasses import dataclass, field
# 强制要求数据类中的所有字段必须以关键字参数的形式提供。即不能以位置参数的方式传递。
@dataclass(kw_only=True)
class AgentState(InputState):
    """The router's classification of the user's query."""
    steps: list[str] = field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""
    question: str = field(default_factory=str) # 这个参数用来与子图进行交互
    answer: str = field(default_factory=str)  # 这个参数用来与子图进行交互

# 子图A
async def _search_local_kb(question: str) -> List[Dict[str, Any]]:
    """直接查本地向量库（Milvus）。

    graphrag-query 默认用 Cypher 查 Neo4j，其数据源是 recipe.json；但 8 本古籍的译文
    只灌进了 Milvus。实测问「随园食单的萝卜汤圆怎么做」时 agent 会选 predefined_cypher
    → 图谱里没这道菜 → 直接答"没查到"，而 Milvus 里该条目相似度 0.72 排第一。
    这里作为"图谱未命中"时的一次直接兜底检索。
    """
    if not (question or "").strip():
        return []
    try:
        from gustobot.infrastructure.knowledge.knowledge_service import KnowledgeService

        return await KnowledgeService().search(question, top_k=5) or []
    except Exception as exc:  # noqa: BLE001
        kb_logger.warning("Milvus 兜底检索失败: {}", exc)
        return []


def create_multi_tool_workflow(
    llm: BaseChatModel,
    graph: Neo4jGraph,
    tool_schemas: List[type[BaseModel]],
    predefined_cypher_dict: Dict[str, str],
    cypher_example_retriever: BaseCypherExampleRetriever,
    scope_description: Optional[str] = None,
    llm_cypher_validation: bool = True,
    max_attempts: int = 3,
    attempt_cypher_execution_on_final_attempt: bool = False,
    default_to_text2cypher: bool = True,
) -> CompiledStateGraph:
    """
    Create a multi tool Agent workflow using LangGraph.
    This workflow allows an agent to select from various tools to complete each identified task.

    Parameters
    ----------
    llm : BaseChatModel
        The LLM to use for processing
    graph : Neo4jGraph
        The Neo4j graph wrapper.
    tool_schemas : List[BaseModel]
        A list of Pydantic class defining the available tools.
    predefined_cypher_dict : Dict[str, str]
        A Python dictionary of Cypher query names as keys and Cypher queries as values.
    scope_description: Optional[str], optional
        A short description of the application scope, by default None
    cypher_example_retriever: BaseCypherExampleRetriever
        The retriever used to collect Cypher examples for few shot prompting.
    llm_cypher_validation : bool, optional
        Whether to perform LLM validation with the provided LLM, by default True
    max_attempts: int, optional
        The max number of allowed attempts to generate valid Cypher, by default 3
    attempt_cypher_execution_on_final_attempt, bool, optional
        THIS MAY BE DANGEROUS.
        Whether to attempt Cypher execution on the last attempt, regardless of if the Cypher contains errors, by default False
    default_to_text2cypher : bool, optional
        Whether to attempt Text2Cypher if no tool calls are returned by the LLM, by default True
    initial_state: Optional[InputState], optional
        An initial state passed from parent graph, by default None

    Returns
    -------
    CompiledStateGraph
        The workflow.
    """
    # 1. 创建guardrails节点
    # Guardrails 节点决定传入的问题是否在检索的范围内（比如是否和电商（自家的产品相关））。如果不在，则提供默认消息，并且工作流路由到最终的答案生成。
    
    # {"next_action": "planner", "summary": None, "steps": ["guardrails"]}
    guardrails = create_guardrails_node(
        llm=llm, graph=graph, scope_description=scope_description
    )

    kb_logger.info("Guardrails 节点已被创建，是个函数：{}", guardrails)

    # 2. 如果通过guardrails，则会针对用户的问题进行任务分解，sub_tasks
    planner = create_planner_node(llm=llm)

    kb_logger.info("Planner 节点已被创建，是个函数：{}", planner)

    # 3. 创建cypher_query节点，用来根据用户的问题生成Cypher查询语句 大模型生成Cypher查询语句
    cypher_query = create_cypher_query_node()

    kb_logger.info("Cypher Query 节点已被创建，是个函数：{}", cypher_query)

    predefined_cypher = create_predefined_cypher_node(
        graph=graph, predefined_cypher_dict=predefined_cypher_dict
    ) #预定义的自定Cypher查询语句

    customer_tools = create_graphrag_query_node() # lightrag_query
    
    text2sql_query = create_text2sql_tool_node(graph)

    # 工具选择节点，根据用户的问题选择合适的工具
    tool_selection = create_tool_selection_node(
        llm=llm,
        tool_schemas=tool_schemas,
        default_to_text2cypher=default_to_text2cypher,
    )
    summarize = create_summarization_node(llm=llm)

    final_answer = create_final_answer_node()

    # 创建状态图运行时会维护一个“全局状态”（OverallState），入口状态类型是 InputState，最终产出是 OutputState。节点函数读写的就是这个状态。
    main_graph_builder = StateGraph(OverallState, input=InputState, output=OutputState)

    main_graph_builder.add_node(guardrails)# 安全护栏敏感内容过滤、权限/配额校验
    main_graph_builder.add_node(planner) #决定下一步要用的工具/路径。
    main_graph_builder.add_node("cypher_query", cypher_query)#命名 "cypher_query" 的节点，执行 cypher_query 函数（通常是对图数据库生成/执行 Cypher）。
    main_graph_builder.add_node(predefined_cypher) #预设查询（当无需动态生成时）。
    main_graph_builder.add_node("customer_tools", customer_tools) #lightrag_query
    main_graph_builder.add_node("text2sql_query", text2sql_query)
    main_graph_builder.add_node(summarize) # 总结
    main_graph_builder.add_node(tool_selection) #工具选择的中间控制节点（通常结合 planner 的输出）。
    main_graph_builder.add_node(final_answer)


    # 添加边
    main_graph_builder.add_edge(START, "guardrails")
    main_graph_builder.add_conditional_edges(
        "guardrails",
        guardrails_conditional_edge,
    ) #这是条件边：执行完 guardrails 后，不是固定跳到某个节点，而是调用 guardrails_conditional_edge(state) 来返回下一跳的节点名（或一个映射）。
    main_graph_builder.add_conditional_edges(
        "planner",
        map_reduce_planner_to_tool_selection, #据 planner 写进 state 的结果，返回下一个要去的节点名
        ["tool_selection"], #从 planner 出来只能跳到 "tool_selection"，且由 map_reduce_planner_to_tool_selection(state) 来决定（但这里其实被限制成只能选这一个）。
    )

    main_graph_builder.add_edge("cypher_query", "summarize")
    main_graph_builder.add_edge("predefined_cypher", "summarize")
    main_graph_builder.add_edge("customer_tools", "summarize")
    main_graph_builder.add_edge("text2sql_query", "summarize")
    main_graph_builder.add_edge("summarize", "final_answer")

    main_graph_builder.add_edge("final_answer", END)

    return main_graph_builder.compile()


# kb_logger = get_logger(service="kb-multi-tool")


class KBGuardrailsDecision(BaseModel):
    decision: Literal["proceed", "end"]
    summary: Optional[str] = None
    rationale: Optional[str] = None


class KBRouteDecision(BaseModel):
    route: Literal["local", "external", "hybrid"]
    rationale: str
    tools: List[Literal["milvus", "postgres"]] = Field(
        description="本地知识源检索工具列表，支持 milvus/postgres",
    )


class KBInputState(TypedDict):
    question: str
    history: List[Dict[str, str]]


class KBWorkflowState(TypedDict):
    question: str
    history: List[Dict[str, str]]
    guardrails_decision: str
    summary: str
    route: str
    kb_tools: List[str]
    milvus_results: List[Dict[str, Any]]
    postgres_results: List[Dict[str, Any]]
    local_results: List[Dict[str, Any]]
    external_results: List[Dict[str, Any]]
    answer: str
    steps: Annotated[List[str], add]
    # 覆盖语义（原来用 Annotated[..., add] 累加）：子图 state 会经 checkpoint
    # 跨轮恢复，累加会让上一轮的 URL 一直滚到下一轮（表现为「每次都是那几个网址」）。
    sources: List[str]


class KBOutputState(TypedDict):
    answer: str
    steps: List[str]
    sources: List[str]

# 子图B
def create_kb_multi_tool_workflow(
    llm: BaseChatModel,
    knowledge_service: Optional[KnowledgeService] = None,
    *,
    top_k: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    filter_expr: Optional[str] = None,
    allow_external: Optional[bool] = None,
    external_search_url: Optional[str] = None,
    external_search_timeout: Optional[float] = None,
    scope_description: Optional[str] = None,
) -> CompiledStateGraph:
    """
    Create a multi-tool workflow for knowledge base queries.

    This workflow performs guardrails checking, routes the question to the most
    appropriate retrieval source (local vector store, external API, or both),
    and then synthesises a response with safety-aware instructions.
    此工作流执行安全检查，将问题路由到最合适的检索源（本地向量存储、外部API或两者结合），然后生成一个包含安全意识指令的综合响应。
    """

    knowledge_service = knowledge_service or KnowledgeService()
    effective_top_k = top_k or settings.KB_TOP_K

    effective_threshold = (  # 0.2
        similarity_threshold
        if similarity_threshold is not None
        else settings.KB_SIMILARITY_THRESHOLD
    )

   
    allow_external_search = (  # False
        allow_external
        if allow_external is not None
        else settings.KB_ENABLE_EXTERNAL_SEARCH
    )
    
    # http://localhost:8000
    ingest_service_base = settings.INGEST_SERVICE_URL.rstrip("/") if settings.INGEST_SERVICE_URL else None

    # 此接口的功能：检索向量库
    postgres_search_url = (
        f"{ingest_service_base}/api/v1/knowledge/search" if ingest_service_base else None
    )

    external_url = external_search_url or settings.KB_EXTERNAL_SEARCH_URL
    
    kb_logger.info("过滤表达式：{}".format(filter_expr))  # None
    kb_logger.info("是否允许外部检索：{}".format(allow_external_search))  # False
    kb_logger.info("向量库检索接口：{}".format(postgres_search_url))  # http://localhost:8000/api/v1/knowledge/search
    kb_logger.info("外部检索服务地址：{}".format(external_url))  # http://localhost:8000/api/search
    kb_logger.info("外部检索超时设置：{}".format(external_search_timeout))  # None


    if allow_external_search and not external_url:
        kb_logger.warning(
            "External search enabled but KB_EXTERNAL_SEARCH_URL 未配置，已自动关闭外部检索。"
        )
        allow_external_search = False

    # 判断外部检索是否使用向量库
    external_is_postgres = bool(
        allow_external_search
        and external_url
        and postgres_search_url
        and external_url.rstrip("/") == postgres_search_url.rstrip("/")
    )

    if not allow_external_search:
        external_url = None

    request_timeout = (
        external_search_timeout
        if external_search_timeout is not None
        else settings.KB_EXTERNAL_SEARCH_TIMEOUT
    )

    # scope：范围，即知识库的服务范围
    scope_text = scope_description or (
        "菜谱助手处理一切与「吃」有关的问题，包括但不限于："
        "菜谱做法与步骤、食材与调料搭配、烹饪技巧与厨具用法、"
        "营养功效与适宜人群、菜系流派与饮食文化、菜谱的历史渊源与典故、"
        "以及需要联网补充的外部信息（当下流行菜、应季食材、健康饮食参考资料）。"
        "用户询问关于自己的信息或之前说过的事（姓名、城市、职业、忌口、偏好）也必须放行。"
        "只有与饮食完全无关的话题才算超范围：政治、娱乐八卦、新闻时事、天气预报、"
        "网购推荐、纯医疗诊断与用药建议、编程与数学题等。"
        "拿不准时选择放行——本系统还有外部检索兜底，宁可多查也不要误拒。"
    )

    # 1. 安全检查：问题是否在服务范围内？
    # output：decision = 'proceed' 或 'end'，有时出现 'continue'
    guardrails_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是菜谱助手的范围与安全检查员。服务范围：" + chr(10)
                    + scope_text + chr(10) + chr(10)
                    + "请判断用户问题是否位于该范围，并确保不包含违法、隐私或未授权内容。"
                    + "注意：知识库覆盖不全并不代表问题超范围——只要与饮食/烹饪相关就应放行，"
                    + "后续还有联网检索可补充。"
                    + "仅当问题明显与饮食无关或含违法内容时，才返回 decision='end' 并给出中文 summary；"
                    + "否则返回 decision='proceed'。"
                ),
            ),
            ("human", "用户问题：{question}"),
        ]
    )
    guardrails_chain = guardrails_prompt | llm.with_structured_output(KBGuardrailsDecision)

    # 2. 路由决策：选择哪些工具？
    # output：route = 'local' 或 'external' 或 'hybrid'，tools = ['postgres', 'milvus']（推荐） 或 ['postgres'] 或 ['milvus'] 或 []
    # 3. 检索方式：本地/外部检索？
    # 先检索Postgres，若有直接返回，若没有再检索Milvus
    router_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是菜谱文化知识检索路由器。专门负责将历史文化类问题路由到最合适的知识库。\n\n"
                    "## 服务范围\n"
                    '仅限于"菜谱历史/渊源/命名"、"历史名人与菜谱关系"、"菜谱条目级小传与典故"、"菜系流派介绍"。\n\n'
                    "## 本地知识源说明\n"
                    "- **postgres**（PostgreSQL pgvector）：**第一优先级**，存放结构化表格数据、Excel导入的枚举字段\n"
                    "  - 数据更准确、查询更快、覆盖面广\n"
                    "  - 适合：菜谱名称、菜系、历史事件、人物关系等结构化查询\n"
                    "  - 典型问题：菜谱相关的任何历史文化问题\n"
                    "  - 执行策略：系统会**先查 postgres**，如果有结果就直接使用，**不会查询 milvus**\n"
                    "- **milvus**（Milvus向量库）：**仅作为兜底**，存放长文本、文章、典故故事等非结构化内容\n"
                    "  - 只有在 postgres 无结果时才会查询\n"
                    "  - 典型问题：\"宫保鸡丁的完整历史故事\"、\"川菜的详细发展史\"（需要长篇叙事时）\n\n"
                    "## 路由决策规则（严格执行：postgres 优先）\n"
                    "请根据问题特征选择合适的路由和工具：\n\n"
                    "**1. 通用历史文化查询（默认推荐）**\n"
                    "   - 适用于：大部分历史、典故、由来、文化、背景、流派、特点等问题\n"
                    "   - route: local, tools: ['postgres', 'milvus']\n"
                    "   - 执行流程：postgres → 有结果则返回 → 无结果才用 milvus 兜底\n\n"
                    "**2. 明确的结构化查询（postgres 足够）**\n"
                    "   - 适用于：菜名查询、简短事实查询、人物关系、年代查询\n"
                    "   - route: local, tools: ['postgres']\n\n"
                    "**3. 明确需要长文本叙事（可能需要 milvus）**\n"
                    "   - 适用于：用户明确要求\"完整故事\"、\"详细历史\"、\"长篇介绍\"\n"
                    "   - route: local, tools: ['milvus']\n\n"
                    "**4. 外部检索类（需要外网资料）**\n"
                    "   - 本地知识库可能不足，需要外部检索\n"
                    "   - route: hybrid, tools: ['milvus']\n\n"
                    "**5. 超出范围类（拒绝回答）**\n"
                    "   - 问题涉及烹饪步骤、食材搭配等非文化内容\n"
                    "   - route: local, tools: []（空列表表示无法处理）\n\n"
                    "## 输出格式\n"
                    "请输出三个字段：\n"
                    "- route：local（本地）/ external（外部）/ hybrid（混合）\n"
                    "- tools：列表，元素为 'postgres' 和/或 'milvus'，若拒绝回答则为空列表 []\n"
                    "  - **默认推荐**: ['postgres', 'milvus'] 让系统自动优先使用 postgres\n"
                    "- rationale：中文简要说明选择理由（1-2句话）"
                ),
            ),
            (
                "human",
                "用户问题：{question}\n\n最近对话历史：\n{history}",
            ),
        ]
    )
    router_chain = router_prompt | llm.with_structured_output(KBRouteDecision)

    # 4. 综合结果：融合本地和外部检索结果
    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是菜谱与饮食助手，需要依据给定的检索结果回答用户问题。请遵循：" + chr(10)
                    + "1. 可以回答与烹饪饮食相关的各类问题：菜谱做法与步骤、食材调料搭配、烹饪技巧与厨具、"
                    + "营养功效、菜系文化与历史典故——不要自我设限。" + chr(10)
                    + "2. 优先使用检索结果，尤其是「外部检索结果」中的时效性信息（当下流行、新做法、"
                    + "新厨具用法等）；把不同来源的要点融合成通顺的回答，不要生硬堆砌原文。" + chr(10)
                    + "3. 若检索结果确实不足，可以基于常识给出一般性建议，并说明这部分未来自知识库；"
                    + "不要动辄回复“知识库暂无记载”。" + chr(10)
                    + "4. 语气专业友好，使用简体中文；如引用了资料，在结尾列出来源名称或链接。" + chr(10)
                    + "5. 只有问题与饮食完全无关（政治、娱乐、天气、编程等）时才委婉拒答。"
                ),
            ),
            (
                "human",
                (
                    "用户问题：{question}\n\n"
                    "Milvus 向量检索结果：\n{milvus_context}\n\n"
                    "PostgreSQL 结构化检索结果：\n{postgres_context}\n\n"
                    "外部检索结果：\n{external_context}"
                ),
            ),
        ]
    )

    def _history_to_text(history: List[Dict[str, str]], limit: int = 4) -> str:
        if not history:
            return "（无历史对话）"
        segments: List[str] = []
        for item in history[-limit:]:
            role = item.get("role", "user")
            content = item.get("content", "")
            segments.append(f"{role}: {content}")
        return "\n".join(segments)

    TOOL_LABELS = {
        "milvus": "Milvus",
        "postgres": "PostgreSQL",
    }

    def _format_results(
        results: List[Dict[str, Any]],
        *,
        default_label: str,
        empty_hint: str,
    ) -> str:
        if not results:
            return empty_hint
        snippets: List[str] = []
        for idx, doc in enumerate(results[:effective_top_k]):
            content = doc.get("content") or doc.get("document") or ""
            snippet = content.strip().replace("\n", " ")
            snippet = snippet[:500]
            meta = doc.get("metadata") or {}
            source = (
                doc.get("source")
                or doc.get("source_table")
                or meta.get("source")
                or meta.get("source_table")
                or meta.get("title")
                or ""
            )
            tool_label = TOOL_LABELS.get(str(doc.get("tool", "")).lower(), default_label)
            tag = f"[{tool_label}#{idx + 1}] {snippet}"
            if source:
                tag = f"{tag}\n来源：{source}"
            snippets.append(tag)
        return "\n\n".join(snippets)

    def _format_milvus_results(results: List[Dict[str, Any]]) -> str:
        return _format_results(
            results,
            default_label="Milvus",
            empty_hint="（Milvus 暂无检索结果）",
        )

    def _format_postgres_results(results: List[Dict[str, Any]]) -> str:
        return _format_results(
            results,
            default_label="PostgreSQL",
            empty_hint="（PostgreSQL 暂无检索结果）",
        )

    def _format_combined_local_results(results: List[Dict[str, Any]]) -> str:
        return _format_results(
            results,
            default_label="本地",
            empty_hint="（无本地检索结果）",
        )

    def _format_external_results(results: List[Dict[str, Any]]) -> str:
        if not results:
            return "（无外部检索结果）"
        snippets: List[str] = []
        for idx, item in enumerate(results[:effective_top_k]):
            content = item.get("content") or item.get("summary") or ""
            snippet = content.strip().replace("\n", " ")
            snippet = snippet[:500]
            meta = item.get("metadata") or {}
            source = (
                item.get("source")
                or item.get("source_table")
                or meta.get("source")
                or meta.get("source_table")
                or meta.get("url")
                or ""
            )
            tag = f"[外部#{idx + 1}] {snippet}"
            if source:
                tag = f"{tag}\n来源：{source}"
            snippets.append(tag)
        return "\n\n".join(snippets)

    def _collect_sources(
        *result_sets: List[Dict[str, Any]],
    ) -> List[str]:
        collected: List[str] = []
        for dataset in result_sets:
            for doc in dataset or []:
                meta = doc.get("metadata") or {}
                candidate = (
                    doc.get("source")
                    or doc.get("source_table")
                    or doc.get("document_id")
                    or doc.get("source_id")
                    or doc.get("id")
                    or meta.get("source")
                    or meta.get("source_table")
                    or meta.get("url")
                    or meta.get("title")
                )
                if candidate:
                    collected.append(str(candidate))
        # 去重但保留顺序
        seen: Dict[str, None] = {}
        for source in collected:
            seen.setdefault(source, None)
        return list(seen.keys())

    # 第一步：安全审查
    async def guardrails(state: KBWorkflowState) -> Dict[str, Any]:
        # {'question': '明朝引进了哪些食物？', 'history': [], 'steps': [], 'sources': []}
        kb_logger.info("第一步：安全审查，state: {}", state)

        question = state.get("question", "")
        decision = await guardrails_chain.ainvoke({"question": question})  # 返回 'proceed' 或 'end'

        # decision='proceed' summary='用户问题涉及的是历史和文化知识，属于菜谱文化知识库的范围。' rationale=None
        kb_logger.info("第一步：安全审查: {}", decision)

        summary = decision.summary or (
            "抱歉，该问题不在菜谱文化知识库的支持范围内，请询问菜谱历史、典故或名人故事相关内容。"
            if decision.decision == "end"
            else ""
        )

        # proceed
        kb_logger.info("KB guardrails decision: {}", decision.decision)
        return {
            "guardrails_decision": decision.decision,  # proceed
            "summary": summary,  # '用户问题涉及的是历史和文化知识，属于菜谱文化知识库的范围。'
            "steps": ["guardrails"],
        }

    # 第二步：路由决策
    async def router(state: KBWorkflowState) -> Dict[str, Any]:
        # {'question': '明朝引进了哪些食物？', 'history': [], 'guardrails_decision': 'proceed', 'summary': '用户问题涉及的是历史和文化知识，属于菜谱文化知识库的范围。', 'steps': ['guardrails'], 'sources': []}
        kb_logger.info("第二步：路由决策，state: {}", state)

        question = state.get("question", "")
        history_text = _history_to_text(state.get("history", []))

        decision = await router_chain.ainvoke(  # 返回 route = "local" 或 "external" 或 "hybrid"；tools = ['milvus', 'postgres'] 或 []
            {
                "question": question,
                "history": history_text,
            }
        )

        # route='local' rationale='用户询问明朝引进的食物，属于明确的历史结构化查询，优先使用postgres库查询准确数据。' tools=['postgres']
        kb_logger.info("第二步：路由决策: {}", decision)

        route = decision.route
        if route in {"external", "hybrid"} and not allow_external_search:
            kb_logger.info(
                "Router requested {} but external search is disabled; using local instead.",
                route,
            )
            route = "local"
        tools = [tool for tool in decision.tools or [] if tool in {"milvus", "postgres"}]
        if route != "external" and not tools:
            # 默认使用 postgres + milvus 兜底策略
            tools = ["postgres", "milvus"]
            kb_logger.info("Router 未指定工具，使用默认策略: postgres 优先 + milvus 兜底")

        # KB router decision: local tools=['postgres'] (用户询问明朝引进的食物，属于明确的历史结构化查询，优先使用postgres库查询准确数据。)
        kb_logger.info(
            "KB router decision: {} tools={} ({})",
            route,
            tools,
            decision.rationale,
        )
        return {
            "route": route,
            "kb_tools": tools,
            "steps": ["router"],
        }

    # 第三步：本地检索
    async def local_search(state: KBWorkflowState) -> Dict[str, Any]:
        """
        优先使用 PostgreSQL pgvector 结构化查询，如果无结果再用 Milvus 兜底。

        执行策略：
        1. 优先查询 PostgreSQL（如果在工具列表中）
        2. 如果 PostgreSQL 有结果（>= 1条），直接使用，跳过 Milvus
        3. 如果 PostgreSQL 无结果或未被选择，查询 Milvus 作为兜底
        """
        # {'question': '明朝引进了哪些食物？', 'history': [], 'guardrails_decision': 'proceed', 'summary': '用户问题涉及的是历史和文化知识，属于菜谱文化知识库的范围。', 'route': 'local', 'kb_tools': ['postgres'], 'steps': ['guardrails', 'router'], 'sources': []}
        kb_logger.info("第三步：本地检索，state: {}", state)
        question = state.get("question", "")

        # 如果问题为空，则直接返回空结果
        if not question.strip():
            return {
                "milvus_results": [],
                "postgres_results": [],
                "local_results": [],
                "steps": ["local_search"],
            }

        # ['postgres']
        selected_tools = state.get("kb_tools") or ["postgres", "milvus"]
        kb_logger.info("第三步：本地检索，选择检索工具: {}", selected_tools)

        milvus_results: List[Dict[str, Any]] = []
        postgres_results: List[Dict[str, Any]] = []

        # Step 1: 优先查询 PostgreSQL（如果在工具列表中）
        should_try_postgres = "postgres" in selected_tools
        should_try_milvus = "milvus" in selected_tools

        # 确保优先级：如果同时选择了两个工具，先尝试 PostgreSQL
        if should_try_postgres:
            # 如果 PostgreSQL 不可用，直接使用 Milvus
            if not postgres_search_url:
                kb_logger.warning(
                    "PostgreSQL 工具被选中，但 INGEST_SERVICE_URL 未配置，跳过 PostgreSQL 直接使用 Milvus。"
                )
                
                should_try_milvus = True
            else:
                kb_logger.info("🔍 [优先] 查询 PostgreSQL pgvector 结构化数据库...")
                payload: Dict[str, Any] = {
                    "query": question,
                    "top_k": effective_top_k,
                }

                # {'query': '明朝引进了哪些食物？', 'top_k': 5}
                kb_logger.info("第三步：本地搜索：POSTgreSQL 查询参数: {}", payload)

                # 给 payload 添加阈值
                if settings.KB_POSTGRES_SIMILARITY_THRESHOLD is not None:
                    payload["threshold"] = settings.KB_POSTGRES_SIMILARITY_THRESHOLD

                try:
                    timeout_cfg = aiohttp.ClientTimeout(total=request_timeout)
                    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
                        async with session.post(postgres_search_url, json=payload) as response:  # 向 PostgreSQL 发送查询请求
                            kb_logger.info("第三步：本地搜索：POSTgreSQL 查询结果: {}", response)

                            if response.status == 200:  # PostgreSQL 查询成功
                                body = await response.json()
                                data_results = body.get("results") or []  # 原始结果列表

                                # 处理 PostgreSQL 结果，标准化结果格式
                                if isinstance(data_results, list):
                                    for idx, item in enumerate(data_results):
                                        item_copy = dict(item)
                                        metadata_copy = dict(item_copy.get("metadata") or {})
                                        item_copy["metadata"] = metadata_copy
                                        item_copy["tool"] = "postgres"  # 标记来源
                                        similarity = (  # 获取相似度分数
                                            item_copy.get("similarity")
                                            if item_copy.get("similarity") is not None
                                            else item_copy.get("score")
                                        )
                                        if similarity is not None:
                                            try:
                                                item_copy["similarity"] = float(similarity)
                                            except (TypeError, ValueError):
                                                item_copy["similarity"] = 0.0
                                        item_copy["id"] = str(  # 生成唯一 id
                                            item_copy.get("id")
                                            or item_copy.get("document_id")
                                            or item_copy.get("source_id")
                                            or f"postgres_{idx}"
                                        )
                                        postgres_results.append(item_copy)
                                    
                                    kb_logger.info("第三步：本地搜索：标准化后的 POSTgreSQL 查询结果: {}", postgres_results)

                                    # 对 PostgreSQL 结果进行 重排序，提升相关性
                                    if postgres_results and knowledge_service.reranker.enabled:
                                        postgres_results = await knowledge_service.reranker.rerank(
                                            question, postgres_results, effective_top_k
                                        )
                                        kb_logger.info("第三步：本地搜索：POSTgreSQL 重排序结果: {}", postgres_results)
                                    filtered_postgres: List[Dict[str, Any]] = []

                                    # 过滤 PostgreSQL 重排序 结果，根据相似度阈值和重排序阈值
                                    for doc in postgres_results:
                                        similarity = float(doc.get("similarity") or doc.get("score") or 0.0)  # 向量相似度
                                        rerank_score = float(doc.get("rerank_score") or 0.0)  # 重排序分数
                                        if knowledge_service.reranker.enabled:
                                            if (  # 同时满足相似度阈值和重排序阈值（双阈值过滤）
                                                similarity >= settings.KB_POSTGRES_SIMILARITY_THRESHOLD
                                                and rerank_score >= settings.KB_POSTGRES_RERANK_THRESHOLD
                                            ):
                                                filtered_postgres.append(doc)
                                        else:  # 单阈值过滤
                                            if similarity >= settings.KB_POSTGRES_SIMILARITY_THRESHOLD:
                                                filtered_postgres.append(doc)
                                    postgres_results = filtered_postgres[:effective_top_k]  # 限制结果数量
                                    kb_logger.info(
                                        "✅ PostgreSQL 返回 {} 条结果，过滤后保留 {} 条",
                                        len(data_results),
                                        len(postgres_results),
                                    )
                                else:
                                    kb_logger.warning(
                                        "Unexpected PostgreSQL search payload structure: {}",
                                        body,
                                    )
                            else:  # PostgreSQL 查询失败
                                error_text = await response.text()
                                kb_logger.warning(
                                    "PostgreSQL KB search failed ({}): {}",
                                    response.status,
                                    error_text,
                                )
                except Exception as exc:  # pragma: no cover - defensive logging
                    kb_logger.error("PostgreSQL knowledge search error: {}", exc)

        # Step 2: 根据 PostgreSQL 结果决定是否需要 Milvus 兜底
        if postgres_results and len(postgres_results) > 0:
            # PostgreSQL 有结果，直接使用，跳过 Milvus
            kb_logger.info(
                "✅ PostgreSQL 有结果（{}条），直接使用结构化数据，跳过 Milvus 向量查询",
                len(postgres_results)
            )
            combined_results = postgres_results
        else:
            # PostgreSQL 无结果或不可用，使用 Milvus 兜底
            if should_try_milvus:
                if not postgres_results:
                    kb_logger.info("⚠️ PostgreSQL 无结果，使用 Milvus 向量库兜底...")
                else:
                    kb_logger.info("⚠️ PostgreSQL 不可用，使用 Milvus 向量库...")

                try:
                    # 查询 Milvus
                    docs = await knowledge_service.search(
                        query=question,
                        top_k=effective_top_k,
                        similarity_threshold=settings.KB_SIMILARITY_THRESHOLD,
                        filter_expr=filter_expr,
                        filter_by_similarity=not knowledge_service.reranker.enabled,
                    )

                    kb_logger.info("第三步：本地搜索：Milvus 查询原始结果: {}", docs)

                    for doc in docs:
                        doc_copy = dict(doc)
                        metadata_copy = dict(doc.get("metadata") or {})
                        doc_copy["metadata"] = metadata_copy
                        doc_copy["tool"] = "milvus"
                        milvus_results.append(doc_copy)
                    kb_logger.info("✅ Milvus 兜底返回 {} 条结果", len(milvus_results))
                    kb_logger.info("第三步：本地搜索：Milvus 查询格式化后结果: {}", milvus_results)
                except Exception as exc:  # pragma: no cover - defensive logging
                    kb_logger.error("Milvus knowledge search failed: {}", exc)
                combined_results = milvus_results
            else:
                kb_logger.warning("⚠️ 未选择任何可用的知识库工具")
                combined_results = []

        route = state.get("route", "local")
        if (
            not combined_results
            and route in {"local", "hybrid"}
            and allow_external_search
            and external_url
        ):
            kb_logger.info("Local searches empty, falling back to external search.")
            route = "external"

        return {
            "milvus_results": milvus_results,
            "postgres_results": postgres_results,
            "local_results": combined_results,
            "route": route,
            "steps": ["local_search"],
        }

    async def external_search(state: KBWorkflowState) -> Dict[str, Any]:
        if not (allow_external_search and external_url):
            return {"external_results": [], "steps": ["external_search"]}

        if external_is_postgres and "postgres" in (state.get("kb_tools") or []):
            kb_logger.debug(
                "Skip external search: router already执行了 PostgreSQL 工具，且外部检索与其同源。"
            )
            return {"external_results": [], "steps": ["external_search"]}

        question = state.get("question", "")
        if not question.strip():
            return {"external_results": [], "steps": ["external_search"]}

        # 1) 优先用进程内 SearchTool（DuckDuckGo，免费且无需额外服务）；
        #    原来只走 KB_EXTERNAL_SEARCH_URL，未配置时这条外部检索等于不存在。
        try:
            from gustobot.infrastructure.tools.search import SearchTool

            _tool = SearchTool()
            _web = await asyncio.to_thread(
                _tool.search, question, num_results=effective_top_k
            )
            if _web:
                kb_logger.info("SearchTool 外部检索返回 {} 条", len(_web))
                return {
                    "external_results": [
                        {
                            "content": (r.get("snippet") or r.get("title") or ""),
                            "source": r.get("url") or "web",
                            "metadata": {"title": r.get("title") or "", "url": r.get("url") or ""},
                        }
                        for r in _web
                    ],
                    "steps": ["external_search"],
                }
        except Exception as exc:  # noqa: BLE001
            kb_logger.warning("SearchTool 外部检索失败，回退 external_url: {}", exc)

        if not external_url:
            return {"external_results": [], "steps": ["external_search"]}

        payload: Dict[str, Any] = {
            "query": question,
            "top_k": effective_top_k,
        }
        if effective_threshold is not None:
            payload["threshold"] = effective_threshold

        results: List[Dict[str, Any]] = []
        try:
            timeout_cfg = aiohttp.ClientTimeout(total=request_timeout)
            async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
                async with session.post(external_url, json=payload) as response:
                    if response.status == 200:
                        body = await response.json()
                        data_results = body.get("results") or []
                        if isinstance(data_results, list):
                            results = data_results
                        else:
                            kb_logger.warning(
                                "Unexpected external search payload structure: {}",
                                body,
                            )
                    else:
                        error_text = await response.text()
                        kb_logger.warning(
                            "External KB search failed ({}): {}",
                            response.status,
                            error_text,
                        )
        except Exception as exc:  # pragma: no cover - defensive logging
            kb_logger.error("External KB search error: {}", exc)

        return {
            "external_results": results,
            "steps": ["external_search"],
        }

    async def finalize(state: KBWorkflowState, *, config: RunnableConfig) -> KBOutputState:
        # {'question': '明朝引进了哪些食物？', 'history': [], 'guardrails_decision': 'proceed', 'summary': '用户问题涉及的是历史和文化知识，属于菜谱文化知识库的范围。', 'route': 'local', 'kb_tools': ['postgres'], 'milvus_results': [], 'postgres_results': [{'id': 'history_003_0', 'content': '元朝多了豇豆，...，则往往显得单调而匮乏。', 'score': 0.737809419631958, 'metadata': {'recipe_id': 'history_003', 'name': '中国饮食文化历史资料 - 第3段', 'category': '历史文化', 'difficulty': ''}, 'rerank_score': 0.9741574735494072, 'tool': 'postgres', 'similarity': 0.737809419631958}, {'id': 'data.txt_4', 'content': '元朝多了豇豆、...，则往往显得单调而匮乏。', 'score': 0.6897258758544922, 'metadata': {'recipe_id': 'data.txt', 'name': '', 'category': '', 'difficulty': ''}, 'rerank_score': 0.9602475162305351, 'tool': 'postgres', 'similarity': 0.6897258758544922}], 'local_results': [{'id': 'history_003_0', 'content': '元朝多了豇豆、...，则往往显得单调而匮乏。', 'score': 0.737809419631958, 'metadata': {'recipe_id': 'history_003', 'name': '中国饮食文化历史资料 - 第3段', 'category': '历史文化', 'difficulty': ''}, 'rerank_score': 0.9741574735494072, 'tool': 'postgres', 'similarity': 0.737809419631958}, {'id': 'data.txt_4', 'content': '元朝多了豇豆、...，则往往显得单调而匮乏。', 'score': 0.6897258758544922, 'metadata': {'recipe_id': 'data.txt', 'name': '', 'category': '', 'difficulty': ''}, 'rerank_score': 0.9602475162305351, 'tool': 'postgres', 'similarity': 0.6897258758544922}], 'steps': ['guardrails', 'router', 'local_search'], 'sources': []}
        kb_logger.info("第四步：本地检索，state: {}", state)

        if state.get("guardrails_decision") == "end":
            summary = state.get("summary") or "抱歉，该问题暂时无法回答。"
            return {"answer": summary, "sources": [], "steps": ["finalize"]}

        milvus_results = state.get("milvus_results", [])
        postgres_results = state.get("postgres_results", [])
        local_results = state.get("local_results", []) or (milvus_results + postgres_results)
        external_results = state.get("external_results", [])

        milvus_context = _format_milvus_results(milvus_results)
        postgres_context = _format_postgres_results(postgres_results)
        local_context = _format_combined_local_results(local_results)
        external_context = _format_external_results(external_results)

        sources = _collect_sources(milvus_results, postgres_results, external_results)
        # ['history_003_0', 'data.txt_4']
        kb_logger.info("第四步：本地检索，sources: {}", sources)

        # 如果本地和外部检索结果都为空，则先做一次 Milvus 直接检索兜底
        if not local_results and not external_results:
            fb = await _search_local_kb(state.get("question", ""))
            if fb:
                local_results = fb
                local_context = _format_combined_local_results(fb)
                sources = _collect_sources(fb)
                kb_logger.info("图谱未命中 -> Milvus 兜底检索到 {} 条", len(fb))
            else:
                fallback = "抱歉，菜谱文化知识库暂未找到相关记载，请尝试描述得更具体一些或稍后再试。"
                return {"answer": fallback, "sources": sources, "steps": ["finalize"]}

        # 构建 最终 prompt
        messages = final_prompt.format_messages(
            question=state.get("question", ""),
            milvus_context=milvus_context,
            postgres_context=postgres_context,
            external_context=external_context,
        )

        try:
            response = await llm.ainvoke(messages, config=config)
            # content='明朝时期，中国的饮食文化经历了显著的变化，...，明朝的饮食文化因此呈现出更加丰富和多元的特征。\n\n引用来源：PostgreSQL#1, PostgreSQL#2' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 310, 'prompt_tokens': 1166, 'total_tokens': 1476, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_eb37e061ec', 'finish_reason': 'stop', 'logprobs': None} id='run--52fdbc22-83c3-49fa-8d52-c3ab7a19de9b-0' usage_metadata={'input_tokens': 1166, 'output_tokens': 310, 'total_tokens': 1476, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
            kb_logger.info("第四步：本地检索，LLM 回答: {}", response)
            content = getattr(response, "content", None)
            if isinstance(content, str):
                answer = content.strip()
            else:
                answer = str(response)
        except Exception as exc:  # pragma: no cover - defensive logging
            kb_logger.error("Failed to synthesise KB answer: {}", exc)
            answer = local_context if local_context and local_context != "（无本地检索结果）" else ""
            if not answer:
                answer = "检索已完成，但当前无法生成可靠的菜谱文化回答。"

        if not answer:
            answer = "检索已完成，但当前无法生成可靠的菜谱文化回答。"

        if sources:
            sources = list(dict.fromkeys(sources))

        return {
            "answer": answer,
            "sources": sources,
            "steps": ["finalize"],
        }

    def guardrails_router(state: KBWorkflowState) -> str:
        return "finalize" if state.get("guardrails_decision") == "end" else "kb_router"

    def router_edge(state: KBWorkflowState) -> str:
        return "external_search" if state.get("route") == "external" else "local_search"

    def local_edge(state: KBWorkflowState) -> str:
        route = state.get("route", "local")
        if route in {"hybrid", "external"} and allow_external_search and external_url:
            return "external_search"
        return "finalize"

    graph_builder = StateGraph(
        KBWorkflowState,
        input=KBInputState,
        output=KBOutputState,
    )

    graph_builder.add_node("guardrails", guardrails)
    graph_builder.add_node("kb_router", router)
    graph_builder.add_node("local_search", local_search)
    graph_builder.add_node("external_search", external_search)
    graph_builder.add_node("finalize", finalize)

    graph_builder.add_edge(START, "guardrails")
    graph_builder.add_conditional_edges("guardrails", guardrails_router)
    graph_builder.add_conditional_edges("kb_router", router_edge)
    graph_builder.add_conditional_edges("local_search", local_edge)
    graph_builder.add_edge("external_search", "finalize")
    graph_builder.add_edge("finalize", END)

    return graph_builder.compile()
