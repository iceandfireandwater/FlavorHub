from gustobot.application.agents.lg_prompts import (
    ROUTER_SYSTEM_PROMPT,
    GET_ADDITIONAL_SYSTEM_PROMPT,
    GENERAL_QUERY_SYSTEM_PROMPT,
    GET_IMAGE_SYSTEM_PROMPT,
    GUARDRAILS_SYSTEM_PROMPT,
    RAGSEARCH_SYSTEM_PROMPT,
    CHECK_HALLUCINATIONS,
    GENERATE_QUERIES_SYSTEM_PROMPT,
    IMAGE_GENERATION_ENHANCE_PROMPT,
    IMAGE_GENERATION_SUCCESS_PROMPT
)
from langchain_core.runnables import RunnableConfig
from gustobot.config import settings
from gustobot.infrastructure.core.logger import get_logger
from typing import cast, Literal, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from gustobot.application.agents.lg_states import AgentState, InputState, Router, GradeHallucinations
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.recipe_retriever import \
    RecipeCypherRetriever
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.planner.node import create_planner_node
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.workflows.multi_agent.multi_tool import (
    create_multi_tool_workflow,
    create_kb_multi_tool_workflow,
)
from gustobot.application.agents.kg_sub_graph.kg_neo4j_conn import get_neo4j_graph
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langchain_core.runnables.base import Runnable
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.utils.utils import \
    retrieve_and_parse_schema_from_graph_for_prompts
from langchain_core.prompts import ChatPromptTemplate
import base64
import os
import aiohttp
import asyncio
import json
import time
from pathlib import Path
from PIL import Image
import io

from langchain_openai import ChatOpenAI
from gustobot.application.agents.kb_tools import create_knowledge_query_node, KnowledgeQueryInputState
from gustobot.infrastructure.knowledge import KnowledgeService

class AdditionalGuardrailsOutput(BaseModel):
    """
    格式化输出，用于判断用户的问题是否与图谱内容相关
    """
    decision: Literal["end", "proceed", "continue"] = Field(
        description="Decision on whether the question is related to the graph contents."
    )


# 构建日志记录器
logger = get_logger(service="lg_builder")


def _message_text(msg: Any) -> str:
    """取一条消息的纯文本（兼容 BaseMessage 与 {"role":..,"content":..} 字典）。"""
    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item if isinstance(item, str) else str(item.get("text", ""))
            for item in content
        )
    return str(content)


def _max_prompt_chars() -> int:
    import os as _os

    try:
        return int(_os.getenv("GUSTOBOT_MAX_PROMPT_CHARS", "16000"))
    except (TypeError, ValueError):
        return 16000


def build_llm_messages(system_prompt: str, state: Any, *, max_chars: Optional[int] = None) -> List[Any]:
    """统一的 LLM 消息组装入口（所有面向用户的生成节点都必须走这里）。

    顺序：[system 提示] + [会话记忆摘要(若有)] + 窗口内的对话历史。
    - 摘要统一提到最前，避免 SystemMessage 夹在对话中间（部分 provider 不接受）；
    - 末尾做一次字符预算兜底裁剪，保证任何情况下都不会把整段历史原样灌给模型；
      裁剪从最旧的消息开始，但至少保留最后一条（当前用户提问）。
    """
    head: List[Any] = [{"role": "system", "content": system_prompt}]

    memory = getattr(state, "memory", None) or {}
    if memory:
        from gustobot.application.agents.memory import render_memory

        rendered = render_memory(memory)
        if rendered:
            head.append({"role": "system", "content": rendered})

    body: List[Any] = list(getattr(state, "messages", None) or [])
    budget = max_chars if max_chars is not None else _max_prompt_chars()
    total = sum(len(_message_text(m)) for m in head) + sum(len(_message_text(m)) for m in body)
    while len(body) > 1 and total > budget:
        total -= len(_message_text(body.pop(0)))
    return head + body


def _ensure_router(router_obj: Any, *, fallback_question: str = "") -> Router:
    """将任意 router 结构转换为 Router 模型，保持字段访问兼容。"""
    if isinstance(router_obj, Router):
        return router_obj
    if isinstance(router_obj, dict):
        try:
            return Router.model_validate(router_obj)  # 按Rounter类的标准，校验router_obj这个数据
        except Exception:
            pass
    return Router(type="kb-query", logic="missing router", question=fallback_question)


def _extract_configurable(config: Any) -> Dict[str, Any]:
    """提取 LangGraph RunnableConfig 中的 configurable 字段，确保返回字典。"""
    if not config:
        return {}
    if isinstance(config, dict):
        value = config.get("configurable", {})
        return value if isinstance(value, dict) else {}
    # LangGraph RunnableConfig 支持属性访问或字典接口
    configurable = getattr(config, "configurable", None)
    if isinstance(configurable, dict):
        return configurable
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            value = getter("configurable", {})
            if isinstance(value, dict):
                return value
        except Exception:  # pragma: no cover - 容错
            pass
    return {}

def _coerce_to_bool(value: Any, *, default: bool = False) -> bool:
    """Best-effort conversion of dynamic configuration values to boolean."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


async def analyze_and_route_query(
        state: AgentState, *, config: RunnableConfig
    ) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    # AgentState(messages=[HumanMessage(content='今天天气真好，你觉得呢？', additional_kwargs={}, response_metadata={}, id='10bbf527-569a-4c8f-934a-bc1aae06e6cb')], router=Router(logic='', type='general-query', question='', decision=None, confidence=None, reasoning=None), steps=[], documents=[], question='', answer='', hallucination=GradeHallucinations(binary_score='0'), sources=[])
    # logger.info(f"输入参数 state: {state}")
    logger.info("gustobot\application\agents\lg_builder.py------------analyze_and_route_query()---begin")

    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured for router analysis.")

    model = ChatOpenAI(streaming=True, 
        openai_api_key=settings.OPENAI_API_KEY,
        model_name=settings.OPENAI_MODEL,
        openai_api_base=settings.OPENAI_API_BASE,
        temperature=0.7,
        tags=["router"],
    )

    # 拼接提示模版 + 用户的实时问题（包含历史上下文对话）
    messages = build_llm_messages(ROUTER_SYSTEM_PROMPT, state)
    logger.info("-----Analyze user query type-----")
    logger.info(f"History messages: {state.messages}")
    # [HumanMessage(content='今天天气真好，你觉得呢？', additional_kwargs={}, response_metadata={}, id='afa2dcdc-23da-4804-8e82-3292d6bda714')]

    # 获取最近一次用户问题
    question_text = state.messages[-1].content if state.messages else ""

    # 在这里，根据关键词硬匹配，分出 graph_rag_query 和 sql_rag_query
    heuristic_router = _heuristic_router(question_text)
    
    # 兜底机制，如果上一步无效，则使用默认的 kb_query
    fallback_router: Router = heuristic_router or Router(
        type="kb-query",
        logic="fallback: default to knowledge base routing",
        question=question_text,
    )

    allowed_types: set[str] = {
        "general-query",
        "additional-query",
        "kb-query",
        "graphrag-query",
        "image-query",
        "file-query",
        "text2sql-query",
    }

    try:
        # 在这里确定了用户询问的类型！！！LLM 必须返回符合 Router 结构的数据
        # with_structured_output(Router)：要求LLM按照Router结构输出，并自动解析为Router对象
        # 1. 把传入的 Router 类用 model_json_schema() 转换为 JSON schema
        # 2. 底层通过 function calling 让 LLM 以参数格式输出 JSON
        # 3. 用 PydanticOutputParser + model_validate() 校验解析成 Router 实例
        structured_model = model.with_structured_output(Router)
        raw_response = await structured_model.ainvoke(messages)
        logger.info(f"Router LLM response: {raw_response}")
        # logic='' type='general-query' question='' decision=None confidence=None reasoning=None
        # 这是 Router 类的实例
    except Exception as exc:
        logger.warning("Router LLM failed: %s. Falling back to KB query.", exc)
        return {"router": fallback_router}

    response = raw_response if isinstance(raw_response, Router) else Router.model_validate(raw_response)
    router_type = response.type  # 获取路由类型，general-query
    logic = response.logic or ""

    # 兜底，如果 LLM 输出无效类型，则使用启发式路由（text2sql-query 或 graphrag-query），若启发式路由也无效，则使用 kb-query
    if not router_type or router_type not in allowed_types:
        logger.warning(
            "Router returned invalid type `%s`; applying heuristic fallback.", router_type
        )
        heuristic_router = _heuristic_router(question_text)  # text2sql-query 或 graphrag-query
        if heuristic_router:
            sanitized = heuristic_router
            if not sanitized.logic:
                sanitized.logic = logic or ""
            return {"router": sanitized}
        return {
            "router": Router(
                type="kb-query",
                logic=logic or "fallback: invalid router output",
                question=question_text,
            )
        }

    sanitized_router = Router(
        type=router_type,  # general-query
        logic=logic,  # logic=''
        question=response.question or question_text,  # question='今天天气真好，你觉得呢？'
        decision=response.decision,  # decision=None
        confidence=response.confidence,  # confidence=None
        reasoning=response.reasoning,  # reasoning=None
    )

    # Heuristic router is only used when the LLM output is invalid (handled above).
    # logic='' type='general-query' question='今天天气真好，你觉得呢？' decision=None confidence=None reasoning=None
    # logger.info(f"Analyze user query type completed, result: {sanitized_router}")
    logger.info("gustobot\application\agents\lg_builder.py------------analyze_and_route_query()---end")
    return {"router": sanitized_router}


def route_query(
        state: AgentState,
    ) -> Literal[
    "respond_to_general_query", "get_additional_info", "create_research_plan", "create_image_query", "create_file_query", "create_kb_query"]:
    """根据查询分类确定下一步操作。

    Args:
        state (AgentState): 当前代理状态，包括路由器的分类。

    Returns:
        Literal["respond_to_general_query", "get_additional_info", "create_research_plan", "create_image_query", "create_file_query"，"create_kb_query"]: 下一步操作。
    """
    logger.info("gustobot\application\agents\lg_builder.py------------route_query()---begin")
    router = _ensure_router(getattr(state, "router", None), fallback_question=state.messages[-1].content if state.messages else "")
    state.router = router
    _type = router.type or "kb-query"
    print(f"[DEBUG] Router type: {_type}")  # Debug: show which route the query takes
    logger.info("gustobot\application\agents\lg_builder.py------------route_query()---end")

    # 检查配置中是否有图片或文件路径，如果有，优先对应处理
    if hasattr(state, "config") and state.config:
        cfg = state.config.get("configurable", {})
        if cfg.get("image_path"):
            logger.info("检测到图片路径，转为图片查询处理")
            return "create_image_query"
        if cfg.get("file_path"):
            logger.info("检测到文件路径，转为文件上传处理")
            return "create_file_query"

    if _type == "general-query":
        return "respond_to_general_query"
    elif _type == "additional-query":
        return "get_additional_info"
    elif _type in ("graphrag-query", "text2sql-query"):  # 图查询或结构化问数
        return "create_research_plan"
    elif _type == "image-query":
        return "create_image_query"
    elif _type == "file-query":
        return "create_file_query"
    elif _type=="kb-query":
        return "create_kb_query"
    else:
        raise ValueError(f"Unknown router type {_type}")


async def respond_to_general_query(
        state: AgentState, *, config: RunnableConfig
    ) -> Dict[str, List[BaseMessage]]:
    """生成对一般查询的响应，完全基于大模型，不会触发任何外部服务的调用，包括自定义工具、知识库查询等。
    当路由器将查询分类为一般问题时，将调用此节点。
    Args:
        state (AgentState): 当前代理状态，包括对话历史和路由逻辑。
        config (RunnableConfig): 用于配置响应生成的模型。
    Returns:
        Dict[str, List[BaseMessage]]: 包含'messages'键的字典，其中包含生成的响应。
    """
    logger.info("-----generate general-query response-----")

    # 使用大模型生成回复
    model = ChatOpenAI(streaming=True, openai_api_key=settings.OPENAI_API_KEY, model_name=settings.OPENAI_MODEL,
                       openai_api_base=settings.OPENAI_API_BASE, temperature=0.7,
                       tags=["general_query"])

    router = _ensure_router(getattr(state, "router", None), fallback_question=state.messages[-1].content if state.messages else "")
    state.router = router

    # AgentState(messages=[HumanMessage(content='今天天气怎么样？', additional_kwargs={}, response_metadata={}, id='8fa742b3-f594-4960-a4c9-64ffabbeb321')], router=Router(logic='', type='general-query', question='今天天气怎么样？', decision=None, confidence=None, reasoning=None), steps=[], documents=[], question='', answer='', hallucination=GradeHallucinations(binary_score='0'), sources=[])
    logger.info(f"state: {state}")
    # logic='' type='general-query' question='今天天气怎么样？' decision=None confidence=None reasoning=None
    logger.info(f"Router: {router}")

    system_prompt = GENERAL_QUERY_SYSTEM_PROMPT.format(
        logic=router.logic
    )

    messages = build_llm_messages(system_prompt, state)
    response = await model.ainvoke(messages, config=config)
    return {"messages": [response]}

#大模型生成输出多了一些额外消息
async def get_additional_info(
        state: AgentState, *, config: RunnableConfig
    ) -> Dict[str, List[BaseMessage]]:
    """生成一个响应，要求用户提供更多信息。

    当路由确定需要从用户那里获取更多信息时，将调用此函数。

    Args:
        state (AgentState): 当前代理状态，包括对话历史和路由逻辑。
        config (RunnableConfig): 用于配置响应生成的模型。

    Returns:
        Dict[str, List[BaseMessage]]: 包含'messages'键的字典，其中包含生成的响应。
    """
    logger.info("gustobot\application\agents\lg_builder.py------------get_additional_info()---begin")
    logger.info("------continue to get additional info------")

    # 使用大模型生成回复
    model = ChatOpenAI(streaming=True, openai_api_key=settings.OPENAI_API_KEY, model_name=settings.OPENAI_MODEL,
                       openai_api_base=settings.OPENAI_API_BASE, temperature=0.7,
                       tags=["additional_info"])
    # 如果用户的问题是菜谱相关，但与自己的业务无关，则需要返回"无关问题"

    # 首先连接 Neo4j 图数据库
    try:
        neo4j_graph = get_neo4j_graph()  # 获取 Neo4j 图数据库连接，创建一个 Neo4j 图数据库对象
        logger.info("success to get Neo4j graph database connection")
    except Exception as e:
        logger.error(f"failed to get Neo4j graph database connection: {e}")
        neo4j_graph = None

    # 定义菜谱助手服务范围（用户友好的业务描述）
    scope_description = """
    菜谱智能助手服务范围：为您提供全方位的烹饪指导和美食知识，包括但不限于：

    🍳 菜谱查询与制作指导
    - 各类中华料理的详细做法和烹饪技巧
    - 食材用量、烹饪时长、火候掌握
    - 分步骤的烹饪指导和小贴士

    🥬 食材知识与营养价值
    - 食材的营养成分和健康功效
    - 食材的选购、储存和处理方法
    - 食材之间的搭配和替代建议

    🌶️ 口味与烹饪技法
    - 各种口味特点（麻辣、酱香、清淡等）
    - 不同烹饪方法（炒、蒸、煮、炖、烤等）
    - 菜品分类（热菜、凉菜、汤品、主食等）

    💊 食疗养生建议
    - 食材的中医食疗功效
    - 季节性饮食调理建议
    - 特定人群的饮食注意事项

    暂不支持：政治、娱乐八卦、新闻时事、天气预报、网购推荐、医疗诊断等非烹饪美食相关内容。
    如遇此类问题，我会礼貌地引导您回到烹饪美食话题～
    """

    scope_context = (
        f"参考此范围描述来决策:\n{scope_description}"
        if scope_description is not None
        else ""
    )

    # 动态从 Neo4j 图表中获取图表结构
    graph_context = (
        f"\n参考图表结构来回答:\n{retrieve_and_parse_schema_from_graph_for_prompts(neo4j_graph)}"
        if neo4j_graph is not None
        else ""
    )

    message = scope_context + graph_context + "\nQuestion: {question}"

    logger.info(f"scope_context: {scope_context}")
    logger.info(f"graph_context: {graph_context}")
    logger.info(f"message: {message}")

    # 拼接提示模版
    full_system_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                GUARDRAILS_SYSTEM_PROMPT,
            ),
            (
                "human",
                (message),
            ),
        ]
    )

    # 构建格式化输出的 Chain， 如果匹配，返回 continue，否则返回 end
    # model 必须输出满足 AdditionalGuardrailsOutput 结构：decision[end, proceed, continue]
    guardrails_chain = full_system_prompt | model.with_structured_output(AdditionalGuardrailsOutput)

    # LLM 回复
    guardrails_output: AdditionalGuardrailsOutput = await guardrails_chain.ainvoke(
        {"question": state.messages[-1].content if state.messages else ""}
    )

    # 空值检查：如果 LLM 返回 None，默认为 proceed
    if guardrails_output is None:
        logger.warning("Guardrails returned None, defaulting to proceed")
        guardrails_output = AdditionalGuardrailsOutput(decision="proceed")

    # 根据格式化输出的结果，返回不同的响应
    # 返回 end，拒绝回复，返回固定提示
    if guardrails_output.decision == "end":
        logger.info("-----Fail to pass guardrails check-----")
        return {"messages": [AIMessage(content="厨友您好～抱歉哦，这个问题不太属于我们的菜谱范围呢，我主要帮您解答菜谱和烹饪方面的问题～😊")]}
    else:  # 返回 proceed 或 continue，继续回复
        logger.info("-----Pass guardrails check-----")
        router = _ensure_router(getattr(state, "router", None), fallback_question=state.messages[-1].content if state.messages else "")
        state.router = router
        system_prompt = GET_ADDITIONAL_SYSTEM_PROMPT.format(
            logic=router.logic
        )
        messages = build_llm_messages(system_prompt, state)  # system 提示 + 会话摘要 + 窗口内历史
        response = await model.ainvoke(messages, config=config)
        return {"messages": [response]}


async def _generate_image(user_query: str, state: AgentState) -> Dict[str, List[BaseMessage]]:
    """使用doubao-seedream-4.0 API生成图片

    Args:
        user_query: 用户的图片生成请求
        state: 当前代理状态

    Returns:
        包含生成图片信息的消息字典
    """
    try:
        # 步骤1: 使用LLM优化用户提示词
        model = ChatOpenAI(streaming=True, 
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=0.7
        )

        enhance_prompt = IMAGE_GENERATION_ENHANCE_PROMPT.format(user_query=user_query)
        enhance_messages = [{"role": "user", "content": enhance_prompt}]

        logger.info(f"Enhancing user prompt: {user_query}")

        # 用 LLM 写一段优化后的生成图片的提示词
        enhanced_response = await model.ainvoke(enhance_messages)

        enhanced_prompt = enhanced_response.content.strip()
        logger.info(f"Enhanced prompt: {enhanced_prompt}")

        # 步骤2: 调用CogView-4 API生成图片
        api_key = settings.IMAGE_GENERATION_API_KEY
        base_url = settings.IMAGE_GENERATION_BASE_URL
        model_name = settings.IMAGE_GENERATION_MODEL
        size = settings.IMAGE_GENERATION_SIZE

        if not api_key:
            logger.error("IMAGE_GENERATION_API_KEY not configured")
            return {"messages": [AIMessage(content="抱歉，图片生成服务配置不完整，无法生成图片。")]}

        # 构建API请求
        api_url = f"{base_url}/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "prompt": enhanced_prompt,
            "size": size
        }

        logger.info(f"Calling doubao-seedream-4.0 API: {api_url}")
        logger.info(f"Payload: model={model_name}, size={size}")

        # 异步HTTP请求，调用doubao-seedream-4.0 API
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload, headers=headers, timeout=60) as resp:
                if resp.status != 200:  # 若状态码不是200，即失败
                    error_text = await resp.text()
                    logger.error(f"doubao-seedream-4.0 API error: {resp.status} - {error_text}")
                    return {"messages": [AIMessage(content="抱歉，图片生成失败，请稍后再试。")]}

                result = await resp.json()
                logger.info(f"doubao-seedream-4.0 API response: {json.dumps(result, ensure_ascii=False)}")

        # 步骤3: 解析响应获取图片URL
        if "data" not in result or len(result["data"]) == 0:
            logger.error(f"doubao-seedream-4.0 API returned no image data: {result}")
            return {"messages": [AIMessage(content="抱歉，图片生成失败，请稍后再试。")]}

        image_url = result["data"][0].get("url", "")
        if not image_url:
            logger.error(f"doubao-seedream-4.0 API returned no image URL: {result}")
            return {"messages": [AIMessage(content="抱歉，图片生成失败，请稍后再试。")]}

        logger.info(f"Image generated successfully: {image_url}")

        # 步骤4: 提取菜名（简单处理）
        dish_name = "菜品"
        for keyword in ["宫保鸡丁", "红烧肉", "麻婆豆腐", "糖醋排骨", "鱼香肉丝"]:
            if keyword in user_query:
                dish_name = keyword
                break

        # 步骤5: 格式化成功响应
        success_message = IMAGE_GENERATION_SUCCESS_PROMPT.format(dish_name=dish_name)
        response_content = f"{success_message}\n\n图片链接: {image_url}"

        return {"messages": [AIMessage(content=response_content)]}

    except asyncio.TimeoutError:
        logger.error("CogView-4 API timeout")
        return {"messages": [AIMessage(content="抱歉，图片生成超时，请稍后再试。")]}
    except Exception as e:
        logger.error(f"Error generating image: {e}", exc_info=True)
        return {"messages": [AIMessage(content=f"抱歉，图片生成过程中出现错误：{str(e)}")]}


async def create_image_query(
        state: AgentState, *, config: RunnableConfig
    ) -> Dict[str, List[BaseMessage]]:
    """处理图片查询并生成描述回复

    Args:
        state (AgentState): 当前代理状态，包括对话历史
        config (RunnableConfig): 配置参数，包含线程ID等配置信息

    Returns:
        Dict[str, List[BaseMessage]]: 包含'messages'键的字典，其中包含生成的响应
    """
    logger.info("-----Handle Image Query-----")
    image_path = config.get("configurable", {}).get("image_path", None)
    user_query = state.messages[-1].content if state.messages else ""

    # 判断是图像识别还是图像生成
    generation_keywords = ["生成", "画", "创建", "制作图片", "做一张", "给我一张", "来一张"]
    is_generation = any(keyword in user_query for keyword in generation_keywords)

    # 情况1: 用户要求生成图片（没有上传图片，或明确要求生成）
    if is_generation and not image_path:
        logger.info(f"Image Generation Request: {user_query}")
        return await _generate_image(user_query, state)

    # 情况2: 用户上传了图片，进行识别
    if not image_path:
        logger.warning(f"User Upload Image Path is None for recognition")
        return {"messages": [AIMessage(content="抱歉，我无法查看这张图片，请重新上传。")]}

    if not Path(image_path).exists():
        logger.warning(f"User Upload Image Not Found: {image_path}")
        return {"messages": [AIMessage(content="抱歉，我无法查看这张图片，请重新上传。")]}

    # 获取视觉模型配置
    api_key = settings.VISION_API_KEY
    base_url = settings.VISION_BASE_URL
    vision_model = settings.VISION_MODEL

    if not api_key or not base_url or not vision_model:
        logger.error("Vision Model Configuration Not Complete")
        return {"messages": [AIMessage(content="抱歉，我无法查看这张图片，请重新上传。")]}

    logger.info(f"Using Vision Model: {vision_model} to process image: {image_path}")

    try:
        # 读取并压缩图片
        with Image.open(image_path) as img:
            # 设置最大尺寸
            max_size = 1024
            # 计算缩放比例
            width, height = img.size
            ratio = min(max_size / width, max_size / height)

            # 如果图片尺寸已经小于最大尺寸，不需要缩放
            if width <= max_size and height <= max_size:
                resized_img = img
            else:
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)

            # 转换为JPEG格式，并调整质量
            img_byte_arr = io.BytesIO()
            if resized_img.mode != 'RGB':
                resized_img = resized_img.convert('RGB')
            resized_img.save(img_byte_arr, format='JPEG', quality=85)
            img_byte_arr.seek(0)

            # 转换为base64
            image_data = base64.b64encode(img_byte_arr.read()).decode('utf-8')

            logger.info(
                f"Image Compressed, Original Size: {width}x{height}, New Size: {resized_img.width}x{resized_img.height}")

        # 构建API请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": vision_model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的菜谱图像分析助手。请详细分析图片中的内容，特别关注菜品名称、食材、烹饪方法、摆盘等细节。"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4000,
            "temperature": 0.7
        }

        # 发送API请求
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60  # 增加超时时间
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    image_description = result["choices"][0]["message"]["content"]
                    logger.info(f"Successfully processed image and generated description")
                    # 使用图片描述和用户问题生成最终回复
                    # 从lg_prompts导入菜谱助手模板

                    # 构建回复请求
                    model = ChatOpenAI(streaming=True, openai_api_key=settings.OPENAI_API_KEY, model_name=settings.OPENAI_MODEL,
                                       openai_api_base=settings.OPENAI_API_BASE, temperature=0.7,
                                       tags=["image_query"])
                    # 使用专门的图片查询提示模板
                    system_prompt = GET_IMAGE_SYSTEM_PROMPT.format(
                        image_description=image_description
                    )
                    messages = build_llm_messages(system_prompt, state)
                    response = await model.ainvoke(messages, config=config)
                    return {"messages": [response]}

                else:
                    error_text = await response.text()
                    logger.error(f"Vision API Request Failed: {response.status} - {error_text}")
                    return {"messages": [AIMessage(content=f"抱歉，我无法查看这张图片，请重新上传。")]}




    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {"messages": [AIMessage(content=f"抱歉，我无法查看这张图片，请重新上传。")]}


async def create_file_query(
        state: AgentState, *, config: RunnableConfig
    ) -> Dict[str, List[BaseMessage]]:
    """Handle user-provided files for ingestion."""

    logger.info("-----Found User Upload File-----")
    config_opts = _extract_configurable(config)
    file_path = config_opts.get("file_path")
    ingest_service_url = settings.INGEST_SERVICE_URL

    service = KnowledgeService()

    if not file_path:
        logger.warning("User Upload File Path is None")
        return {"messages": [AIMessage(content="请提供要处理的文件路径。")]}

    p = Path(file_path)
    if not p.exists() or not p.is_file():
        logger.warning("User Upload File Not Found: %s", file_path)
        return {"messages": [AIMessage(content="抱歉，未找到该文件，请确认路径是否正确。")]}

    try:
        suffix = p.suffix.lower()
        size_bytes = p.stat().st_size
        if size_bytes > settings.FILE_UPLOAD_MAX_MB * 1024 * 1024:
            return {"messages": [AIMessage(content=f"文件过大（>{settings.FILE_UPLOAD_MAX_MB}MB），请分割后重新上传。")]}

        # 如果是 excel 文件
        if suffix in {".xlsx", ".xls"}:
            # Excel must be handled by external ingestion service
            if not ingest_service_url:
                return {"messages": [AIMessage(content="未配置外部接入服务 INGEST_SERVICE_URL，无法处理 Excel 导入。")]}
            incremental_flag = _coerce_to_bool(
                config_opts.get("incremental"),
                default=settings.INGEST_INCREMENTAL_DEFAULT,
            )
            regenerate_flag = _coerce_to_bool(config_opts.get("regenerate"), default=False)
            payload = {
                "excel_path": str(p),
                "incremental": incremental_flag,
                "regenerate": regenerate_flag,
            }

            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{ingest_service_url.rstrip('/')}/api/ingest/excel",
                    json=payload,
                    timeout=60,
                ) as resp:
                    if resp.status not in (200, 202):
                        txt = await resp.text()
                        return {"messages": [AIMessage(content=f"外部 Excel 导入请求失败（{resp.status}）：{txt}")]}
            return {"messages": [AIMessage(content="已启动 Excel 导入（外部服务），完成后可直接检索或提问。")]}

        raw_text = ""
        if suffix in {".txt", ".md", ".csv", ".log"}:
            raw_text = p.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".json":
            import json
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            raw_text = json.dumps(data, ensure_ascii=False, indent=2)
        else:
            return {"messages": [AIMessage(content=f"暂不支持该文件类型：{suffix}。当前仅支持 .txt/.md/.json/.csv/.log/.xlsx/.xls。")]}

        import uuid
        doc_id = f"upload_{p.stem}_{uuid.uuid4().hex[:6]}"
        title = p.stem
        success = await service.add_document(
            doc_id=doc_id,
            title=title,
            content=raw_text,
            metadata={"category": "uploaded"},
        )
        if not success:
            return {"messages": [AIMessage(content="文件已读取，但保存到知识库失败，请稍后重试。")]}

        knowledge_node = create_knowledge_query_node(knowledge_service=service)
        user_question = state.messages[-1].content if state.messages else title
        input_state: KnowledgeQueryInputState = {
            "task": user_question,
            "context": {"top_k": 5},
            "steps": ["file_upload"],
        }
        kb_result = await knowledge_node(input_state)
        answer_text = kb_result.get("answer") or f"文件《{title}》已上传并加入知识库，可直接对我提问相关内容。"
        return {"messages": [AIMessage(content=answer_text)]}
    except Exception as exc:
        logger.exception("Failed to ingest uploaded file: %s", exc)
        return {"messages": [AIMessage(content="文件导入出现异常，请稍后再试或联系管理员。")]}

async def create_kb_query(
        state: AgentState, *, config: RunnableConfig
    ) -> Dict[str, List[BaseMessage]]:
    """Query the vector knowledge base (and optional external API) via a multi-agent workflow."""
    logger.info("------execute KB multi-tool query------")

    last_message = state.messages[-1].content if state.messages else ""
    if not last_message.strip():
        return {"messages": [AIMessage(content="请告诉我具体的问题，我才能帮您查询知识库。")]}

    config_opts = _extract_configurable(config)
    kb_top_k = config_opts.get("kb_top_k") or settings.KB_TOP_K  # 5

    kb_similarity_threshold = (
        config_opts.get("kb_similarity_threshold")  # 0.2
        if config_opts.get("kb_similarity_threshold") is not None
        else settings.KB_SIMILARITY_THRESHOLD
    )

    kb_filter_expr = config_opts.get("kb_filter_expr")

    knowledge_service: Optional[KnowledgeService] = None

    try:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured for KB multi-tool workflow.")

        llm = ChatOpenAI(streaming=True, 
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.OPENAI_MODEL,
            openai_api_base=settings.OPENAI_API_BASE,
            temperature=0.7,
            tags=["kb_multi_tool"],
        )

        knowledge_service = KnowledgeService()

        external_url = settings.KB_EXTERNAL_SEARCH_URL
        logger.info("是否允许外部检索：{}".format(settings.KB_ENABLE_EXTERNAL_SEARCH))
        logger.info(f"若允许，使用外部检索服务：{external_url}")

        if not external_url and settings.INGEST_SERVICE_URL:
            external_url = f"{settings.INGEST_SERVICE_URL.rstrip('/')}/api/search"

        workflow = create_kb_multi_tool_workflow(
            llm=llm,
            knowledge_service=knowledge_service,
            top_k=kb_top_k,
            similarity_threshold=kb_similarity_threshold,
            filter_expr=kb_filter_expr,
            allow_external=settings.KB_ENABLE_EXTERNAL_SEARCH,  # False
            external_search_url=external_url,
        )

        logger.info("workflow 已构建")

        history_payload = [
            {
                "role": getattr(msg, "type", "user"),
                "content": getattr(msg, "content", ""),
            }
            for msg in state.messages[:-1]
            if getattr(msg, "content", "").strip()
        ]

        response = await workflow.ainvoke(
            {
                "question": last_message,
                "history": history_payload,
                "sources": [],   # 每轮重置，避免上一轮来源经 checkpoint 累积下来
            },
            config=config,
        )

        logger.info("KB multi-tool workflow response: %s", response)
        answer_text = response.get("answer") or "检索完成，但暂时没有可以分享的结果。"
        sources = response.get("sources", [])

        # 创建包含sources的AIMessage
        ai_message = AIMessage(content=answer_text)
        # 将sources附加到消息的additional_kwargs中
        ai_message.additional_kwargs["sources"] = sources

        return {"messages": [ai_message], "sources": sources}
    except Exception as exc:
        logger.warning(str(exc))
        logger.warning("KB multi-tool workflow unavailable (%s); falling back to direct search.", exc)

    # Fallback: direct KB query
    if knowledge_service is None:
        knowledge_service = KnowledgeService()
    knowledge_node = create_knowledge_query_node(knowledge_service=knowledge_service)
    input_state: KnowledgeQueryInputState = {
        "task": last_message,
        "context": {
            "top_k": kb_top_k,
            "similarity_threshold": kb_similarity_threshold,
            "filter_expr": kb_filter_expr,
        },
        "steps": ["kb_query"],
    }
    result = await knowledge_node(input_state)
    answer_text = result.get("answer", "") or "抱歉，我暂时无法从知识库中找到答案。"
    return {"messages": [AIMessage(content=answer_text)]}

# 图工具 查询节点
async def create_research_plan(
        state: AgentState, *, config: RunnableConfig
    ) -> Dict[str, List[str] | str]:
    """通过查询本地图知识库回答客户问题，执行任务分解，创建分布查询计划。

    Args:
        state (AgentState): 当前代理状态，包括对话历史。
        config (RunnableConfig): 用于配置计划生成的模型。

    Returns:
        Dict[str, List[str] | str]: 包含'steps'键的字典，其中包含研究步骤列表。
    """
    logger.info("------execute local knowledge base query------")

    # 使用大模型生成查询/多跳、并行查询计划
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured for research plan generation.")

    model = ChatOpenAI(streaming=True, 
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,
        model_name=settings.OPENAI_MODEL,
        temperature=0.7,
        tags=["research_plan"],
    )

    # 初始化必要参数
    #  Neo4j图数据库连接 - 使用配置中的连接信息
    neo4j_graph=None
    try:
        neo4j_graph = get_neo4j_graph()
        logger.info("success to get Neo4j graph database connection")
    except Exception as e:
        logger.error(f"failed to get Neo4j graph database connection: {e}")

    #  创建菜谱场景的检索器实例，根据 Graph Schema创建 Cypher ， 优先生成对应问题的cypher模版 用来引导大模型生成正确的Cypher查询语句
    cypher_retriever = RecipeCypherRetriever()

    #  定义工具模式列表
    from gustobot.application.agents.kg_sub_graph.kg_tools_list import (
        cypher_query,
        predefined_cypher,
        microsoft_graphrag_query,
        text2sql_query,
    )
    tool_schemas: List[type[BaseModel]] = [
        cypher_query,
        predefined_cypher,
        microsoft_graphrag_query,
        text2sql_query,
    ]

    # 预定义的Cypher查询 为菜谱场景定义有用的查询
    # 是一个字典，键是Cypher查询名称，值是Cypher查询语句。
    from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.predefined_cypher.cypher_dict import \
        predefined_cypher_dict

    # 定义菜谱助手服务范围
    scope_description = """
    菜谱智能助手服务范围：为您提供全方位的烹饪指导和美食知识，包括但不限于：

    🍳 菜谱查询与制作指导
    - 各类中华料理的详细做法和烹饪技巧
    - 食材用量、烹饪时长、火候掌握
    - 分步骤的烹饪指导和小贴士

    🥬 食材知识与营养价值
    - 食材的营养成分和健康功效
    - 食材的选购、储存和处理方法
    - 食材之间的搭配和替代建议

    🌶️ 口味与烹饪技法
    - 各种口味特点（麻辣、酱香、清淡等）
    - 不同烹饪方法（炒、蒸、煮、炖、烤等）
    - 菜品分类（热菜、凉菜、汤品、主食等）

    💊 食疗养生建议
    - 食材的中医食疗功效
    - 季节性饮食调理建议
    - 特定人群的饮食注意事项

    暂不支持：政治、娱乐八卦、新闻时事、天气预报、网购推荐、医疗诊断等非烹饪美食相关内容。
    """

    # 创建多工具工作流
    multi_tool_workflow = create_multi_tool_workflow(
        llm=model,
        graph=neo4j_graph,
        tool_schemas=tool_schemas,
        predefined_cypher_dict=predefined_cypher_dict,  # 预定义的Cypher查询
        cypher_example_retriever=cypher_retriever,  # 获取 cypher 查询语句
        scope_description=scope_description,
        llm_cypher_validation=True,
    )

    # return multi_tool_workflow
    # 准备输入状态
    last_message = state.messages[-1].content if state.messages else ""
    input_state = {
        "question": last_message,
        "data": [],
        "history": [],
        "route_type": _ensure_router(getattr(state, "router", None)).type,
    }

    # 执行工作流
    response = await multi_tool_workflow.ainvoke(input_state, config=config)
    return {"messages": [AIMessage(content=response["answer"])]}

# 貌似没用
async def check_hallucinations(
        state: AgentState, *, config: RunnableConfig
    ) -> dict[str, Any]:
    """Analyze the user's query and checks if the response is supported by the set of facts based on the document retrieved,
    providing a binary score result.

    This function uses a language model to analyze the user's query and gives a binary score result.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured for hallucination checks.")

    model = ChatOpenAI(streaming=True, 
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,
        model_name=settings.OPENAI_MODEL,
        temperature=0.7,
        tags=["hallucinations"],
    )

    system_prompt = CHECK_HALLUCINATIONS.format(
        documents=state.documents,
        generation=state.messages[-1]
    )

    messages = [
                   {"role": "system", "content": system_prompt}
               ] + state.messages

    logger.info("---CHECK HALLUCINATIONS---")

    response = cast(GradeHallucinations, await model.with_structured_output(GradeHallucinations).ainvoke(messages))

    return {"hallucination": response}


_pg_pool = None          # AsyncConnectionPool（在应用 startup 时打开）
_pg_saver = None         # AsyncPostgresSaver


import json as _json  # noqa: E402

from psycopg.types.json import set_json_dumps, set_json_loads  # noqa: E402


def _tolerant_json_dumps(obj, **kwargs):
    """psycopg 的 JSON 编码器：未知类型（如 LangGraph 的 Send / Command）转字符串兜底。

    LangGraph 的 PostgresSaver 用 `Jsonb(copy)` 直接存 checkpoint，checkpoint 里的
    pending_sends 又是 Send 对象，而这条路径**不走 saver 的 serde**，所以必须在
    psycopg 层兜住，否则每轮对话都会在收尾时报错。
    """
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("default", str)
    return _json.dumps(obj, **kwargs)


set_json_dumps(_tolerant_json_dumps)
set_json_loads(_json.loads)

import pickle  # noqa: E402

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # noqa: E402


class _PickleFallbackSerde(JsonPlusSerializer):
    """JsonPlus 优先；遇到它处理不了的对象（如 LangGraph 的 Send / Command）回退 pickle。

    子图 tool_selection / edges 用 Send 做动态并行，这些对象进不了 JSON，
    而 Postgres checkpoint 的默认 serde 只认 JSON —— 会抛
    "Object of type Send is not JSON serializable"，且崩在回答已推完之后。
    """

    def dumps_typed(self, obj):  # type: ignore[override]
        try:
            return super().dumps_typed(obj)
        except Exception:  # noqa: BLE001
            return ("pickle", pickle.dumps(obj))

    def loads_typed(self, data):  # type: ignore[override]
        enc, payload = data
        if enc == "pickle":
            return pickle.loads(payload)
        return super().loads_typed(data)


def _build_checkpointer():
    """构造 LangGraph 的 checkpoint 存储。

    用 **Async**PostgresSaver：本项目全链路走 `graph.astream`，同步版 saver 会直接抛
    NotImplementedError。state（含 messages 与结构化 memory）会落库，**进程重启后
    上下文与记忆都还在**；依赖不可用时降级为 MemorySaver（只在内存里，重启即丢）。
    """
    global _pg_pool, _pg_saver
    import os as _os

    dsn = _os.getenv("LANGGRAPH_PG_DSN") or "postgresql://postgres:postgres@kb_postgres:5432/vector_db"
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

        _pg_pool = AsyncConnectionPool(
            conninfo=dsn, max_size=10, kwargs={"autocommit": True}, open=False
        )
        _pg_saver = AsyncPostgresSaver(_pg_pool, serde=_PickleFallbackSerde())
        logger.info("checkpointer = AsyncPostgresSaver（持久化，重启不丢上下文）")
        return _pg_saver
    except Exception as exc:  # noqa: BLE001
        logger.warning("AsyncPostgresSaver 不可用（{}），降级为 MemorySaver（重启会丢上下文）", exc)
        _pg_pool = None
        _pg_saver = None
        return MemorySaver()


async def prepare_checkpointer() -> None:
    """应用 startup 时调用：打开连接池并确保 checkpoint 表存在（幂等）。"""
    if _pg_pool is None or _pg_saver is None:
        return
    try:
        await _pg_pool.open()
        await _pg_saver.setup()
        logger.info("LangGraph checkpoint 连接池已就绪")
    except Exception as exc:  # noqa: BLE001
        logger.warning("打开 checkpoint 连接池失败: {}", exc)


checkpointer = _build_checkpointer()

# 定义状态图
builder = StateGraph(AgentState, input=InputState)
# 添加节点
builder.add_node(analyze_and_route_query) # 意图识别
builder.add_node(respond_to_general_query)#默认回复
builder.add_node(get_additional_info) # 图结构信息
builder.add_node("create_research_plan", create_research_plan)  # 这里是graphrag neo4j-query
builder.add_node(create_image_query)
builder.add_node(create_file_query)
builder.add_node(create_kb_query)


# 添加边
builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)

graph = builder.compile(checkpointer=checkpointer)

# png_bytes = graph.get_graph().draw_mermaid_png()
# output_path = Path(__file__).resolve().parent / "lg_builder_workflow.png"
# output_path.write_bytes(png_bytes)
# logger.info("工作流图已保存到 %s", output_path)
#
# try:
#     from IPython.display import Image as IPythonImage, display as ipython_display
# except ImportError:  # pragma: no cover - optional dependency
#     logger.info("IPython 未安装，跳过图像内联展示。")
# else:
#     ipython_display(IPythonImage(png_bytes))
def _heuristic_router(question: str) -> Optional[Router]:
    """Fallback routing based on simple keyword heuristics."""
    if not question:
        return None

    lowered = question.lower()

    graphrag_keywords = [
        "怎么做",
        "如何做",
        "做法",
        "步骤",
        "火候",
        "食材",
        "原料",
        "需要什么",
        "配料",
        "用什么",
    ]

    text2sql_keywords = ["统计", "多少", "总数", "数量", "排名"]

    if any(keyword in lowered for keyword in text2sql_keywords):
        return Router(
            type="text2sql-query",
            logic="keyword fallback: text2sql",
            question=question,
        )

    if any(keyword in lowered for keyword in graphrag_keywords):
        return Router(
            type="graphrag-query",
            logic="keyword fallback: graphrag",
            question=question,
        )

    return None


def build_supervisor_graph():
    """向后兼容的 Supervisor Graph 构建接口。"""
    return graph


async def safety_guardrails(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """兼容旧接口，复用 get_additional_info 的护栏逻辑。"""
    return await get_additional_info(state, config=config)
