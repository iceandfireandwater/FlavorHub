"""
Unified Chat API with Agent Integration

Provides a single endpoint for chat interactions with automatic routing.
"""
import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, status
from fastapi.responses import StreamingResponse
from loguru import logger

from pydantic import BaseModel, Field

from gustobot.application.agents.lg_builder import graph
from gustobot.application.agents.memory import prepare_context
from gustobot.application.agents.user_memory import load_user_memory, save_user_memory
from gustobot.application.agents.memory import extract_delta, merge_memory
from gustobot.config import settings
from gustobot.infrastructure.core.database import get_db
from gustobot.infrastructure.persistence.crud import chat_message, chat_session
from gustobot.interfaces.http.models.chat_message import ChatMessageCreate, ChatMessageResponse
from gustobot.interfaces.http.models.chat_session import ChatSessionCreate, ChatSessionResponse
from sqlalchemy.orm import Session


router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., description="User message", min_length=1, max_length=5000)
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field("default_user", description="User identifier")
    stream: bool = Field(False, description="Enable streaming response")
    image_path: Optional[str] = Field(None, description="Path to uploaded image file")
    file_path: Optional[str] = Field(None, description="Path to uploaded file")
    ingest_incremental: Optional[bool] = Field(
        None,
        description="Override whether Excel ingestion uses incremental mode (defaults to server setting)",
    )


class ChatResponse(BaseModel):
    """Chat response model"""
    message: str
    session_id: str
    message_id: str
    route: Optional[str] = None
    route_logic: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatStreamChunk(BaseModel):
    """Streaming response chunk"""
    type: str = Field(..., description="Chunk type: 'message', 'metadata', 'error', 'done'")
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    route: Optional[str] = None

# 获取或创建会话
def get_or_create_session(db: Session, session_id: Optional[str], user_id: str) -> str:
    """Get existing session or create new one"""
    logger.info("gustobot\interfaces\http\v1\chat.py------------get_or_create_session()---begin")
    if session_id:
        session = chat_session.get(db, id=session_id)
        if session:
            return session_id

    # Create new session
    new_session_id = str(uuid.uuid4())  # 生成 UUID 作为会话 ID
    session_data = ChatSessionCreate(
        id=new_session_id,
        user_id=user_id,
        title="新对话"  # 首条消息发出后由 LLM 概括成短标题
    )
    chat_session.create(db, obj_in=session_data)
    logger.info("gustobot\interfaces\http\v1\chat.py------------get_or_create_session()---end")
    return new_session_id

# 保存消息到 chat_messages 表，在这里完成数据的装配
async def save_message(db: Session, session_id: str, message: str, is_user: bool,
                       route: Optional[str] = None, metadata: Optional[Dict] = None):
    """Save message to database"""
    logger.info("gustobot\interfaces\http\v1\chat.py------------save_message()---begin")
    try:
        last_message = chat_message.get_latest_by_session(db, session_id=session_id)
        next_order_index = (last_message.order_index + 1) if last_message else 1

        message_metadata: Dict[str, Any] = {}
        if isinstance(metadata, dict):
            # Avoid persisting huge/unstable objects (e.g. full agent graph state).
            message_metadata.update({k: v for k, v in metadata.items() if k != "agent_state"})
        elif metadata is not None:
            message_metadata["metadata"] = str(metadata)

        if route:
            message_metadata["route"] = route

        # 合成完整消息 ← 这里组装所有字段！！！
        # {'session_id': '96ed2efe-aef5-4276-b6f7-fca1a5a55bf8', 'message_type': 'agent_response', 'content': '亲～感谢您的咨询～😊 不好意思呢，这个问题可能不太属于我们的菜谱范围，建议您可以查看天气预报应用或网站获取详细信息。如果您有任何关于菜谱或烹饪的问题，随时欢迎询问哦～🍲', 'message_metadata': '{"session_id": "96ed2efe-aef5-4276-b6f7-fca1a5a55bf8", "route": "general-query"}', 'order_index': 2}
        message_data = ChatMessageCreate(
            session_id=session_id,
            message_type="user_query" if is_user else "agent_response",
            content=message,
            message_metadata=message_metadata or None,
            order_index=next_order_index,
        )

        # 保存消息到数据库！！！
        created = chat_message.create(db, obj_in=message_data)

        # Update session activity timestamp so session list ordering stays correct.
        # 就是更新 chat_sessions 表的 activity_timestamp 字段
        chat_session.update_activity(db, session_id=session_id)
        logger.info("gustobot\interfaces\http\v1\chat.py------------save_message()---end")
        return str(created.id)
    except Exception as e:
        logger.error(f"Failed to save message: {e}")

        
        return None


async def _persist_memory(config: Dict[str, Any], user_id: str) -> None:
    """回答推完后，把**本轮**（含助手回复）再抽一次并存进长期记忆。

    为什么必须放这里：抽取原本只在"下一次请求开始时"跑，而那时本轮消息才刚进 state
    —— 结果"只说一句就结束"的会话永远抽不到（facts 用例库内全空的根因）。
    放流末尾还顺带避开首字延迟（用户已经看到回答了）。
    """
    try:
        snapshot = await graph.aget_state(config)
        values = getattr(snapshot, "values", None) or {}
        msgs = [m for m in (values.get("messages") or []) if getattr(m, "id", None)]
        if not msgs:
            return
        prev = load_user_memory(user_id) or {}
        delta = await extract_delta(msgs[-6:], None)
        if delta:
            save_user_memory(user_id, merge_memory(prev, delta))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"长期记忆沉淀失败（不影响回答）: {exc}")


async def _persist_assistant_message(
    session_id: str,
    content: str,
    route: Optional[str] = None,
    sources: Optional[list] = None,
    chain: Optional[Dict[str, Any]] = None,
) -> None:
    """把流式生成的完整回答落库。

    流式响应是在 generator 里跑完的，那时请求级 db 已经失效，所以这里开一个独立的
    SessionLocal。原先流式路径**完全没有保存助手消息**，导致刷新后历史里只剩提问。
    """
    if not session_id or not (content or "").strip():
        return
    try:
        from gustobot.infrastructure.core.database import SessionLocal

        db = SessionLocal()
        try:
            await save_message(
                db,
                session_id,
                content,
                is_user=False,
                route=route,
                metadata={"sources": sources or [], "chain": chain or {}},
            )
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"保存助手消息失败: {exc}")


def _router_attr(router_info: Any, name: str) -> Any:
    """从 Router（对象或 dict）里取字段。"""
    v = getattr(router_info, name, None)
    if v is None and isinstance(router_info, dict):
        v = router_info.get(name)
    return v


def _safe_sources(raw: Any) -> List[Dict[str, Any]]:
    """把 sources 归一化成可 JSON 序列化的结构。

    子图有时会把 LangGraph 的内部对象（如 Send）混进 sources，直接序列化会抛
    "Object of type Send is not JSON serializable"——而且崩在回答已推完之后，
    表现为"回答正常显示, 末尾多一条 [出错]"。这里只保留基本类型。
    """
    out: List[Dict[str, Any]] = []
    for item in raw or []:
        if isinstance(item, str):
            out.append({"document_id": item, "source": item})
        elif isinstance(item, dict):
            out.append({
                k: v for k, v in item.items()
                if v is None or isinstance(v, (str, int, float, bool))
            })
        else:
            out.append({"source": str(item)})
    return out


def _sse(chunk: "ChatStreamChunk") -> str:
    """把事件包成 SSE 行。"""
    return "data: " + chunk.model_dump_json() + chr(10) + chr(10)


def _chunk_text(chunk: Any) -> str:
    """从流式 chunk 取文本（兼容 str / list 形式的 content）。"""
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


# 内部节点的 token 不该直接吐给用户：router 是意图识别、hallucinations 是自检、
# research_plan 是图谱检索的计划文本。
# 整段重放检测：LangGraph 对子图(sumarize/finalize)的 token 会经父子两层各收集一次，
# 导致回答整体重复两遍。重放开始时，新内容的前 N 个字符必然与整段开头相同，
# 用这个特征识别并撤回（N 取大一点以避开正常文本的偶然巧合）。
_REPLAY_PROBE = 24

_INTERNAL_TAGS = {"router", "hallucinations", "research_plan"}
# create_research_plan / create_kb_query 是"容器节点"：它们内部的子图已经逐字
# 流过一遍，节点本身又把完整回答作为一条 message 流一次，会造成回答重复两遍。
_INTERNAL_NODES = {"analyze_and_route_query", "check_hallucinations",
                   "create_research_plan", "create_kb_query"}


def _should_stream(metadata: Dict[str, Any]) -> bool:
    tags = set(metadata.get("tags") or [])
    if tags & _INTERNAL_TAGS:
        return False
    if metadata.get("langgraph_node") in _INTERNAL_NODES:
        return False
    return True


_PLACEHOLDER_TITLES = ("", "新对话")


def _needs_title(session) -> bool:
    """会话还没有像样的标题 -> 需要生成。"""
    title = (getattr(session, "title", "") or "").strip()
    return title in _PLACEHOLDER_TITLES or title.startswith("Chat ")


async def _generate_session_title(session_id: str, first_message: str) -> None:
    """用 LLM 把首条用户消息概括成一句短标题，写回 chat_sessions.title。

    以 asyncio 后台任务运行，不阻塞流式首字；失败则保留原标题。
    """
    try:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            streaming=False,
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.OPENAI_MODEL,
            openai_api_base=settings.OPENAI_API_BASE,
            temperature=0.2,
        )
        prompt = (
            "把下面这句话概括成一个不超过 14 个字的短语，用作聊天记录的标题。"
            "只输出短语本身，不要标点、不要引号、不要解释。" + chr(10) + chr(10) + first_message
        )
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        title = (getattr(resp, "content", "") or "").strip()
        title = title.strip("\"'「」《》【】()（）。，,、：:;；!！?？ ")
        if not title:
            return
        title = title[:40]

        from gustobot.infrastructure.core.database import SessionLocal

        db = SessionLocal()
        try:
            session = chat_session.get(db, id=session_id)
            if session:
                keep_updated = session.updated_at   # 改标题不该刷新"最后活跃时间"
                session.title = title
                session.updated_at = keep_updated
                db.commit()
                logger.info(f"会话标题已生成: {session_id} -> {title}")
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"生成会话标题失败: {exc}")


def _schedule_title_generation(db: Session, session_id: str, message: str) -> None:
    """首条消息时，异步生成会话标题。"""
    try:
        session = chat_session.get(db, id=session_id)
        if session and _needs_title(session):
            asyncio.create_task(_generate_session_title(session_id, message))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"检查会话标题失败: {exc}")


async def _build_input_state(
    message: str,
    session_id: str,
    image_path: Optional[str] = None,
    file_path: Optional[str] = None,
    ingest_incremental: Optional[bool] = None,
    user_id: str = "default_user",
) -> tuple:
    """构造 (config, input_state)，其中已包含会话记忆压缩的结果。

    流式与非流式两条路径共用，避免“只有非流式才做记忆”的不一致。
    """
    incremental_flag = (
        settings.INGEST_INCREMENTAL_DEFAULT if ingest_incremental is None else bool(ingest_incremental)
    )
    config = {
        "configurable": {
            "thread_id": session_id,
            "image_path": image_path,
            "file_path": file_path,
            "incremental": incremental_flag,
        }
    }

    removals: List[Any] = []
    new_memory: Dict[str, Any] = {}
    try:
        snapshot = await graph.aget_state(config)
        values = getattr(snapshot, "values", None) or {}
        existing = list(values.get("messages", []) or [])
        # 长期记忆优先：新会话也能继承同一用户过往积累的约束/偏好/话题
        prev_memory = load_user_memory(user_id) or (values.get("memory", {}) or {})
        removals, new_memory = await prepare_context(existing, prev_memory)
        if new_memory and new_memory != prev_memory:
            save_user_memory(user_id, new_memory)   # 沉淀到长期记忆
        if removals:
            logger.info(
                "会话记忆：压缩 %d 条旧消息 -> 约束%d/偏好%d/菜名%d/问答%d/兜底%d"
                % (
                    len(removals),
                    len(new_memory.get("constraints", [])),
                    len(new_memory.get("preferences", {})),
                    len(new_memory.get("dishes", [])),
                    len(new_memory.get("answered", [])),
                    len(new_memory.get("notes", [])),
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"上下文整理失败，按未裁剪继续: {exc}")

    input_state: Dict[str, Any] = {
        "messages": [*removals, {"type": "human", "content": message}],
        # 每轮显式重置：父图 state 里没有任何节点负责清空 sources，
        # 走 graphrag/additional/general 路径时没人写它，上一轮的值就会一直留着
        #（表现为"每次都是那几个网址"，且 general 类问题也挂着来源）。
        "sources": [],
    }
    if new_memory:
        input_state["memory"] = new_memory
    return config, input_state


async def process_agent_query(
    message: str, session_id: str,
                            image_path: Optional[str] = None,
                            file_path: Optional[str] = None,
                            ingest_incremental: Optional[bool] = None,
                            user_id: str = "default_user",
) -> Dict[str, Any]:
    """Process query through agent system"""
    logger.info("gustobot\interfaces\http\v1\chat.py------------process_agent_query()---begin")
    # 会话记忆压缩 + config/input_state 构造（与流式路径共用同一份逻辑）
    config, input_state = await _build_input_state(
        message, session_id, image_path, file_path, ingest_incremental, user_id
    )

    # ── 会话记忆 ────────────────────────────────────────────────────────────
    # 把超出窗口的历史压缩成滚动摘要，并用 RemoveMessage 把旧消息真正从 state 删除。
    # 不做这一步：state.messages 会随轮数无限累加，checkpoint 与每轮的 prompt 一起膨胀，
    # 最终把模型上下文撑爆（MemorySaver 是进程内内存，只增不减）。


    try:
        # Invoke agent graph
        # langGraph 会根据路由类型选择节点
        result = await graph.ainvoke(input_state, config=config)
        await _persist_memory(config, user_id)
        # logger.info(f"gustobot\interfaces\http\v1\chat.py-------Agent result: {result}")
        # {'messages': [HumanMessage(content='我死了！', additional_kwargs={}, response_metadata={}, id='79285198-820c-4444-9e0f-342f75e7ae91'), AIMessage(content='亲～感谢您的咨询～😊 不好意思呢，这个表达可能让我有 些困惑。如果您有任何关于菜谱或烹饪的问题，随时欢迎询问哦～🍲希望能帮到您！', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 50, 'prompt_tokens': 628, 'total_tokens': 678, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_eb37e061ec', 'finish_reason': 'stop', 'logprobs': None}, id='run--d95dbf95-7447-41f5-9a98-d8e4aa7a4140-0', usage_metadata={'input_tokens': 628, 'output_tokens': 50, 'total_tokens': 678, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})], 'router': Router(logic='', type='general-query', question='我死了！', decision=None, confidence=None, reasoning=None)}

        # Extract response and metadata
        response_text = ""
        if result.get("messages"):
            response_text = result["messages"][-1].content  # '亲～感谢您的咨询～😊 不好意思呢，这个表达可能让我有 些困惑。如果您有任何关于菜谱或烹饪的问题，随时欢迎询问哦～🍲希望能帮到您！'

        # Extract route information
        router_info = result.get("router", {})  # Router(logic='', type='general-query', question='我死了！', decision=None, confidence=None, reasoning=None)
        route = router_info.get("type")  # general-query
        route_logic = router_info.get("logic")  # ''

        # Extract sources if available
        sources_raw = result.get("sources", [])

        # Convert sources to expected format (list of dicts)
        sources = []
        if sources_raw:
            # If sources is a list of strings, convert to list of dicts
            if isinstance(sources_raw[0], str):
                for src in sources_raw:
                    sources.append({"document_id": src, "source": src})
            else:
                # Already in correct format
                sources = sources_raw

        logger.info("gustobot\interfaces\http\v1\chat.py------------process_agent_query()---end")

        return {
            "message": response_text,  # LLM 的回复内容
            "route": route,  # 路由信息
            "route_logic": route_logic,  # 路由逻辑
            "sources": sources,  # 源信息
            "metadata": {
                "session_id": session_id,
                "agent_state": result
            }
        }
    except Exception as e:
        logger.error("Agent query failed: {}", str(e), exc_info=True)
        return {
            "message": "抱歉，处理您的请求时出现了错误。请稍后重试。",
            "route": "error",
            "route_logic": f"Error: {str(e)}",
            "sources": [],
            "metadata": {"error": str(e)}
        }

async def stream_agent_response(message: str, session_id: str,
                               image_path: Optional[str] = None,
                               file_path: Optional[str] = None,
                               ingest_incremental: Optional[bool] = None,
                            user_id: str = "default_user",
) -> AsyncGenerator[str, None]:
    """真实流式：把 LangGraph 的 LLM token 逐个推给前端。

    旧实现是假的：先 await 完整回答，再按空格拆分逐个 sleep 发送；中文没有空格，
    等于整段一次性发出。现在直接消费 graph.astream(stream_mode="messages")。
    """
    try:
        config, input_state = await _build_input_state(
            message, session_id, image_path, file_path, ingest_incremental, user_id
        )
    except Exception as exc:  # noqa: BLE001
        logger.opt(exception=True).error("构造上下文失败: {}", repr(exc))
        yield _sse(ChatStreamChunk(type="error", content=f"处理请求时出错: {exc}", session_id=session_id))
        yield _sse(ChatStreamChunk(type="done", session_id=session_id))
        return

    yield _sse(ChatStreamChunk(type="metadata", metadata={"status": "processing"}, session_id=session_id))

    buffer = ""
    replayed = False
    try:
        async for chunk, metadata in graph.astream(input_state, stream_mode="messages", config=config):
            if not _should_stream(metadata or {}):
                continue
            text = _chunk_text(chunk)
            if not text:
                continue
            buffer += text

            if replayed:
                continue

            if len(buffer) >= _REPLAY_PROBE * 2:
                # 若 buffer 的结尾恰好等于它自己的开头（连续 >= _REPLAY_PROBE 字相同），
                # 说明已经进入第二遍重放；正确内容就是被重复掉的那一段。
                limit = min(160, len(buffer) // 2)
                for k in range(_REPLAY_PROBE, limit + 1):
                    if buffer.endswith(buffer[:k]):
                        replayed = True
                        correct = buffer[: len(buffer) - k]
                        buffer = correct
                        logger.warning("检测到流式内容重放，已撤回重复段: 保留 {} 字", len(correct))
                        yield _sse(ChatStreamChunk(type="replace", content=correct, session_id=session_id))
                        break
                if replayed:
                    continue

            yield _sse(ChatStreamChunk(type="message", content=text, session_id=session_id))

        # 流结束后补路由/来源（这些只有跑完才知道）
        snapshot = await graph.aget_state(config)
        values = getattr(snapshot, "values", None) or {}
        router_info = values.get("router")
        route = getattr(router_info, "type", None)
        if route is None and isinstance(router_info, dict):
            route = router_info.get("type")
        logic = getattr(router_info, "logic", None)
        if logic is None and isinstance(router_info, dict):
            logic = router_info.get("logic")

        if not buffer.strip():   # 兜底：个别节点不经 messages 流返回
            msgs = values.get("messages") or []
            fallback = ""
            if msgs:
                last = msgs[-1]
                fallback = last.content if isinstance(last.content, str) else _chunk_text(last)
            if fallback:
                yield _sse(ChatStreamChunk(type="message", content=fallback, session_id=session_id))

        # 落库：把完整回答写进 chat_messages，否则刷新页面后历史里只剩提问
        # 来源取"本轮消息"上挂的（kb 路径在 ai_message.additional_kwargs 里写入），
        # 而不是 state["sources"] —— 后者会经 checkpoint 跨轮残留下来：父图没有任何
        # 节点负责清空它，走 graphrag/additional 路径时上一轮的值就原样留下
        # （表现为 general 类问题也挂着来源、且每次都是同样几个网址）。
        msgs = values.get("messages") or []
        turn_sources = []
        if msgs:
            turn_sources = (getattr(msgs[-1], "additional_kwargs", None) or {}).get("sources") or []
        # 只认本轮消息上的 sources。绝不 fallback 回 state["sources"]：
        # kb 轮会把 5 条写进 state，而其它路径没有任何节点清空它，于是同一会话里
        # 之后的每一轮（包括 general/additional/graphrag）都会挂着那批旧 URL。
        safe_sources = _safe_sources(turn_sources)
        # 调用链路：落库一份，刷新页面后历史回答下面也能看到
        chain = {
            "route": route,
            "confidence": _router_attr(values.get("router"), "confidence"),
            "steps": [str(x) for x in (values.get("steps") or [])],
            "sources": len(safe_sources),
        }
        await _persist_assistant_message(session_id, buffer, route, safe_sources, chain)

        yield _sse(ChatStreamChunk(
            type="metadata", metadata={"route": route, "logic": logic}, session_id=session_id, route=route))
        await _persist_memory(config, user_id)   # 本轮也纳入长期记忆（不占首字延迟）
        chain["logic"] = (str(_router_attr(values.get("router"), "logic") or ""))[:120]
        yield _sse(ChatStreamChunk(
            type="done", metadata={"sources": safe_sources, "chain": chain}, session_id=session_id))

    except Exception as e:  # noqa: BLE001
        import traceback as _tb
        logger.error("流式生成失败: {} || {}", repr(e), _tb.format_exc())
        yield _sse(ChatStreamChunk(type="error", content=f"处理请求时出错: {str(e)}", session_id=session_id))
        yield _sse(ChatStreamChunk(type="done", session_id=session_id))
# 指定这个端点返回的数据格式为 ChatResponse，接口为 /api/v1/chat/
@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,  # 前端传来的请求数据
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)  # 获得数据库会话，可用来执行查询、添加、更新和删除数据等操作
) -> ChatResponse:
    """
    Unified chat endpoint with automatic agent routing

    - Automatically routes queries to appropriate agents
    - Maintains conversation history
    - Supports file uploads and images
    """
    logger.info("gustobot\interfaces\http\v1\chat.py------------chat()---begin")
    # Get or create session
    session_id = get_or_create_session(db, request.session_id, request.user_id)

    # Save user message
    await save_message(db, session_id, request.message, is_user=True)

    # 首条消息 -> 后台生成一个内容摘要式标题
    _schedule_title_generation(db, session_id, request.message)

    # Process through agent
    effective_incremental = (
        request.ingest_incremental
        if request.ingest_incremental is not None
        else settings.INGEST_INCREMENTAL_DEFAULT
    )

    result = await process_agent_query(
        request.message,
        session_id,
        request.image_path,
        request.file_path,
        effective_incremental,
        request.user_id,
    )

    # Save assistant message
    message_id = await save_message(
        db,
        session_id,
        result["message"],
        is_user=False,
        route=result["route"],
        metadata=result.get("metadata")
    )

    logger.info("gustobot\interfaces\http\v1\chat.py------------chat()---end")

    return ChatResponse(
        message=result["message"],
        session_id=session_id,
        message_id=message_id or str(uuid.uuid4()),
        route=result["route"],
        route_logic=result["route_logic"],
        sources=result.get("sources"),
        metadata=result.get("metadata")
    )


# ---------------------------------------------------------------------------
# Legacy alias routes (backwards compatibility)
#
# Older docs/scripts use `/api/v1/chat/chat` and `/api/v1/chat/chat/stream`.
# Keep them working to reduce migration friction.
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse, include_in_schema=False)
async def chat_legacy(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> ChatResponse:
    return await chat(request, background_tasks, db)


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    db: Session = Depends(get_db)
) -> StreamingResponse:
    """
    Streaming chat endpoint with automatic agent routing

    Returns responses in Server-Sent Events (SSE) format
    """
    # Get or create session
    session_id = get_or_create_session(db, request.session_id, request.user_id)

    # Save user message
    await save_message(db, session_id, request.message, is_user=True)

    # 首条消息 -> 后台生成一个内容摘要式标题
    _schedule_title_generation(db, session_id, request.message)

    effective_incremental = (
        request.ingest_incremental
        if request.ingest_incremental is not None
        else settings.INGEST_INCREMENTAL_DEFAULT
    )

    # Return streaming response
    return StreamingResponse(
        stream_agent_response(
            request.message,
            session_id,
            request.image_path,
            request.file_path,
            effective_incremental,
            request.user_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.post("/chat/stream", include_in_schema=False)
async def chat_stream_legacy_post(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    return await chat_stream(request, db)


@router.get("/chat/stream", include_in_schema=False)
async def chat_stream_legacy_get(
    message: str = Query(..., min_length=1, max_length=5000),
    session_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query("default_user"),
    image_path: Optional[str] = Query(None),
    file_path: Optional[str] = Query(None),
    ingest_incremental: Optional[bool] = Query(None),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    request = ChatRequest(
        message=message,
        session_id=session_id,
        user_id=user_id,
        stream=True,
        image_path=image_path,
        file_path=file_path,
        ingest_incremental=ingest_incremental,
    )
    return await chat_stream(request, db)


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    db: Session = Depends(get_db),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0)
):
    """Get chat history for a session.

    不加返回类型注解：FastAPI 会用 `-> List[ChatMessageResponse]` 当 response_model，
    而该模型要求 order_index，会把这里显式构造的 dict 过滤掉（500）。
    """
    # Verify session exists
    session = chat_session.get(db, id=session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Get messages
    messages = chat_message.get_by_session(
        db,
        session_id=session_id,
        skip=offset,
        limit=limit
    )

    # 显式带上 message_metadata：刷新页面后要靠它恢复 sources 与调用链路
    return [
        {
            "id": m.id,
            "session_id": m.session_id,
            "message_type": m.message_type,
            "content": m.content,
            "route": getattr(m, "route", None),
            "message_metadata": getattr(m, "message_metadata", None) or {},
            "created_at": m.created_at,
        }
        for m in messages
    ]


@router.delete("/session/{session_id}")
async def clear_session(
    session_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Clear all messages in a session
    """
    session = chat_session.get(db, id=session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Delete all messages in session
    chat_message.delete_by_session(db, session_id=session_id)

    return {"message": "Session cleared successfully", "session_id": session_id}


@router.get("/routes")
async def get_route_info() -> Dict[str, Any]:
    """
    Get information about available routes and their purposes
    """
    return {
        "routes": {
            "general-query": {
                "name": "日常对话",
                "description": "处理问候、寒暄等日常对话",
                "examples": ["你好", "谢谢", "今天天气不错"]
            },
            "additional-query": {
                "name": "补充信息",
                "description": "当问题模糊时，询问更多信息",
                "examples": ["我想做菜", "帮我推荐一道菜"]
            },
            "kb-query": {
                "name": "知识库查询",
                "description": "查询历史文化、典故等内容",
                "examples": ["宫保鸡丁的历史", "川菜的特点"]
            },
            "graphrag-query": {
                "name": "图谱查询",
                "description": "查询做法、食材、烹饪技巧",
                "examples": ["红烧肉怎么做", "需要什么食材"]
            },
            "text2sql-query": {
                "name": "统计查询",
                "description": "统计分析、计数、排名",
                "examples": ["有多少道菜", "最受欢迎的菜"]
            },
            "image-query": {
                "name": "图片处理",
                "description": "生成或分析图片",
                "examples": ["生成一张红烧肉的图片"]
            },
            "file-query": {
                "name": "文件处理",
                "description": "处理上传的菜谱文件",
                "examples": ["分析这个菜谱文档"]
            }
        },
        "auto_routing": "系统会根据您的问题自动选择合适的处理方式"
    }
