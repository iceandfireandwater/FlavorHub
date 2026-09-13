"""GustoBot 多层级会话记忆 —— 超窗历史压缩为长期摘要。

替代原先"超窗直接丢弃"的窗口裁剪：
- 窗口内（最近 N 轮）：完整保留，供上下文引用
- 窗口外：由 LLM 压缩为长期摘要注入会话，保留关键事实（菜名/食材/口味/偏好/已问问题）
- LLM 不可用时降级为原窗口裁剪行为

接入点：gustobot/application/agents/main.py（CLI）、API 会话层可复用。
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from langchain_core.messages import BaseMessage, SystemMessage

logger = logging.getLogger(__name__)

MEMORY_SUMMARY_PREFIX = "[长期记忆摘要] "
SUMMARY_PROMPT = (
    "你是会话记忆压缩器。请将以下对话历史压缩为一段中文要点摘要（不超过 200 字），"
    "必须保留：提到的菜名、食材、口味偏好、用户需求、已经问过的问题与得到的答案要点。"
    "只输出摘要正文，不要任何前缀或解释。\n\n对话历史：\n{history}"
)


def select_messages_to_compress(existing: Sequence[BaseMessage], limit: Optional[int]) -> List[BaseMessage]:
    """与 main.py 原 _select_messages_to_remove 一致：返回超窗需要处理的旧消息。"""
    if not existing or limit is None:
        return []
    humans_to_keep = max(limit - 1, 0)
    if humans_to_keep == 0:
        return [msg for msg in existing if getattr(msg, "id", None)]
    humans_seen = 0
    keep_from = 0
    for index in range(len(existing) - 1, -1, -1):
        message = existing[index]
        if getattr(message, "type", None) == "human":
            humans_seen += 1
            if humans_seen == humans_to_keep:
                keep_from = index
                break
    else:
        keep_from = 0
    return [msg for msg in existing[:keep_from] if getattr(msg, "id", None)]


def _stringify(msg: BaseMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


async def _summarize(messages: Sequence[BaseMessage], llm=None) -> str:
    """LLM 摘要压缩；失败返回空串（调用方降级为直接删除）。"""
    try:
        if llm is None:
            from langchain_core.messages import HumanMessage
            from langchain_openai import ChatOpenAI
            from gustobot.config.settings import settings
            llm = ChatOpenAI(
                openai_api_key=settings.OPENAI_API_KEY,
                model_name=settings.OPENAI_MODEL,
                openai_api_base=settings.OPENAI_API_BASE,
                temperature=0.2,
            )
        history = "\n".join(
            f"{'用户' if msg.type == 'human' else '助手'}: {_stringify(msg)}"
            for msg in messages
        )
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
        prompt_messages = await prompt.ainvoke({"history": history})
        if hasattr(llm, "ainvoke"):
            resp = await llm.ainvoke(prompt_messages)
        else:
            resp = llm.invoke(prompt_messages)
        text = _stringify(resp)
        return text.strip()
    except Exception as exc:  # noqa
        logger.warning("记忆摘要生成失败，降级为直接裁剪: %s", exc)
        return ""


async def compress_history(
    existing: Sequence[BaseMessage],
    limit: Optional[int],
    llm=None,
) -> Tuple[List[BaseMessage], List[BaseMessage]]:
    """将超窗历史压缩为长期摘要。

    Returns:
        (inject_messages, remove_messages)
        - inject: [SystemMessage(长期摘要)]（未超窗或压缩失败时为空）
        - remove: [RemoveMessage...] 待删除的旧消息
    """
    from langgraph.graph.message import RemoveMessage

    to_compress = select_messages_to_compress(existing, limit)
    if not to_compress:
        return [], []

    removals = [
        RemoveMessage(id=msg.id) for msg in to_compress if getattr(msg, "id", None)
    ]
    summary = await _summarize(to_compress, llm)
    if not summary:
        return [], removals  # 降级：直接删除（原行为）

    return [
        SystemMessage(content=f"{MEMORY_SUMMARY_PREFIX}{summary}"),
    ], removals
