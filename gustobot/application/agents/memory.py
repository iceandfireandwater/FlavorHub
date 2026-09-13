"""会话记忆：窗口裁剪 + 超窗历史的结构化抽取。

与「自由文本摘要」的区别：
- LLM 只负责**抽取本轮增量**（delta），不负责重写整段摘要；
- 合并规则由代码确定性执行（constraints 只增去重 / preferences 键覆盖 /
  dishes 并集 / answered 按问题去重），旧字段不会被 LLM 在重写时丢掉，
  也不会出现"越压越糊"的逐轮退化；
- 硬性约束（忌口、过敏、给老人吃…）单独成字段，注入时显式强调；
- 抽取失败时降级：先退回自由文本摘要（存 notes），再失败才"只删不记"。
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import RemoveMessage

logger = logging.getLogger(__name__)

MEMORY_HEADER = "[会话记忆]"

EMPTY_MEMORY: Dict[str, Any] = {
    "constraints": [],   # 硬性约束：只增不删（去重），用户撤回的移到 relaxed
    "relaxed": [],       # 用户后来主动解除的限制（更新而非删除，避免信息丢失）
    "preferences": {},   # 口味/做法偏好：键覆盖
    "dishes": [],        # 提到的菜名/食材：并集去重
    "facts": [],         # 用户自述的关于自己的事实（姓名/城市/职业/家庭…）：并集去重
    "answered": [],      # 已问过并已答复的问题：[{"q":..,"a":..}]
    "notes": [],         # 降级时的自由文本摘要（抽取失败时兜底）
}

# 注意：这里刻意用「静态 system 文本」，历史另外放进 HumanMessage，
# 而不是用 ChatPromptTemplate——因为 JSON 示例里的花括号会被模板当成变量，
# 曾导致抽取 100% 失败（报 missing variables {'\n  "constraints"}）。
EXTRACT_SYSTEM = """你是会话记忆抽取器。从用户给出的对话片段中抽取**值得跨轮记住**的信息，只输出一个严格合法的 JSON 对象，不要 markdown 代码块、不要任何解释文字。
JSON 结构（四个字段都必须出现，没有内容就给空数组/空对象）：
{
  "constraints": ["用户明确的硬性要求、忌口、过敏、场景约束，如 家里有老人 / 不吃辣 / 对花生过敏"],
  "constraints_removed": ["用户明确表示「不用管了 / 改主意了 / 不再需要 / 取消了」的限制，填关键名词，如 花生 / 辣椒"],
  "preferences": {"口味": "清淡", "做法": "少油"},
  "dishes": ["对话中出现的菜名或食材名"],
  "facts": ["用户自述的、关于自己的事实：姓名、所在城市、职业、家庭成员、生活状态等，如 用户叫阿强 / 用户住在成都 / 用户家里有老人"],
  "answered": [{"q": "用户问过的问题", "a": "已经给出的结论要点"}]
}
要求：
- constraints 只收用户明确表达的硬性要求或忌口，不要收录普通寒暄；
- facts 收**用户陈述关于自己的客观信息**（谁、在哪、做什么、家里有什么人），哪怕是顺口一提也要收，
  这是让系统"记得住人"的关键字段；不要把疑问句或助手的推测写进去；
- answered 只收**真的给出了有效结论**的问题；凡是"没找到 / 无法回答 / 不在范围内 / 拒绝回答 / 建议换种问法"
  这类**没有实质信息**的答案，一律不要写进 answered（否则错误的空答案会一直污染后续对话）；
- 所有字符串都要简短。"""

LEGACY_SUMMARY_SYSTEM = """你是会话记忆压缩器。把用户给出的对话历史压缩成一段中文要点摘要（不超过 200 字）。必须保留：用户提到的食材、菜名、口味偏好与忌口、健康/场景约束（如给老人吃）、已经问过的问题及其结论要点。丢弃寒暄和重复内容。只输出摘要正文，不要任何前缀、标题或解释。"""


def resolve_turn_limit() -> Optional[int]:
    """保留的对话轮数；<=0 视为不限制。"""
    raw = os.getenv("GUSTOBOT_MEMORY_TURNS", os.getenv("GUSTOBOT_MAX_MEMORY_TURNS", "5"))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 5
    return value if value > 0 else None


def select_overflow(existing: Sequence[BaseMessage], limit: Optional[int]) -> List[BaseMessage]:
    """挑出窗口之外、需要抽取掉的旧消息（只按 human 消息计轮次）。"""
    if not existing or limit is None:
        return []

    humans_to_keep = max(limit - 1, 0)
    if humans_to_keep == 0:
        return [m for m in existing if getattr(m, "id", None)]

    seen = 0
    keep_from = 0
    for index in range(len(existing) - 1, -1, -1):
        if getattr(existing[index], "type", None) == "human":
            seen += 1
            if seen == humans_to_keep:
                keep_from = index
                break
    else:
        keep_from = 0
    return [m for m in existing[:keep_from] if getattr(m, "id", None)]


def _stringify(msg: Any) -> str:
    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item if isinstance(item, str) else str(item.get("text", "")) for item in content
        )
    return str(content)


def _default_llm():
    from langchain_openai import ChatOpenAI

    from gustobot.config.settings import settings

    return ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model_name=settings.OPENAI_MODEL,
        openai_api_base=settings.OPENAI_API_BASE,
        temperature=0.1,
    )


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    """尽力从 LLM 输出里抠出一个 JSON 对象；失败返回 None。"""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(t[i:j + 1])
    except Exception:  # noqa: BLE001
        return None
    return obj if isinstance(obj, dict) else None


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in (None, "", {}):
        return []
    return [value]


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def merge_memory(previous: Optional[Dict[str, Any]], delta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """确定性合并：LLM 只给增量，字段合并规则在这里。"""
    old = previous or {}
    merged: Dict[str, Any] = {
        "constraints": [str(x).strip() for x in _as_list(old.get("constraints")) if str(x).strip()],
        "relaxed": [str(x).strip() for x in _as_list(old.get("relaxed")) if str(x).strip()],
        "preferences": {str(k).strip(): str(v).strip() for k, v in _as_dict(old.get("preferences")).items()},
        "dishes": [str(x).strip() for x in _as_list(old.get("dishes")) if str(x).strip()],
        "facts": [str(x).strip() for x in _as_list(old.get("facts")) if str(x).strip()],
        "answered": [x for x in _as_list(old.get("answered")) if isinstance(x, dict)],
        "notes": [str(x).strip() for x in _as_list(old.get("notes")) if str(x).strip()],
    }

    d = delta or {}
    # 先处理"撤回"：用户改主意/取消的限制必须从既有 constraints 移除。
    # constraints 本身仍"只增"（防 LLM 重写丢字段），撤回走这条独立通道。
    for item in _as_list(d.get("constraints_removed")):
        kw = str(item).strip()
        if not kw:
            continue
        # 双向宽松匹配：LLM 给的关键词与约束原文不一定词面一致
        # （如 keyword="生冷食物" vs constraint="不吃生冷的东西"）
        keep, moved = [], []
        for c in merged["constraints"]:
            toks = [t for t in __import__("re").split(r"[，,、\s]+", kw) if t]
            hit = kw in c or c in kw or any(t in c for t in toks)
            (moved if hit else keep).append(c)
        merged["constraints"] = keep
        # 关键：不是删掉，而是记成"已解除" —— 否则跨会话再问时模型无从回答，
        # 只能反问用户（实测 9/10 条判失败就是这个原因）。
        for c in moved:
            merged["relaxed"].append(c)
        if not moved:
            merged["relaxed"].append(kw)

    for item in _as_list(d.get("constraints")):
        text = str(item).strip()
        if text and text not in merged["constraints"]:
            merged["constraints"].append(text)          # 只增不删（撤回见上）
    for key, value in _as_dict(d.get("preferences")).items():
        key, value = str(key).strip(), str(value).strip()
        if key and value:
            merged["preferences"][key] = value          # 键覆盖
    for item in _as_list(d.get("dishes")):
        text = str(item).strip()
        if text and text not in merged["dishes"]:
            merged["dishes"].append(text)               # 并集
    for item in _as_list(d.get("facts")):
        text = str(item).strip()
        if text and text not in merged["facts"]:
            merged["facts"].append(text)                # 并集
    seen_q = {str(x.get("q", "")).strip() for x in merged["answered"]}
    for item in _as_list(d.get("answered")):
        if isinstance(item, dict):
            q = str(item.get("q", "")).strip()
            if q and q not in seen_q:
                merged["answered"].append({"q": q, "a": str(item.get("a", "")).strip()})
                seen_q.add(q)                           # 按问题去重

    merged["answered"] = merged["answered"][-20:]
    merged["facts"] = merged["facts"][-30:]
    merged["dishes"] = merged["dishes"][-50:]
    merged["notes"] = merged["notes"][-3:]
    merged["constraints"] = merged["constraints"][-20:]
    merged["relaxed"] = list(dict.fromkeys(merged["relaxed"]))[-10:]
    return merged


def render_memory(memory: Optional[Dict[str, Any]]) -> str:
    """把结构化记忆渲染成给模型看的一段文本（硬约束放最前并显式强调）。"""
    if not memory:
        return ""
    parts: List[str] = []
    if memory.get("constraints"):
        parts.append("硬性约束（必须遵守，不得违背）：" + "；".join(str(x) for x in memory["constraints"]))
    if memory.get("relaxed"):
        parts.append(
            "用户已主动解除的限制（不要再拿它当禁忌，可以正常推荐）："
            + "；".join(str(x) for x in memory["relaxed"])
        )
    if memory.get("preferences"):
        parts.append("口味/做法偏好：" + "；".join("%s=%s" % (k, v) for k, v in memory["preferences"].items()))
    if memory.get("facts"):
        parts.append("关于这位用户（过往对话中了解到）：" + "；".join(str(x) for x in memory["facts"]))
    if memory.get("dishes"):
        parts.append("已提及的菜名/食材：" + "、".join(str(x) for x in memory["dishes"]))
    if memory.get("answered"):
        recent = memory["answered"][-8:]
        parts.append(
            "已问过并已答复的问题："
            + "；".join("%s → %s" % (i.get("q", ""), i.get("a", "")) for i in recent)
        )
    if memory.get("notes"):
        parts.append("历史摘要：" + " ".join(str(x) for x in memory["notes"]))
    if not parts:
        return ""
    guide = (
        "（以上是你和这位用户过往对话中积累的记忆。硬性约束必须遵守；"
        "当用户问到记忆里已有的信息（他是谁、住哪、有什么要求、之前聊过什么）时，"
        "必须直接引用上面的内容作答；与问题无关时才不必提起。" + chr(10) +
        "不要生硬罗列，也不要编造这里没有的细节。）"
    )
    body = "\n".join(parts)
    return MEMORY_HEADER + "\n" + body + "\n" + guide


def _history_text(overflow: Sequence[BaseMessage]) -> str:
    return "\n".join(
        "%s: %s" % ("用户" if getattr(m, "type", None) == "human" else "助手", _stringify(m))
        for m in overflow
    )


async def extract_delta(overflow: Sequence[BaseMessage], llm=None) -> Optional[Dict[str, Any]]:
    """抽取本轮增量；结构化失败时退回自由文本摘要（放进 notes）；再失败返回 None。"""
    if not overflow:
        return {}
    if llm is None:
        llm = _default_llm()
    history = _history_text(overflow)

    # 结构化抽取（不经过 ChatPromptTemplate，见文件头注释）
    try:
        resp = await llm.ainvoke(
            [
                SystemMessage(content=EXTRACT_SYSTEM),
                HumanMessage(content="对话片段：\n" + history),
            ]
        )
        delta = _parse_json(_stringify(resp))
        if delta is not None:
            return delta
        logger.warning("记忆抽取返回非法 JSON，退回自由文本摘要")
    except Exception as exc:  # noqa: BLE001
        logger.warning("记忆抽取调用失败: %s", exc)

    # 降级 1：自由文本摘要兜底
    try:
        resp = await llm.ainvoke(
            [
                SystemMessage(content=LEGACY_SUMMARY_SYSTEM),
                HumanMessage(content="对话历史：\n" + history),
            ]
        )
        text = _stringify(resp).strip()
        if text:
            return {"notes": [text]}
    except Exception as exc:  # noqa: BLE001
        logger.warning("自由文本摘要兜底也失败: %s", exc)
    return None


async def prepare_context(
    existing: Sequence[BaseMessage],
    previous_memory: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    llm=None,
    recent_count: int = 4,
) -> Tuple[List[RemoveMessage], Dict[str, Any]]:
    """压缩前的上下文整理。

    Returns:
        (removals, new_memory)
        - removals: 要真正从 state 删除的旧消息
        - new_memory: 合并后的结构化记忆（没有溢出时原样返回）
    """
    if limit is None:
        limit = resolve_turn_limit()

    overflow = select_overflow(existing, limit)

    # 除"滑出窗口的旧消息"之外，最近的若干条也要抽一遍：这样刚说过的话能立刻
    # 进入长期记忆（跨会话可见），而不必等它滑出窗口。重复内容由 merge_memory 去重。
    overflow_ids = {id(m) for m in overflow}
    tail = [
        m for m in existing
        if getattr(m, "id", None) and id(m) not in overflow_ids
    ][-recent_count:]
    targets = list(overflow) + tail
    if not targets:
        return [], previous_memory or {}

    removals = [RemoveMessage(id=m.id) for m in overflow if getattr(m, "id", None)]
    delta = await extract_delta(targets, llm)
    if delta is None:
        return removals, previous_memory or {}   # 降级：只删不记
    return removals, merge_memory(previous_memory, delta)
