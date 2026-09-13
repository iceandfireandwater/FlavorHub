"""tests/test_memory_summary.py —— 多层级会话记忆（摘要压缩）单元测试。

覆盖：未超窗不压缩 / 超窗压缩为摘要 / LLM 失败降级直接裁剪 / 关闭窗口不限。
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from gustobot.application.agents.memory import (
    compress_history,
    select_messages_to_compress,
    MEMORY_SUMMARY_PREFIX,
)


class MockLLM:
    """最小 LLM 桩：返回固定摘要。"""
    async def ainvoke(self, messages):
        class Resp:
            content = "用户询问了宫保鸡丁的做法，助手给出了鸡肉切丁、爆炒花生的步骤要点。"
        return Resp()


class FailingLLM:
    async def ainvoke(self, messages):
        raise RuntimeError("llm down")


def make_messages(n_turns: int):
    msgs = []
    for i in range(n_turns):
        msgs.append(HumanMessage(content=f"问题{i}", id=f"h{i}"))
        msgs.append(AIMessage(content=f"回答{i}", id=f"a{i}"))
    return msgs


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_within_window_no_compression():
    msgs = make_messages(2)
    inject, removals = run(compress_history(msgs, limit=5))
    assert inject == []
    assert removals == []


def test_over_window_compresses_to_summary():
    msgs = make_messages(8)  # 8 轮，窗口 5 → 超窗
    inject, removals = run(compress_history(msgs, limit=5, llm=MockLLM()))
    assert len(inject) == 1
    assert isinstance(inject[0], SystemMessage)
    assert inject[0].content.startswith(MEMORY_SUMMARY_PREFIX)
    assert "宫保鸡丁" in inject[0].content
    assert len(removals) > 0  # 旧消息被标记删除
    # 窗口内最近 4 条 human 消息保留
    kept_humans = [m for m in msgs if m.type == "human"][-4:]
    removed_ids = {r.id for r in removals}
    assert all(m.id not in removed_ids for m in kept_humans)


def test_llm_failure_falls_back_to_drop():
    msgs = make_messages(8)
    inject, removals = run(compress_history(msgs, limit=5, llm=FailingLLM()))
    assert inject == []          # 摘要失败不注入
    assert len(removals) > 0     # 降级为直接裁剪


def test_unlimited_window_no_compression():
    msgs = make_messages(20)
    inject, removals = run(compress_history(msgs, limit=None))
    assert inject == []
    assert removals == []


def test_select_messages_to_compress():
    msgs = make_messages(6)
    to_compress = select_messages_to_compress(msgs, limit=5)
    removed_ids = {m.id for m in to_compress}
    assert "h0" in removed_ids and "a0" in removed_ids
    assert "h4" not in removed_ids  # 最近 4 轮 human 保留
