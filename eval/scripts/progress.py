# -*- coding: utf-8 -*-
"""统一的评测进度条：终端原地刷新 + 已用/剩余时间估计。

设计要点
- **线程安全**：评测器普遍用 ThreadPoolExecutor 并发，`update()` 内部加锁。
- **非 tty 自动降级**：重定向到文件时 `\r` 会刷出满屏，改成每 N 条打一行。
- **ETA 用滑动窗口**：整体平均速率在并发下抖动很大（前面被拖慢、后面又追上来），
  所以用"最近若干次完成"的瞬时速率算剩余时间，稳定得多。

用法：
    from .progress import make_progress

    prog = make_progress(len(cases), label="路由")
    for case in cases:
        ...                       # 跑一条
        prog.update(ok=True, note=case["id"])   # ok 可省略
    prog.finish()
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from types import SimpleNamespace


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return "%ds" % seconds
    if seconds < 3600:
        return "%dm%02ds" % (seconds // 60, seconds % 60)
    return "%dh%02dm" % (seconds // 3600, (seconds % 3600) // 60)


def make_progress(
    total: int,
    *,
    label: str = "",
    width: int = 28,
    non_tty_every: int = 10,
    window: int = 20,
):
    """返回带 update(ok=None, note="") / finish() 的进度对象。"""
    is_tty = sys.stdout.isatty()
    lock = threading.Lock()
    state = {"done": 0, "ok": 0, "start": time.time(), "printed": 0}
    stamps: deque = deque(maxlen=window)      # 最近 N 次完成的时刻，用于滑动速率

    def _rates() -> tuple:
        done = state["done"]
        elapsed = max(time.time() - state["start"], 1e-6)
        avg = done / elapsed
        if len(stamps) >= 2:
            span = stamps[-1] - stamps[0]
            instant = (len(stamps) - 1) / span if span > 0 else avg
        else:
            instant = avg
        # 并发下瞬时速率容易尖峰，取两者较小值更保守（剩余时间不会报得太乐观）
        rate = min(instant, avg) if avg > 0 else instant
        eta = (total - done) / rate if rate > 0 else 0.0
        return elapsed, eta

    def _line(done: int, note: str) -> str:
        elapsed, eta = _rates()
        pct = (100.0 * done / total) if total else 100.0
        filled = int(width * done / total) if total else width
        bar = "#" * filled + "." * (width - filled)
        head = ("%s " % label) if label else ""
        return "  %s[%s] %3d/%d %5.1f%%  已用 %s  剩余 ~%s  %s%s" % (
            head, bar, done, total, pct,
            _fmt_duration(elapsed), _fmt_duration(eta),
            ("通过 %d " % state["ok"]) if state["ok"] else "",
            note[:26],
        )

    def update(ok=None, note: str = "") -> None:
        # ⚠️ 关键：**锁内绝不做 IO**。
        # 之前把 sys.stdout.write / print 放在 `with lock` 里，在 asyncio 协程中
        # 持 threading.Lock 做阻塞写，管道写满就会永久卡死 —— 实测幻觉评测跑成
        # 0 CPU 占用、所有线程停在 futex_wait。现在锁内只更新计数并算好字符串，
        # IO 全部挪到锁外，并吞掉异常（进度条绝不该影响评测本身）。
        line, newline = None, False
        with lock:
            state["done"] += 1
            if ok:
                state["ok"] += 1
            stamps.append(time.time())
            done = state["done"]
            if is_tty:
                line = chr(13) + _line(done, note) + "   "
                newline = done >= total
            elif done % non_tty_every == 0 or done >= total:
                line = _line(done, note)
        if line is None:
            return
        try:
            sys.stdout.write(line + (chr(10) if newline else ""))
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass

    def finish() -> None:
        with lock:
            already = state["done"] >= total
            state["done"] = total
            if is_tty:
                sys.stdout.write(chr(13) + _line(total, "完成") + "   " + chr(10))
                sys.stdout.flush()
            elif not already:          # 非 tty 时最后一条已打过，不再重复
                print(_line(total, "完成"))
            else:
                print()                # 收尾换行，避免报告挤在进度条同一行

    return SimpleNamespace(update=update, finish=finish, total=total)
