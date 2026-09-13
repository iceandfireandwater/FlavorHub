# -*- coding: utf-8 -*-
"""会话内多轮记忆评测（intra-session multi-turn memory）。

与跨会话记忆不同：这里**全程在同一个会话里**，测的是会话级记忆——
用户前面几轮说过的话，在**滑出上下文窗口**（GUSTOBOT_MEMORY_TURNS，默认 5 轮）之后，
还能不能被记住。它检验的是 L2（会话摘要压缩）而不是 L3（跨会话记忆表）。

流程：setup 轮 → fillers 轮（撑过窗口）→ probe 追问 → 检查 expect_any / forbid_any。

指标：
- recall            : 期望项命中率
- pass_rate         : 整条用例通过率
- window_overflow_rate : 实际触发窗口溢出的比例（>=6 轮才会溢出）
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import client
from . import quiet  # noqa: F401  (import 即静音日志，只留进度条)
from .progress import make_progress

CASES = "eval/cases/intra_memory_eval.jsonl"
# 用独立的 user_id 隔离，避免评测写入的"人设"污染真实用户的长期记忆
EVAL_USER_ID = "eval_intra_bot"
REPORT = "eval/results/intra_memory_eval_report.json"
USER_PREFIX = "eval_intra_"

# 每次运行换一个 tag：长期记忆按 user_id 持久化，复用同一个 id 会让本轮
# 被上一轮残留的 answered 结论污染（实测导致 update 类假失败）。
RUN_TAG = __import__("uuid").uuid4().hex[:6]
MIN_FILLERS = 5          # 配合 GUSTOBOT_MEMORY_TURNS=5，保证 setup 滑出窗口


def load_cases(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _match_expect_all(ans: str, groups) -> list:
    """expect_all 的每一项是一个"同义词组"，每组命中其一即算该组通过。

    （旧实现是扁平列表 + "命中数 == 期望数"，配上别名展开后会要求模型把
      「痛风/嘌呤」这类同义词全部说一遍，multi 类因此 7/10 被误判失败。）
    """
    out = []
    for grp in groups or []:
        candidates = grp if isinstance(grp, (list, tuple)) else [grp]
        out.append([k for k in candidates if k in ans])
    return out


def run_case(c) -> dict:
    """单条用例：全程同一会话，setup -> fillers -> probe。每条用独立 user_id 以便并发。"""
    uid = USER_PREFIX + c["id"] + "_" + RUN_TAG
    sid = None
    turns_used = 0
    for t in c.get("setup", []):
        sid = client.chat_stream(t, session_id=sid, user_id=uid)["session_id"]
        turns_used += 1
    fillers = list(c.get("fillers", []))
    while len(fillers) < MIN_FILLERS:
        fillers.append("还有别的推荐吗")
    for t in fillers:
        sid = client.chat_stream(t, session_id=sid, user_id=uid)["session_id"]
        turns_used += 1
    r = client.chat_stream(c["probe"], session_id=sid, user_id=uid)
    ans = r["text"] or ""
    exp_any = c.get("expect_any") or []
    exp_all = c.get("expect_all") or []
    forb = c.get("forbid_any") or []
    hit_any = [k for k in exp_any if k in ans]
    hit_all = _match_expect_all(ans, exp_all)
    bad = [k for k in forb if k in ans]
    ok = (not exp_any or len(hit_any) >= 1) and all(hit_all) and not bad
    return {
        "id": c["id"], "kind": c.get("kind"), "probe": c["probe"],
        "turns_used": turns_used + 1, "same_session": r["session_id"] == sid,
        "expect_any": exp_any, "hit_any": hit_any,
        "expect_all": exp_all, "hit_all": hit_all,
        "forbid_any": forb, "forbidden_hit": bad,
        "answer": ans[:400], "ok": ok,
    }


def run(cases, limit=None, workers: int = 6):
    if limit:
        cases = cases[:limit]
    prog = make_progress(len(cases), label="会话内记忆")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_case, c) for c in cases]
        details = []
        for fut in as_completed(futs):
            d = fut.result()
            details.append(d)
            prog.update(ok=bool(d.get("ok")), note=d.get("id", ""))
    per_kind = {}
    for kind in {d["kind"] for d in details}:
        sub = [d for d in details if d["kind"] == kind]
        per_kind[kind] = round(sum(1 for d in sub if d["ok"]) / len(sub), 4)
    prog.finish()
    hits = sum(len(d["hit_any"]) for d in details) + sum(
        sum(1 for grp in d["hit_all"] if grp) for d in details)
    total_expect = sum(len(d["expect_any"]) for d in details) + sum(len(d["expect_all"]) for d in details)
    poison_total = sum(1 for d in details if d["forbid_any"])
    poison_hit = sum(1 for d in details if d["forbid_any"] and d["forbidden_hit"])
    passed = sum(1 for d in details if d["ok"])
    overflowed = sum(1 for d in details if d["turns_used"] >= 6)
    report = {
        "n": len(cases),
        "recall": hits / total_expect if total_expect else 0.0,
        "pass_rate": passed / len(cases) if cases else 0.0,
        "poison_rate": poison_hit / poison_total if poison_total else 0.0,
        "window_overflow_rate": overflowed / len(cases) if cases else 0.0,
        "per_kind_pass_rate": per_kind,
    }
    return report, details


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=CASES)
    ap.add_argument("--output", default=REPORT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    report, details = run(load_cases(args.cases), args.limit, args.workers)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(args.output.replace(".json", "_details.jsonl"), "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("\n============ 会话内多轮记忆评测 ============")
    print(f"样本数:                {report['n']}")
    print(f"Recall:                {report['recall']:.4f}")
    print(f"Pass Rate:             {report['pass_rate']:.4f}")
    print(f"Poison Rate(越低越好):  {report['poison_rate']:.4f}")
    print(f"窗口溢出比例:           {report['window_overflow_rate']:.4f}")
    print(f"\n明细: {args.output.replace('.json', '_details.jsonl')}")


if __name__ == "__main__":
    main()
