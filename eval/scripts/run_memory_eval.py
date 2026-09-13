# -*- coding: utf-8 -*-
"""跨会话记忆评测（并发版）。

对每个用例：
1. 在**会话 A** 里说几轮（建立记忆）；
2. **另开一个全新会话 B** 提问（probe）——这才是真正的"跨会话"；
3. 检查回答是否命中 expect_any，且不出现 forbid_any。

并发说明：每个用例使用**独立的 user_id**（`eval_mem_<id>`），因此用例之间互不干扰，
可以安全并发。指标：
- recall           : 期望项命中率（记没记住）
- poison_rate       : forbid_any 出现率（错误结论被固化进记忆）——越低越好
- pass_rate        : 整条用例通过率
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import client
from . import quiet  # noqa: F401  (import 即静音日志，只留进度条)
from .progress import make_progress

CASES = "eval/cases/memory_eval.jsonl"
REPORT = "eval/results/memory_eval_report.json"
USER_PREFIX = "eval_mem_"

# 每次运行换一个 tag：长期记忆按 user_id 持久化，复用同一个 id 会让本轮
# 被上一轮残留的 answered 结论污染（实测导致 update 类假失败）。
RUN_TAG = __import__("uuid").uuid4().hex[:6]


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
    """在隔离的 user_id 下跑一条用例。

    判据：
    - expect_any : 命中**任意一个**即可（宽松）
    - expect_all : **必须全部**命中（严格）
    - forbid_any : 出现任何一个即判失败（防"问句蒙分"/防旧值残留）
    - other_user_lines: 先让**另一个 user_id** 说这些内容，用于验证用户间不会串味
    """
    uid = USER_PREFIX + c["id"] + "_" + RUN_TAG

    # isolation 用例：另一个用户先说话（跑两轮以确保记忆沉淀）
    for t in c.get("other_user_lines", []):
        other = uid + "_other"
        client.chat_stream(t, user_id=other)
        client.chat_stream("好的谢谢", user_id=other)

    sid = None
    for turn in c.get("turns", []):
        r = client.chat_stream(turn, session_id=sid, user_id=uid)
        sid = r["session_id"]
    probe = client.chat_stream(c["probe"], user_id=uid)
    answer = probe["text"] or ""

    exp_any = c.get("expect_any") or []
    exp_all = c.get("expect_all") or []
    forb = c.get("forbid_any") or []
    hit_any = [k for k in exp_any if k in answer]
    hit_all = _match_expect_all(answer, exp_all)
    bad = [k for k in forb if k in answer]

    ok = (not exp_any or len(hit_any) >= 1) and all(hit_all) and not bad
    return {
        "id": c["id"], "kind": c.get("kind"), "probe": c["probe"],
        "new_session": probe["session_id"] != sid,
        "expect_any": exp_any, "hit_any": hit_any,
        "expect_all": exp_all, "hit_all": hit_all,
        "forbid_any": forb, "forbidden_hit": bad,
        "answer": answer[:400], "ok": ok,
    }


def run(cases, limit=None, workers: int = 6):
    if limit:
        cases = cases[:limit]
    prog = make_progress(len(cases), label="跨会话记忆")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_case, c) for c in cases]
        details = []
        for fut in as_completed(futs):
            d = fut.result()
            details.append(d)
            prog.update(ok=bool(d.get("ok")), note=d.get("id", ""))

    # 一个"期望项"= expect_any 的一个词，或 expect_all 的一个同义词组
    prog.finish()
    hits = sum(len(d["hit_any"]) for d in details) + sum(
        sum(1 for grp in d["hit_all"] if grp) for d in details)
    total_expect = sum(len(d["expect_any"]) for d in details) + sum(len(d["expect_all"]) for d in details)
    poison_total = sum(1 for d in details if d["forbid_any"])
    poison_hit = sum(1 for d in details if d["forbid_any"] and d["forbidden_hit"])
    passed = sum(1 for d in details if d["ok"])

    from collections import Counter
    per_kind = {}
    for kind in {d["kind"] for d in details}:
        sub = [d for d in details if d["kind"] == kind]
        per_kind[kind] = round(sum(1 for d in sub if d["ok"]) / len(sub), 4)
    report = {
        "n": len(cases),
        "recall": hits / total_expect if total_expect else 0.0,
        "poison_rate": poison_hit / poison_total if poison_total else 0.0,
        "pass_rate": passed / len(cases) if cases else 0.0,
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

    print("\n============ 跨会话记忆评测 ============")
    print(f"样本数:                {report['n']}（并发 {args.workers}）")
    print(f"Recall:                {report['recall']:.4f}")
    print(f"Poison Rate(越低越好):  {report['poison_rate']:.4f}")
    print(f"Pass Rate:             {report['pass_rate']:.4f}")
    print(f"\n明细: {args.output.replace('.json', '_details.jsonl')}")


if __name__ == "__main__":
    main()
