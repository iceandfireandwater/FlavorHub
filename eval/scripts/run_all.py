# -*- coding: utf-8 -*-
"""GustoBot 评测总入口。

一键跑完全部评测维度（**包含改造前就有的 6 个评测器**），并汇总成一份总报告。

用法：
    python -m eval.scripts.run_all                     # 跑全部
    python -m eval.scripts.run_all --quick             # 每维度只跑 3 条（冒烟）
    python -m eval.scripts.run_all --only memory,guardrail
    python -m eval.scripts.run_all --skip e2e,answer_quality
    python -m eval.scripts.run_all --baseline eval/results/eval_summary.json   # 与上次对比
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from . import client

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "eval" / "results"
SUMMARY_MD = DATA / "EVAL_SUMMARY.md"
SUMMARY_JSON = DATA / "eval_summary.json"

# (维度名, 模块, 中文名, 结果文件, 报告里要展示的关键指标, 是否新增)
DIMENSIONS = [
    # (维度, 模块, 中文名, 报告文件, 要展示的指标路径, 是否新增)
    ("route", "eval.scripts.run_route_eval", "意图路由", "route_eval_report.json",
     ["intent_accuracy", "slot_accuracy_strict", "slot_accuracy_loose"], False),
    ("retrieval", "eval.scripts.run_retrieval_eval", "知识检索", "retrieval_eval_report.json",
     ["top_k.5.recall@k", "top_k.5.mrr", "top_k.5.ndcg"], False),
    ("tool", "eval.scripts.run_tool_eval", "工具选择", "tool_eval_report.json",
     ["tool_accuracy"], False),
    ("hallucination", "eval.scripts.run_hallucination_eval", "幻觉控制", "hallucination_eval_report.json",
     ["claim_hallucination_rate", "sample_hallucination_rate", "groundedness"], False),
    ("answer_quality", "eval.scripts.run_answer_quality", "答案质量", "answer_quality_report.json",
     ["key_fact_coverage"], False),
    ("e2e", "eval.scripts.run_e2e_eval", "端到端", "e2e_eval_report.json",
     ["intent_accuracy", "key_fact_coverage", "effective_tool_coverage"], False),
    ("memory", "eval.scripts.run_memory_eval", "跨会话记忆", "memory_eval_report.json",
     ["recall", "pass_rate", "poison_rate"], True),
    ("guardrail", "eval.scripts.run_guardrail_eval", "拒答正确性", "guardrail_eval_report.json",
     ["reject_rate", "answer_rate"], True),
    ("intra_memory", "eval.scripts.run_intra_session_memory_eval", "会话内多轮记忆",
     "intra_memory_eval_report.json",
     ["recall", "pass_rate", "poison_rate"], True),
]


def _pick(report: dict, keys) -> dict:
    """按点分路径取指标，支持嵌套（如 top_k.5.vector_metrics.mrr）。"""
    out = {}
    for path in keys:
        cur = report
        ok = True
        for part in str(path).split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            out[path.split(".")[-1] if path.count(".") <= 1 else path] = cur
    return out


def run_one(name: str, module: str, quick: bool) -> dict:
    cmd = [sys.executable, "-m", module]
    if quick:
        cmd += ["--limit", "3"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    # 老的评测器不支持 --limit，quick 模式下会因"unrecognized arguments"退出 -> 去掉重试
    if quick and proc.returncode != 0 and "unrecognized arguments" in (proc.stderr or ""):
        proc = subprocess.run([sys.executable, "-m", module], cwd=str(ROOT),
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - t0
    status = "ok" if proc.returncode == 0 else "failed"
    tail = ((proc.stdout or "") + (proc.stderr or ""))[-500:] if status == "failed" else ""
    return {"status": status, "elapsed": round(elapsed, 1), "tail": tail, "cmd": " ".join(cmd)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑这些维度（逗号分隔）")
    ap.add_argument("--skip", default="", help="跳过这些维度（逗号分隔）")
    ap.add_argument("--quick", action="store_true", help="每维度只跑 3 条")
    ap.add_argument("--baseline", default="", help="与旧的 eval_summary.json 对比")
    args = ap.parse_args()

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    skip = {x.strip() for x in args.skip.split(",") if x.strip()}
    picked = [d for d in DIMENSIONS if (not only or d[0] in only) and d[0] not in skip]

    if not client.health():
        print("❌ 后端不可用（http://localhost:8000/health 不通），评测需要真实服务。")
        sys.exit(1)

    results = {}
    print("=" * 64)
    print(f"GustoBot 评测总入口：共 {len(picked)} 个维度" + ("（quick 模式）" if args.quick else ""))
    print("=" * 64)

    for idx, (name, module, label, report_file, metrics, is_new) in enumerate(picked, 1):
        tag = "新增" if is_new else "原有"
        print(f"\n>>> [{idx}/{len(picked)}] {label}（{name}，{tag}）")
        meta = run_one(name, module, args.quick)
        entry = {"label": label, "is_new": is_new, **meta}
        report_path = DATA / report_file
        if meta["status"] == "ok" and report_path.exists():
            try:
                entry["metrics"] = _pick(json.loads(report_path.read_text(encoding="utf-8")), metrics)
            except Exception as exc:  # noqa: BLE001
                entry["metrics"] = {"error": str(exc)}
        results[name] = entry
        print(f"    -> {meta['status']}（{meta['elapsed']}s）{entry.get('metrics', {})}")

    # ── 汇总 ────────────────────────────────────────────────
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quick": args.quick,
        "dimensions": results,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# GustoBot 评测总报告", "",
             f"- 生成时间：{summary['generated_at']}",
             f"- 模式：{'quick（每维度 3 条）' if args.quick else '全量'}",
             "", "| 维度 | 类型 | 状态 | 耗时 | 关键指标 |", "|---|---|---|---|---|"]
    for name, e in results.items():
        m = "、".join(f"{k}={v}" for k, v in (e.get("metrics") or {}).items())
        lines.append(f"| {e['label']} | {'新增' if e['is_new'] else '原有'} | {e['status']} | {e['elapsed']}s | {m or '-'} |")

    # 与基线对比
    if args.baseline and Path(args.baseline).exists():
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        lines += ["", "## 与基线对比", "", "| 维度 | 指标 | 基线 | 当前 | 变化 |", "|---|---|---|---|---|"]
        for name, e in results.items():
            b = (base.get("dimensions") or {}).get(name) or {}
            for k, v in (e.get("metrics") or {}).items():
                bv = (b.get("metrics") or {}).get(k)
                if isinstance(v, (int, float)) and isinstance(bv, (int, float)):
                    d = v - bv
                    lines.append(f"| {e['label']} | {k} | {bv} | {v} | {'+' if d >= 0 else ''}{round(d, 4)} |")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok = sum(1 for e in results.values() if e["status"] == "ok")
    print("\n" + "=" * 64)
    print(f"完成：{ok}/{len(results)} 个维度成功")
    print(f"总报告：{SUMMARY_MD.relative_to(ROOT)}")
    print(f"机读版：{SUMMARY_JSON.relative_to(ROOT)}")
    for name, e in results.items():
        if e["status"] != "ok":
            print(f"  ❌ {e['label']} 失败：{e.get('tail', '')[:200]}")


if __name__ == "__main__":
    main()
