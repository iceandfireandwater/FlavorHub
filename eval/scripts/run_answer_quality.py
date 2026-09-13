"""GustoBot 评测体系 —— 回答质量评测（LLM-as-Judge）。

对已生成的回答做质量评分：
- key_fact_coverage：答案含标注关键信息点占比（无 keywords 的样本跳过）
- llm_judge_score：LLM 按 rubric 对比 answer 与 ground_truth 打分（0-5，相关性/完整性）
- citation_consistency：回答 [n] 引用 ≤ sources 条数（有引用才计）

用法：
    1. 先跑 e2e 拿到回答：python -m eval.scripts.run_e2e_eval --output eval/results/e2e_eval_report.json
    2. 对回答做质量评测：python -m eval.scripts.run_answer_quality
       --details eval/results/e2e_eval_report_details.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List
from . import quiet  # noqa: F401  (import 即静音日志，只留进度条)
from .progress import make_progress

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_env() -> None:
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("LLM_API_KEY", ""))
    os.environ.setdefault("OPENAI_API_BASE", os.environ.get("LLM_BASE_URL", ""))


JUDGE_PROMPT = """你是菜谱问答评测员。请对比【模型回答】和【标准答案】，从三个维度各打 0-5 分：
- 相关性：回答是否切题、覆盖问题核心
- 完整性：回答是否包含标准答案的关键信息（菜名/食材/步骤/数值等）
- 忠实性：回答是否基于给定信息，没有编造（若标准答案不可得，此项按信息自洽给分）

标准答案：{ground_truth}

模型回答：{answer}

只输出 JSON：{{"relevance": 分数, "completeness": 分数, "faithfulness": 分数}}"""


async def judge_one(llm, answer: str, ground_truth: str) -> Dict[str, float]:
    from langchain_core.messages import HumanMessage, SystemMessage
    try:
        resp = await llm.ainvoke([
            SystemMessage(content="你是一个严格的评测员，只输出 JSON。"),
            HumanMessage(content=JUDGE_PROMPT.format(ground_truth=ground_truth[:500], answer=answer[:800])),
        ])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        # 提取 JSON
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {"relevance": 0.0, "completeness": 0.0, "faithfulness": 0.0}
        obj = json.loads(m.group(0))
        return {
            "relevance": float(obj.get("relevance", 0)),
            "completeness": float(obj.get("completeness", 0)),
            "faithfulness": float(obj.get("faithfulness", 0)),
        }
    except Exception:  # noqa
        return {"relevance": 0.0, "completeness": 0.0, "faithfulness": 0.0}


async def run_answer_quality(details_path: str, output: str, limit: int | None = None):
    from . import metrics

    load_env()
    from langchain_openai import ChatOpenAI
    from gustobot.config.settings import settings

    rows = [json.loads(l) for l in open(details_path, encoding="utf-8")]

    # 从评测集按 id 取 ground_truth 和 keywords（e2e details 里没存）
    from .schema import load_eval_set
    eval_set_path = os.path.join(os.path.dirname(details_path), "eval_set.jsonl")
    gt_map = {}
    if os.path.exists(eval_set_path):
        for c in load_eval_set(eval_set_path):
            gt_map[c.id] = (c.ground_truth, c.expected_answer_keywords)

    # 只取"有效回答"（非环境失败且非 ERROR）
    valid = [r for r in rows if not r.get("env_error") and r.get("answer") and not r["answer"].startswith("ERROR")]
    if limit:
        valid = valid[:limit]

    llm = ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model_name=settings.OPENAI_MODEL,
        openai_api_base=settings.OPENAI_API_BASE,
        temperature=0,
    )

    records = []
    prog = make_progress(len(valid), label="答案质量")
    for i, r in enumerate(valid):
        ans = r["answer"]
        gt, kws = gt_map.get(r["id"], ("", []))

        # ① 关键事实覆盖率（逐样本）
        # 与 run_e2e_eval / metrics.pipeline_success_rate 统一口径：
        # 原实现是"关键词原样包含"，而模型会改写措辞（差一个标点就判失败），
        # 两处口径不一致会得出矛盾的数字。
        from .metrics import _keyword_hit

        fact_hit = any(_keyword_hit(ans, kw) for kw in kws) if kws else None

        # ② LLM-as-Judge 打分（有 ground_truth 才打）
        judge = await judge_one(llm, ans, gt) if gt else {"relevance": None, "completeness": None, "faithfulness": None}

        # ③ 引用一致性
        refs = [int(x) for x in re.findall(r"\[(\d+)\]", ans)]
        citation_ok = (max(refs) <= len(r.get("sources", []))) if refs else None

        records.append({
            "id": r["id"], "scenario": r["scenario"],
            "answer": ans[:120],
            "fact_hit": fact_hit,
            "judge": judge,
            "citation_ok": citation_ok,
            "has_gt": bool(gt), "has_keywords": bool(kws), "has_citation": bool(refs),
        })
        prog.update(ok=(fact_hit is not False), note=r["id"])

    prog.finish()
    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    report = {"n_total": len(valid), "n_answer": len(valid)}
    fact_scored = [r for r in records if r["fact_hit"] is not None]
    report["n_fact_scored"] = len(fact_scored)
    if fact_scored:
        report["key_fact_coverage"] = sum(1 for r in fact_scored if r["fact_hit"]) / len(fact_scored)

    judge_scored = [r for r in records if r["judge"].get("completeness") is not None]
    report["n_judge_scored"] = len(judge_scored)
    if judge_scored:
        for dim in ("relevance", "completeness", "faithfulness"):
            report[f"llm_judge_{dim}_avg"] = round(
                sum(r["judge"][dim] for r in judge_scored) / len(judge_scored), 2)

    cite_scored = [r for r in records if r["citation_ok"] is not None]
    report["n_citation_scored"] = len(cite_scored)
    if cite_scored:
        report["citation_consistency"] = sum(1 for r in cite_scored if r["citation_ok"]) / len(cite_scored)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(output.replace(".json", "_details.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n============ 回答质量评测报告 ============")
    for k, v in report.items():
        print(f"  {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", default="eval/results/e2e_eval_report_details.jsonl")
    ap.add_argument("--output", default="eval/results/answer_quality_report.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run_answer_quality(args.details, args.output, args.limit))


if __name__ == "__main__":
    main()
