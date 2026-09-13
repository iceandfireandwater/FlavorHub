"""GustoBot 评测体系 —— 幻觉率评测（Hallucination Rate / Groundedness）。

口径说明（重要）：
- 数据源：eval/results/e2e_eval_report_details.jsonl（端到端实测回答）+ eval/cases/eval_set.jsonl（ground truth）
- 本机 e2e 明细中 sources 字段为空，无法做 faithfulness（回答 vs 检索来源）；
  因此采用 **factuality 口径**：回答的事实断言是否被标注 ground truth 支撑。
- 判定方式：独立 LLM-as-Judge（不走被测系统），断言级拆解 + 逐条判定：
    supported    被 ground truth 支撑
    contradicted 与 ground truth 矛盾（确定幻觉）
    unsupported  ground truth 无对应信息、回答自行给出（无据断言，严格口径计为幻觉）

指标：
    claim_hallucination_rate = (contradicted + unsupported) / 总断言数        # 严格口径
    claim_contradiction_rate = contradicted / 总断言数                        # 保守口径（确定矛盾）
    sample_hallucination_rate = 含 ≥1 条幻觉断言的样本数 / 判定样本数
    groundedness = supported / 总断言数
"""
from __future__ import annotations
from . import quiet  # noqa: F401  (import 即静音日志，只留进度条)
from .progress import make_progress

import argparse
import asyncio
import os
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, Field  # noqa: E402

DATA_DIR = PROJECT_ROOT / "eval" / "results"
DETAILS_PATH = DATA_DIR / "e2e_eval_report_details.jsonl"
EVAL_SET_PATH = DATA_DIR / "eval_set.jsonl"
REPORT_PATH = DATA_DIR / "hallucination_eval_report.json"
REPORT_DETAILS_PATH = DATA_DIR / "hallucination_eval_details.jsonl"

# 有 ground truth 的场景才可判定（negative / multi_turn 无标准答案，跳过）
SCORABLE_SCENARIOS = {"recipe_search", "recipe_detail", "recipe_compare", "history_faq", "stat_query"}


class ClaimJudgement(BaseModel):
    """单条事实断言的判定。"""

    claim: str = Field(description="从回答中拆出的原子事实断言")
    verdict: str = Field(description="supported / contradicted / unsupported 三选一")


class HallucinationJudgement(BaseModel):
    """一条样本的判定结果。"""

    claims: List[ClaimJudgement] = Field(default_factory=list, description="回答中拆出的全部事实断言及判定")


JUDGE_SYSTEM_PROMPT = """你是严格的事实核查员。给你【标准答案 ground truth】和【模型回答】，请：

1. 把模型回答拆成若干**原子事实断言**（每条只含一个可核查的事实点，如做法步骤、食材、耗时、口味）。
2. 对每条断言判定：
   - `supported`：标准答案里能直接/间接验证该断言
   - `contradicted`：与标准答案明确矛盾（确定幻觉，如步骤顺序错、食材张冠李戴、耗时说反）
   - `unsupported`：标准答案里没有对应信息，模型自行编造（无据断言，如凭空加入标准答案没有的食材或步骤）
3. 不要因为"措辞不同"判 contradicted；只判事实层面的对错。
4. 寒暄、引导语（如"你好呀""动手试试"）不算事实断言，忽略。

只输出断言列表及判定，不输出其他内容。"""

JUDGE_TIMEOUT_S = int(os.getenv("EVAL_JUDGE_TIMEOUT", "90"))


def build_judge():
    """独立 LLM-as-Judge（不走被测系统）。"""
    from gustobot.config.settings import settings
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,
        model_name=settings.OPENAI_MODEL,
        temperature=0,
    )
    return llm.with_structured_output(HallucinationJudgement)


def load_inputs() -> List[Dict[str, Any]]:
    """合并 e2e 明细（回答）与评测集（ground truth / 关键词）。"""
    cases = {}
    with EVAL_SET_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                cases[c["id"]] = c

    rows: List[Dict[str, Any]] = []
    with DETAILS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            c = cases.get(r["id"], {})
            r["ground_truth"] = c.get("ground_truth", "")
            r["expected_keywords"] = c.get("expected_answer_keywords", [])
            r["turns"] = c.get("turns", [])
            rows.append(r)
    return rows


def _judge_prompt(question: str, ground_truth: str, answer: str) -> str:
    return (
        f"【用户问题】\n{question}\n\n"
        f"【标准答案 ground truth】\n{ground_truth}\n\n"
        f"【模型回答】\n{answer}\n"
    )


async def judge_one(judge, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """对单条样本做断言级判定。"""
    answer = (row.get("answer") or "").strip()
    gt = (row.get("ground_truth") or "").strip()
    if not answer or not gt:
        return None

    question = (row.get("turns") or [""])[0] if row.get("turns") else ""
    try:
        # 加超时：没有它的话，单次请求 hang 在网络层会让整个 gather 永久等待
        # （实测就是这样卡死过 —— 进程 0 CPU、全线程 futex_wait）。
        result = await asyncio.wait_for(
            judge.ainvoke(
                [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "human", "content": _judge_prompt(question, gt, answer)},
                ]
            ),
            timeout=JUDGE_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001
        return {"id": row["id"], "error": f"{type(exc).__name__}: {exc}"}

    claims = [c.model_dump() for c in (result.claims or [])]
    n_total = len(claims)
    n_supported = sum(1 for c in claims if c["verdict"] == "supported")
    n_contra = sum(1 for c in claims if c["verdict"] == "contradicted")
    n_unsup = sum(1 for c in claims if c["verdict"] == "unsupported")
    has_hallu = (n_contra + n_unsup) > 0
    return {
        "id": row["id"],
        "scenario": row.get("scenario"),
        "question": question,
        "answer": answer[:500],
        "n_claims": n_total,
        "n_supported": n_supported,
        "n_contradicted": n_contra,
        "n_unsupported": n_unsup,
        "sample_hallucinated": has_hallu,
        "claims": claims,
    }


async def run(limit: Optional[int] = None, concurrency: int = 4) -> Dict[str, Any]:
    rows = load_inputs()
    scorable = [
        r for r in rows
        if r.get("scenario") in SCORABLE_SCENARIOS
        and (r.get("answer") or "").strip()
        and (r.get("ground_truth") or "").strip()
    ]
    if limit:
        scorable = scorable[:limit]

    print(f"可判定样本 {len(scorable)} 条（有 ground truth + 有回答）；其余场景（negative/multi_turn）无标准答案跳过")
    judge = build_judge()
    prog = make_progress(len(scorable), label="幻觉判定")

    sem = asyncio.Semaphore(concurrency)

    async def _guarded(row):
        async with sem:
            return await judge_one(judge, row)

    # 进度更新放在**主协程**里：之前写在 _guarded 内部，等于在并发协程中调
    # prog.update()，一旦 stdout 阻塞就会连累整个评测（已修 progress.py，
    # 但把 IO 收敛到单点更稳）。
    tasks = [asyncio.create_task(_guarded(r)) for r in scorable]
    results = []
    for fut in asyncio.as_completed(tasks):
        out = await fut
        results.append(out)
        prog.update(ok=bool(out and "error" not in out), note=(out or {}).get("id", ""))
    prog.finish()
    ok = [r for r in results if r and "error" not in r]
    failed = [r for r in results if r and "error" in r]

    total_claims = sum(r["n_claims"] for r in ok)
    total_supported = sum(r["n_supported"] for r in ok)
    total_contra = sum(r["n_contradicted"] for r in ok)
    total_unsup = sum(r["n_unsupported"] for r in ok)
    sample_hallu = sum(1 for r in ok if r["sample_hallucinated"])

    report: Dict[str, Any] = {
        "n_scored": len(ok),
        "n_failed": len(failed),
        "n_claims_total": total_claims,
        "claim_hallucination_rate": round((total_contra + total_unsup) / total_claims, 4) if total_claims else None,
        "claim_contradiction_rate": round(total_contra / total_claims, 4) if total_claims else None,
        "groundedness": round(total_supported / total_claims, 4) if total_claims else None,
        "sample_hallucination_rate": round(sample_hallu / len(ok), 4) if ok else None,
        "per_scenario": {},
        "failed_ids": [r["id"] for r in failed],
    }
    for sc in sorted({r["scenario"] for r in ok}):
        sub = [r for r in ok if r["scenario"] == sc]
        c = sum(r["n_claims"] for r in sub)
        report["per_scenario"][sc] = {
            "n": len(sub),
            "claim_hallucination_rate": round(
                sum(r["n_contradicted"] + r["n_unsupported"] for r in sub) / c, 4) if c else None,
            "sample_hallucination_rate": round(
                sum(1 for r in sub if r["sample_hallucinated"]) / len(sub), 4) if sub else None,
        }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with REPORT_DETAILS_PATH.open("w", encoding="utf-8") as f:
        for r in results:
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("============ 幻觉率评测报告 ============")
    print(json.dumps({k: v for k, v in report.items() if k != "per_scenario"}, ensure_ascii=False, indent=2))
    print("分场景:")
    print(json.dumps(report["per_scenario"], ensure_ascii=False, indent=2))
    print(f"报告已写入 {REPORT_PATH}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="GustoBot hallucination-rate evaluation")
    parser.add_argument("--limit", type=int, default=None, help="仅判定前 N 条（控制 API 消耗）")
    parser.add_argument("--concurrency", type=int, default=4, help="并发判定数")
    args = parser.parse_args()
    asyncio.run(run(limit=args.limit, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
