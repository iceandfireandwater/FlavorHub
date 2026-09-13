"""GustoBot 评测体系 —— 评测集生成 CLI。

用法：
    python -m eval.generate                          # 模板生成（不调 LLM，history_faq 为占位）
    python -m eval.generate --use-llm                # history_faq 用真实 LLM 出题
    python -m eval.generate --output eval/cases/eval_set.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .dataset import generate_all  # noqa: E402
from .schema import SCENARIOS, save_eval_set  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate GustoBot eval set")
    ap.add_argument("--use-llm", action="store_true", help="use LLM for history_faq questions")
    ap.add_argument("--count", type=int, default=None, help="limit case count")
    ap.add_argument("--output", default="eval/cases/eval_set.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cases = generate_all(count=args.count, use_llm=args.use_llm, seed=args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_eval_set(args.output, cases)

    from collections import Counter
    dist = Counter(c.scenario for c in cases)
    print(f"评测集已生成: {len(cases)} 条 -> {args.output}")
    for s, n in dist.items():
        print(f"  {s:16s} {SCENARIOS[s]:16s} {n} 条")
    print(f"ground truth 来源均标注于 gt_source 字段，可逐条回溯")


if __name__ == "__main__":
    main()
