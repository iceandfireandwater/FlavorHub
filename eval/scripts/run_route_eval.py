"""GustoBot 评测体系 —— 路由评测（Intent Accuracy / Slot Accuracy）。

评测对象是真实系统的主路由节点 analyze_and_route_query（LLM 结构化输出 + 启发式双保险），
不依赖数据库，因此可在任意环境离线运行。

用法：
    python -m eval.scripts.run_route_eval [--eval-set eval/cases/eval_set.jsonl]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# 环境：加载 .env（不打印任何密钥）
# ---------------------------------------------------------------------------
_ENV_LOADED = False


def load_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("LLM_API_KEY", ""))
    os.environ.setdefault("OPENAI_API_BASE", os.environ.get("LLM_BASE_URL", ""))
    _ENV_LOADED = True


# ---------------------------------------------------------------------------
# Slot 抽取（LLM 结构化输出）
# ---------------------------------------------------------------------------
from pydantic import BaseModel, Field  # noqa: E402
from . import quiet  # noqa: F401  (import 即静音日志，只留进度条)
from .progress import make_progress


class SlotExtraction(BaseModel):
    dish: str = Field(default="", description="菜名（如 宫保鸡丁）")
    dish_a: str = Field(default="", description="对比场景的菜 A")
    dish_b: str = Field(default="", description="对比场景的菜 B")
    ingredient: str = Field(default="", description="食材/主料（如 鸡肉）")
    cuisine: str = Field(default="", description="菜系（如 川菜）")
    category: str = Field(default="", description="类型/类目（如 汤羹、烘焙）")


def build_slot_extractor():
    from langchain_openai import ChatOpenAI
    from gustobot.config.settings import settings
    llm = ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model_name=settings.OPENAI_MODEL,
        openai_api_base=settings.OPENAI_API_BASE,
        temperature=0,
    )
    return llm.with_structured_output(SlotExtraction)


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------

async def run_route_eval(eval_set_path: str, output_path: str, limit: int | None = None):
    from langchain_core.messages import HumanMessage
    from .schema import load_eval_set
    from . import metrics

    load_env()
    # 导入真实系统的路由节点
    from gustobot.application.agents.lg_builder import analyze_and_route_query
    from gustobot.application.agents.lg_states import AgentState

    cases = load_eval_set(eval_set_path)
    if limit:
        cases = cases[:limit]

    slot_extractor = build_slot_extractor()
    config = {"configurable": {"thread_id": "route-eval"}}

    results = []
    prog = make_progress(len(cases), label="路由")
    for i, case in enumerate(cases):
        q = case.turns[0]
        state = AgentState(messages=[HumanMessage(content=q)], question=q)
        try:
            router = (await analyze_and_route_query(state, config=config))["router"]
            pred_route = router.type
        except Exception as exc:  # noqa
            pred_route = f"ERROR:{type(exc).__name__}"
            print(f"[{case.id}] 路由调用失败: {exc}")

        # slot 抽取（仅对含 dish/ingredient/cuisine 标注的样本）
        pred_slots: Dict[str, str] = {}
        if case.expected_slots:
            try:
                ext = await slot_extractor.ainvoke(q)
                pred_slots = {k: v for k, v in ext.model_dump().items() if v}
            except Exception as exc:  # noqa
                print(f"[{case.id}] slot 抽取失败: {exc}")

        results.append({
            "id": case.id,
            "scenario": case.scenario,
            "question": q,
            "expected_route": case.expected_route,
            "pred_route": pred_route,
            "expected_slots": case.expected_slots,
            "pred_slots": pred_slots,
        })
        prog.update(ok=(pred_route == case.expected_route), note=case.id)

    prog.finish()

    # ------------------------------------------------------------------
    # 指标
    # ------------------------------------------------------------------
    pred_routes = [r["pred_route"] for r in results]
    exp_routes = [r["expected_route"] for r in results]
    pred_slot_list = [r["pred_slots"] for r in results]
    exp_slot_list = [r["expected_slots"] for r in results]

    report = {
        "n": len(results),
        "intent_accuracy": metrics.intent_accuracy(pred_routes, exp_routes),
        "slot_accuracy_strict": metrics.slot_accuracy(pred_slot_list, exp_slot_list, strict=True),
        "slot_accuracy_loose": metrics.slot_accuracy(pred_slot_list, exp_slot_list, strict=False),
        "per_scenario": {},
    }

    from collections import defaultdict
    by_scenario = defaultdict(list)
    for r in results:
        by_scenario[r["scenario"]].append(r)
    for sc, rows in by_scenario.items():
        report["per_scenario"][sc] = {
            "n": len(rows),
            "intent_accuracy": metrics.intent_accuracy(
                [r["pred_route"] for r in rows], [r["expected_route"] for r in rows]
            ),
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(output_path.replace(".json", "_details.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n================ 路由评测报告 ================")
    print(f"样本数: {report['n']}")
    print(f"Intent Accuracy:      {report['intent_accuracy']:.4f}")
    print(f"Slot Accuracy(strict): {report['slot_accuracy_strict']:.4f}")
    print(f"Slot Accuracy(loose):  {report['slot_accuracy_loose']:.4f}")
    print("\n-- 分场景 Intent Accuracy --")
    for sc, v in report["per_scenario"].items():
        print(f"  {sc:16s} n={v['n']:3d} acc={v['intent_accuracy']:.4f}")
    print(f"\n详细结果: {output_path.replace('.json', '_details.jsonl')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="eval/cases/eval_set.jsonl")
    ap.add_argument("--output", default="eval/results/route_eval_report.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run_route_eval(args.eval_set, args.output, args.limit))


if __name__ == "__main__":
    main()
