"""GustoBot 评测体系 —— 端到端评测（Tool / Args / 回答层指标）。

评测对象：完整 Agent 工作流（LangGraph 主图 graph.invoke）。
依赖完整运行环境（LLM API + Neo4j + MySQL + Milvus + Redis）。

指标：
- Intent Accuracy（端到端路由）
- Effective Tool Coverage / Args Accuracy
- 回答层：key_fact_coverage（关键事实覆盖率）/ rejection_accuracy（拒答正确率）/ citation_consistency（引用一致性）

用法：
    python -m eval.scripts.run_e2e_eval [--eval-set eval/cases/eval_set.jsonl] [--limit 20]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional
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


async def run_e2e_eval(eval_set_path: str, limit: Optional[int], output: str):
    from .schema import load_eval_set
    from . import metrics

    load_env()
    from langchain_core.messages import HumanMessage
    from gustobot.application.agents.lg_builder import graph, prepare_checkpointer
    from gustobot.application.agents.lg_states import AgentState

    # 评测直接调 graph，绕过了 FastAPI 的 startup_event；
    # 这里必须先把 AsyncPostgresSaver 的连接池打开，否则每条都会
    # 报 "the pool 'pool-1' is not open yet"。
    await prepare_checkpointer()

    cases = load_eval_set(eval_set_path)
    if limit:
        cases = cases[:limit]

    records = []
    prog = make_progress(len(cases), label="端到端")
    for i, case in enumerate(cases):
        messages = [HumanMessage(content=t) for t in case.turns]
        state = AgentState(messages=messages, question=case.turns[-1])
        config = {"configurable": {"thread_id": f"e2e-{case.id}"}}
        t0 = time.time()
        try:
            result = await graph.ainvoke(state, config=config)
            pred_route = getattr(result.get("router"), "type", "") or ""
            answer = result.get("answer", "") or ""
            if not answer:
                # 回答可能写在 messages 的最后一条 AIMessage 里（如 general/additional 节点）
                for msg in reversed(result.get("messages", []) or []):
                    if getattr(msg, "type", "") == "ai" and msg.content:
                        answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                        break
            sources = result.get("sources", []) or []
            env_error = False
        except Exception as exc:  # noqa
            pred_route = "ERROR"
            answer = f"ERROR: {exc}"
            sources = []
            err_text = str(exc).lower()
            env_error = any(w in err_text for w in (
                "connect", "connection", "timed out", "refused",
                "milvus", "neo4j", "unavailable", "nonetype", "attribute",
                "pool", "not open", "not json serializable", "checkpointer",
            ))
        elapsed = time.time() - t0

        records.append({
            "id": case.id,
            "scenario": case.scenario,
            "expected_route": case.expected_route,
            "expected_tool": case.expected_tool,
            "expected_args": case.expected_args,
            "expected_keywords": case.expected_answer_keywords,
            "pred_route": pred_route,
            "answer": answer[:500],
            "sources": sources[:5],
            "env_error": env_error,
            "elapsed_s": round(elapsed, 1),
        })

        # 进度条（\r 刷新；重定向文件时每 5 条换行一次）
        # 统一进度条（含已用/剩余时间）
        prog.update(note=case.id)

    prog.finish()

    # ------------------------------------------------------------------
    # 指标：排除环境失败的样本
    # ------------------------------------------------------------------
    valid = [r for r in records if not r["env_error"]]
    report = {
        "n_total": len(records),
        "n_env_failed": len(records) - len(valid),
        "env_failed_ids": [r["id"] for r in records if r["env_error"]],
        "n_scored": len(valid),
    }
    if valid:
        report["intent_accuracy"] = metrics.intent_accuracy(
            [r["pred_route"] for r in valid], [r["expected_route"] for r in valid])
        # 删掉 effective_tool_coverage：原实现两个参数都传 expected_tool
        #（拿标准答案和标准答案比，恒等于 1.0，等于没测）。
        # 而 e2e 层面**拿不到**"实际用了哪个工具"——父图 state 里没有这个字段，
        # 工具选择发生在子图内部且不回传。工具选择已由 run_tool_eval 专门评测，
        # 这里不再放一个算不出来的指标。
        # 回答层指标
        report["key_fact_coverage"] = metrics.pipeline_success_rate(
            [r["answer"] for r in valid], [r["expected_keywords"] for r in valid])
        neg = [r for r in valid if r["scenario"] == "negative"]
        if neg:
            reject_ok = 0
            for r in neg:
                ans = (r["answer"] or "").lower()
                if any(w in ans for w in ("抱歉", "不属于", "不在", "范围", "不好意思", "无法")):
                    reject_ok += 1
                elif any(w in ans for w in ("请问", "具体", "想吃什么", "告诉我", "什么菜", "怎么帮")):
                    reject_ok += 1
            report["rejection_accuracy"] = round(reject_ok / len(neg), 4)
        cited = [r for r in valid if re.search(r"\[\d+\]", r["answer"])]
        if cited:
            consistent = 0
            for r in cited:
                refs = [int(x) for x in re.findall(r"\[(\d+)\]", r["answer"])]
                if refs and max(refs) <= len(r["sources"]):
                    consistent += 1
            report["citation_consistency"] = round(consistent / len(cited), 4)
            report["n_cited"] = len(cited)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(output.replace(".json", "_details.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n============ 端到端评测报告 ============")
    print(f"总数 {report['n_total']}  环境失败 {report['n_env_failed']}  有效 {report['n_scored']}")
    for k, v in report.items():
        if k not in ("env_failed_ids", "n_total", "n_env_failed", "n_scored", "n_cited"):
            print(f"  {k}: {v}")
    if "n_cited" in report:
        print(f"  引用一致性统计样本: {report['n_cited']} 条（仅含 [n] 引用的回答）")
    print(f"环境失败用例: {report['env_failed_ids']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="eval/cases/eval_set.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", default="eval/results/e2e_eval_report.json")
    args = ap.parse_args()
    asyncio.run(run_e2e_eval(args.eval_set, args.limit, args.output))


if __name__ == "__main__":
    main()
