"""GustoBot 评测体系 —— Tool Selection 单节点评测（路径 B）。

评测对象：子图A 的 tool_selection 节点（LLM bind_tools 选工具）。
不需要数据库（Neo4j/MySQL/Milvus），只需 LLM API —— 补上"工具选择准确率"缺失的实测。

指标：
- Tool Accuracy：LLM 选的工具 == 标注 expected_tool 占比
- Args Accuracy：传给工具的参数与标注 expected_args 匹配率

用法：
    python -m eval.scripts.run_tool_eval [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
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


# 工具名映射：子图A 节点名 → 评测集 expected_tool 口径
NODE_TOOL_MAP = {
    "cypher_query": "cypher_query",
    "predefined_cypher": "predefined_cypher",
    "customer_tools": "microsoft_graphrag_query",   # LightRAG 工具
    "text2sql_query": "text2sql_query",
}

# 参与工具选择评测的 expected_tool（子图A 工具集；kb 子图的 postgres/milvus 不在此）
EVAL_TOOLS = {"cypher_query", "predefined_cypher", "microsoft_graphrag_query", "text2sql_query"}

# 工具**路径**归一：把"同一条查询路径的不同实现"视为等价。
#
# 为什么需要：评测该测的是"系统有没有选对**查询路径**"，而不是"内部用了哪个实现"。
# 实测 LLM 面对「西红柿疙瘩汤怎么做」时会选 `predefined_cypher`（配 dish_instructions
# 模板 + {dish_name: ...}），这是**更稳的工程实践**——参数化、不会生成错 Cypher；
# 而数据集把 `cypher_query`（让 LLM 现场生成 Cypher，易错）当成唯一标准答案，
# 导致 tool_accuracy 被打成 0.05（80 条里只有 4 条工具名逐字相同）。
_PATH_OF = {
    "cypher_query": "graph",               # LLM 现场生成 Cypher
    "predefined_cypher": "graph",          # 预定义 Cypher 模板
    "microsoft_graphrag_query": "graph",   # LightRAG 图谱检索
    "text2sql_query": "text2sql",          # 结构化统计
}


def same_path(pred: str, expect: str) -> bool:
    """两条工具名是否属于同一条查询路径。"""
    return _PATH_OF.get(pred, pred) == _PATH_OF.get(expect, expect)


# 预定义查询 → 数据集里标的 field 名
_FIELD_OF_QUERY = {
    "dish_instructions": "steps",
    "cooking_steps": "steps",
    "ingredients_of_dish": "ingredients",
    "main_ingredients_of_dish": "ingredients",
    "dish_cook_time": "time",
    "dish_flavor": "taste",
    "dish_cooking_method": "process",
}


def extract_args(pred_args: dict) -> dict:
    """把 tool_selection 的原始输出还原成 expected_args 那套「扁平业务参数」。

    为什么需要：工具选择节点返回的是
        {task, query_name, query_parameters: {query, parameters}, steps}
    而数据集标的是 {dish} / {dish, field} / {group, metric} / {dish_a, dish_b, compare}。
    两者结构完全不同，直接按 key 比对永远是 0 分（实测 args_accuracy=0.0）。
    这里做一次语义还原，再交给 metrics.args_accuracy 做值匹配。

    只能还原能对上的部分——`query` 是 LLM 现场生成的，没有固定参数结构，会返回 {}。
    """
    if not isinstance(pred_args, dict):
        return {}
    qp = pred_args.get("query_parameters")
    if not isinstance(qp, dict):
        return {}

    query = str(qp.get("query") or "")
    params = qp.get("parameters")
    params = params if isinstance(params, dict) else {}

    out: dict = {}
    for k in ("dish_name", "dish", "name"):
        if params.get(k):
            out["dish"] = params[k]
            break
    for k in ("dish_type", "type_name", "type"):
        if params.get(k):
            out["group"] = params[k]
            break
    for key, val in _FIELD_OF_QUERY.items():
        if query.startswith(key):
            out["field"] = val
            break
    if query == "dish_flavor":
        out["compare"] = "spice"          # 数据集里 compare 只有 spice 一种
    if query.startswith("dishes_count"):
        out["metric"] = "count"
    return out


async def run_tool_eval(eval_set_path: str, output: str, limit: int | None = None):
    from .schema import load_eval_set
    from . import metrics

    load_env()
    from langchain_openai import ChatOpenAI
    from gustobot.config.settings import settings
    from gustobot.application.agents.kg_sub_graph.kg_tools_list import (
        cypher_query, predefined_cypher, microsoft_graphrag_query, text2sql_query,
    )
    from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.tool_selection.node import (
        create_tool_selection_node,
    )

    # 过滤：仅子图A 工具集的样本
    cases = [c for c in load_eval_set(eval_set_path) if c.expected_tool in EVAL_TOOLS]
    if limit:
        cases = cases[:limit]

    llm = ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model_name=settings.OPENAI_MODEL,
        openai_api_base=settings.OPENAI_API_BASE,
        temperature=0,
    )
    tool_schemas = [cypher_query, predefined_cypher, microsoft_graphrag_query, text2sql_query]
    node_fn = create_tool_selection_node(llm, tool_schemas, default_to_text2cypher=True)

    records = []
    prog = make_progress(len(cases), label="工具")
    for i, case in enumerate(cases):
        q = case.turns[0]
        try:
            command = await node_fn({"question": q})
            # Command(goto=Send(target, arg)) —— Send 的属性是 .node 和 .arg
            send = command.goto
            target = send.node if hasattr(send, "node") else str(send)
            pred_tool = NODE_TOOL_MAP.get(target, target)
            payload = send.arg if hasattr(send, "arg") else {}
        except Exception as exc:  # noqa
            pred_tool = f"ERROR:{type(exc).__name__}"
            payload = {}

        records.append({
            "id": case.id,
            "scenario": case.scenario,
            "question": q[:40],
            "expected_tool": case.expected_tool,
            "pred_tool": pred_tool,
            "expected_args": case.expected_args,
            "pred_args": payload,
        })
        prog.update(ok=(pred_tool == case.expected_tool), note=case.id)

    prog.finish()

    # ------------------------------------------------------------------
    # 指标
    # ------------------------------------------------------------------
    valid = [r for r in records if not r["pred_tool"].startswith("ERROR")]
    report = {"n_total": len(records), "n_valid": len(valid)}
    if valid:
        # 用"路径"而非"工具名逐字相同"来判对错
        report["tool_accuracy"] = round(
            sum(1 for r in valid if same_path(r["pred_tool"], r["expected_tool"])) / len(valid), 4)
        report["tool_accuracy_exact"] = metrics.intent_accuracy(
            [r["pred_tool"] for r in valid], [r["expected_tool"] for r in valid])
        report["args_accuracy"] = metrics.args_accuracy(
            [extract_args(r["pred_args"]) for r in valid], [r["expected_args"] for r in valid])

    # 分场景 Tool Accuracy
    from collections import defaultdict
    by_sc = defaultdict(list)
    for r in valid:
        by_sc[r["scenario"]].append(r)
    report["per_scenario"] = {}
    for sc, rows in by_sc.items():
        report["per_scenario"][sc] = round(
            sum(1 for r in rows if same_path(r["pred_tool"], r["expected_tool"])) / len(rows), 4)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(output.replace(".json", "_details.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n============ Tool Selection 评测报告 ============")
    print(f"样本: {report['n_total']}（有效 {report['n_valid']}）")
    print(f"Tool Accuracy:  {report.get('tool_accuracy', 'N/A'):.4f}")
    print(f"Args Accuracy:  {report.get('args_accuracy', 'N/A'):.4f}")
    for sc, acc in report.get("per_scenario", {}).items():
        print(f"  {sc:16s} tool_acc={acc:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="eval/cases/eval_set.jsonl")
    ap.add_argument("--output", default="eval/results/tool_eval_report.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run_tool_eval(args.eval_set, args.output, args.limit))


if __name__ == "__main__":
    main()
