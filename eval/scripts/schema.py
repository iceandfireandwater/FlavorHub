"""GustoBot 评测体系 —— 评测样本数据模型与常量定义。

对标电商客服 Agent 评测集的标注结构：
intent / tool / args / slots / relevant_ids / topic，并补充可追溯的 ground truth。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any


# ---------------------------------------------------------------------------
# 任务场景（对标电商客服 7 类任务）
# ---------------------------------------------------------------------------
SCENARIOS = {
    "recipe_search": "菜谱搜索（做法/步骤）",
    "recipe_detail": "菜谱详情（食材/耗时/口味/工艺）",
    "recipe_compare": "菜谱对比",
    "stat_query": "统计查询（Text2SQL）",
    "history_faq": "历史文化典故问答（FAQ/政策）",
    "multi_turn": "多轮记忆",
    "negative": "越界/模糊输入（Guardrails 负样本）",
}

# 真实系统支持的路由类型（gustobot/application/agents/lg_states.py Router.type）
ROUTES = [
    "general-query",
    "additional-query",
    "kb-query",
    "graphrag-query",
    "image-query",
    "file-query",
    "text2sql-query",
]

# 真实系统的原子工具（kg_tools_list.py + kb 检索工具）
TOOLS = [
    "cypher_query",
    "predefined_cypher",
    "microsoft_graphrag_query",  # 实际由 LightRAG 实现
    "text2sql_query",
    "postgres",   # pgvector 检索
    "milvus",     # 向量库兜底
    "external",   # 外部搜索
    "none",       # 纯 LLM（general-query）
]


@dataclass
class EvalCase:
    """一条评测样本。

    turns: 多轮场景包含多轮用户消息；单轮场景只有 1 条。
    """
    id: str
    scenario: str                       # SCENARIOS 的 key
    turns: List[str]                    # 用户消息序列
    expected_route: str                 # intent（路由类型）
    expected_tool: str = "none"         # 期望调用的工具
    expected_args: Dict[str, Any] = field(default_factory=dict)   # 工具参数
    expected_slots: Dict[str, str] = field(default_factory=dict)  # 槽位（dish/ingredient/cuisine...）
    relevant_ids: List[str] = field(default_factory=list)         # 检索应命中的知识条目 id
    topic: str = ""                     # 知识主题（kb 段落 topic）
    expected_answer_keywords: List[str] = field(default_factory=list)  # 答案关键信息点
    ground_truth: str = ""              # 标准答案原文（可回溯）
    gt_source: str = ""                 # ground truth 来源（如 recipe.json:菜名.做法）
    negative: bool = False              # 是否为负样本（期望被拒绝/追问）
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalCase":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # noqa
        return cls(**{k: v for k, v in d.items() if k in known})


def load_eval_set(path: str) -> List[EvalCase]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(EvalCase.from_dict(json.loads(line)))
    return cases


def save_eval_set(path: str, cases: List[EvalCase]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(c.to_json() + "\n")
