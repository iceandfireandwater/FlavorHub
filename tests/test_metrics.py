"""tests/test_metrics.py —— 指标计算单元测试（保证指标口径正确可复现）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import metrics


def test_intent_accuracy():
    assert metrics.intent_accuracy(["a", "b", "a"], ["a", "b", "b"]) == 2 / 3
    assert metrics.intent_accuracy([], []) == 0.0
    assert metrics.intent_accuracy(["a"], ["a"]) == 1.0


def test_slot_accuracy_strict():
    preds = [{"dish": "宫保鸡丁"}, {"dish": "鱼香肉丝"}, {}]
    labels = [{"dish": "宫保鸡丁"}, {"dish": "麻婆豆腐"}, {}]
    assert metrics.slot_accuracy(preds, labels, strict=True) == 2 / 3


def test_slot_accuracy_loose():
    preds = [{"dish": "宫保鸡丁", "extra": "x"}]
    labels = [{"dish": "宫保鸡丁"}]
    assert metrics.slot_accuracy(preds, labels, strict=False) == 1.0
    assert metrics.slot_accuracy(preds, labels, strict=True) == 0.0


def test_tool_coverage():
    preds = ["cypher_query", "cypher_query", "text2sql_query"]
    labels = ["cypher_query", "cypher_query", "text2sql_query"]
    effective = [True, False, True]
    assert metrics.tool_coverage(preds, labels, effective) == 2 / 3


def test_args_accuracy():
    preds = [{"dish": "宫保鸡丁"}, {"dish_a": "A", "dish_b": "B"}, {}]
    labels = [{"dish": "宫保鸡丁"}, {"dish_a": "A", "dish_b": "B"}, {}]
    assert metrics.args_accuracy(preds, labels) == 1.0
    preds2 = [{"dish": "宫保鸡"}]
    labels2 = [{"dish": "宫保鸡丁"}]
    assert metrics.args_accuracy(preds2, labels2) == 1.0  # 包含匹配


def test_recall_at_k():
    preds = [["a", "b", "c"], ["x", "y", "z"], ["m"]]
    rels = [["b"], ["z"], []]
    assert metrics.recall_at_k(preds, rels) == 1.0
    assert metrics.recall_at_k(preds, rels, k=1) == 0.0  # relevant 均在 top-1 之外


def test_pipeline_success_rate():
    preds = ["做法：先切丁，再爆炒花生", "抱歉没有相关信息", ""]
    kws = [["切丁"], ["花生"], []]
    assert metrics.pipeline_success_rate(preds, kws) == 1 / 2
