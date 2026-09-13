"""tests/test_rrf.py —— 混合检索 RRF（Reciprocal Rank Fusion）算法单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.run_retrieval_eval import keyword_score, rrf_fuse, tokenize


def test_tokenize_chinese():
    toks = tokenize("宫保鸡丁怎么做")
    assert "宫保" in toks and "鸡丁" in toks and "怎么做" in toks or any("做" in t for t in toks)
    assert tokenize("GPT4 教程")  # ASCII 词


def test_keyword_score_higher_for_match():
    index = {
        "doc1": {t: 1 for t in tokenize("宫保鸡丁 鸡肉 花生 辣椒 做法 步骤")},
        "doc2": {t: 1 for t in tokenize("红烧肉 五花肉 糖色 做法")},
    }
    s1 = keyword_score("宫保鸡丁怎么做", "doc1", index)
    s2 = keyword_score("宫保鸡丁怎么做", "doc2", index)
    assert s1 > s2


def test_rrf_fuse_combines_rankings():
    vec = ["a", "b", "c", "d"]
    kw = ["d", "c", "b", "a"]
    fused = rrf_fuse(vec, kw)
    scores = dict(fused)
    # 双路互补对称：a 与 d 并列最高，b 与 c 并列次高（验证融合正确合并两路排序）
    assert scores["a"] == scores["d"] > scores["b"] == scores["c"]


def test_rrf_boost_topic():
    vec = ["kb_para_0", "kb_para_1", "kb_para_2"]
    kw = ["kb_para_2", "kb_para_0", "kb_para_1"]
    fused = rrf_fuse(vec, kw, boost_ids={"kb_para_1": 5.0})
    assert fused[0][0] == "kb_para_1"  # 置顶生效


def test_rrf_empty():
    assert rrf_fuse([], []) == []
