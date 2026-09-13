"""GustoBot 评测体系 —— 指标计算。

对标电商客服评测指标（口径说明，保证可复现）：
- Intent Accuracy      系统路由类型 == 标注 intent 的占比
- Slot Accuracy        系统抽取槽位与标注全匹配（严格）/ 部分匹配（宽松）的占比
- Effective Tool Coverage  系统选中的工具 == 标注 tool 且执行成功的占比
- Effective Args Accuracy  系统传给工具的参数与标注 args 匹配的占比
- Recall@K             检索结果 Top-K 中含 relevant_ids 的占比
- Pipeline Success Rate    端到端回答包含标注关键信息点的占比
"""
from __future__ import annotations

import re

from typing import Any, Dict, List, Optional, Sequence


def intent_accuracy(preds: Sequence[str], labels: Sequence[str]) -> float:
    assert len(preds) == len(labels)
    if not preds:
        return 0.0
    return sum(1 for p, l in zip(preds, labels) if p == l) / len(preds)


def slot_accuracy(
    preds: Sequence[Dict[str, str]],
    labels: Sequence[Dict[str, str]],
    strict: bool = True,
) -> float:
    """槽位匹配：strict=全部 key 完全匹配；否则按 key 覆盖率（宽松，值允许包含匹配）。"""
    assert len(preds) == len(labels)
    if not preds:
        return 0.0
    scores = []
    for p, l in zip(preds, labels):
        if not l:  # 标注无槽位（如 history_faq/negative），抽取为空即算对
            scores.append(1.0 if not p else 0.0)
            continue
        if strict:
            ok = set(p.keys()) == set(l.keys()) and all(
                str(p.get(k, "")).strip() == str(v).strip() for k, v in l.items()
            )
            scores.append(1.0 if ok else 0.0)
        else:
            matched = sum(
                1 for k, v in l.items()
                if _value_match(p.get(k, ""), v)
            )
            scores.append(matched / len(l))
    return sum(scores) / len(scores)


def _value_match(pred: object, label: object) -> bool:
    """值匹配：全等或双向包含（菜名修饰词差异视为匹配，如 '酱牛肉' ∈ '超简单秘制酱牛肉'）。"""
    p, l = str(pred).strip(), str(label).strip()
    if not p or not l:
        return p == l
    return p == l or p in l or l in p


def tool_coverage(preds: Sequence[str], labels: Sequence[str], effective: Sequence[bool]) -> float:
    """Effective Tool Coverage：选中标注工具且执行成功的占比。"""
    assert len(preds) == len(labels) == len(effective)
    if not preds:
        return 0.0
    ok = sum(1 for p, l, e in zip(preds, labels, effective) if p == l and e)
    return ok / len(preds)


def args_accuracy(preds: Sequence[Dict], labels: Sequence[Dict]) -> float:
    """Effective Args Accuracy：参数值匹配率。"""
    assert len(preds) == len(labels)
    if not preds:
        return 0.0
    scores = []
    for p, l in zip(preds, labels):
        if not l:
            scores.append(1.0)
            continue
        matched = 0
        for k, v in l.items():
            pv = str(p.get(k, "")).strip()
            lv = str(v).strip()
            if pv == lv or (lv and lv in pv) or (pv and pv in lv):
                matched += 1
        scores.append(matched / len(l))
    return sum(scores) / len(scores)


def recall_at_k(
    pred_ids: Sequence[List[str]],
    relevant_ids: Sequence[List[str]],
    k: Optional[int] = None,
) -> float:
    """Recall@K：检索结果 Top-K 中含任意 relevant id 的占比。"""
    assert len(pred_ids) == len(relevant_ids)
    if not pred_ids:
        return 0.0
    hits = 0
    for preds, rels in zip(pred_ids, relevant_ids):
        if not rels:
            continue  # 无相关标注的样本不计入
        preds_k = preds[:k] if k else preds
        if any(r in preds_k for r in rels):
            hits += 1
    scored = sum(1 for rels in relevant_ids if rels)
    return hits / scored if scored else 0.0


def precision_at_k(
    pred_ids: Sequence[List[str]],
    relevant_ids: Sequence[List[str]],
    k: int,
) -> float:
    """Precision@K：前 K 个结果中相关文档占比（相关文档稀疏时天然偏低，仅作参考）。"""
    assert len(pred_ids) == len(relevant_ids)
    scored = [i for i, rels in enumerate(relevant_ids) if rels]
    if not scored:
        return 0.0
    total = 0.0
    for i in scored:
        preds_k = pred_ids[i][:k]
        total += sum(1 for d in preds_k if d in relevant_ids[i]) / k
    return total / len(scored)


def mrr(pred_ids: Sequence[List[str]], relevant_ids: Sequence[List[str]], k: Optional[int] = None) -> float:
    """MRR：第一个相关文档位置的倒数，取平均（衡量"第一名精度"）。"""
    assert len(pred_ids) == len(relevant_ids)
    scored = [i for i, rels in enumerate(relevant_ids) if rels]
    if not scored:
        return 0.0
    total = 0.0
    for i in scored:
        preds_k = pred_ids[i][:k] if k else pred_ids[i]
        rr = 0.0
        for rank, doc in enumerate(preds_k):
            if doc in relevant_ids[i]:
                rr = 1.0 / (rank + 1)
                break
        total += rr
    return total / len(scored)


def ndcg_at_k(pred_ids: Sequence[List[str]], relevant_ids: Sequence[List[str]], k: int) -> float:
    """nDCG@K：按相关文档位置算 DCG/IDCG（衡量整体排序质量）。"""
    import math
    assert len(pred_ids) == len(relevant_ids)
    scored = [i for i, rels in enumerate(relevant_ids) if rels]
    if not scored:
        return 0.0

    def _dcg(rel_positions, kk):
        # rel_positions: 1-based 的相关位置集合
        return sum(1 / math.log2(pos + 1) for pos in rel_positions if pos <= kk)

    total = 0.0
    for i in scored:
        preds_k = pred_ids[i][:k]
        rel_positions = set(rank + 1 for rank, d in enumerate(preds_k) if d in relevant_ids[i])
        dcg_val = _dcg(rel_positions, k)
        ideal = _dcg(set(range(1, min(len(relevant_ids[i]), k) + 1)), k)
        total += dcg_val / ideal if ideal else 0.0
    return total / len(scored)


_PUNCT_RE = re.compile(r"[\s，。、；：！？,.;:!?「」『』《》\"'“”‘’（）()【】\[\]\-—～~…·]+")


def _norm_text(s: Any) -> str:
    """去标点/空白，用于宽松匹配。"""
    return _PUNCT_RE.sub("", str(s or ""))


def _keyword_hit(ans: str, kw: Any) -> bool:
    """关键词是否被回答覆盖。

    判据**不能**要求"原样包含整句原文"：模型会改写措辞，实测差一个「的」字就判失败
    （96 条典故题里 80 条是这种误判，覆盖率被压到 0.02）。改成两层：
      ① 去标点后整体包含 —— 处理标点/空白差异
      ② 或把关键词按标点切成片段，命中 >= 60% 即算覆盖 —— 处理"答到要点但换了说法"
    """
    kw_s = str(kw or "")
    kw_norm = _norm_text(kw_s)
    if not kw_norm:
        return False
    ans_norm = _norm_text(ans)
    if kw_norm in ans_norm:
        return True
    parts = [p for p in _PUNCT_RE.split(kw_s) if len(_norm_text(p)) >= 2]
    if not parts:
        return False
    hits = sum(1 for p in parts if _norm_text(p) in ans_norm)
    return hits / len(parts) >= 0.6


def pipeline_success_rate(preds: Sequence[str], keywords_list: Sequence[List[str]]) -> float:
    """Pipeline Success Rate：答案覆盖标注关键信息点的占比（宽松匹配，见 _keyword_hit）。"""
    assert len(preds) == len(keywords_list)
    if not preds:
        return 0.0
    ok = 0
    for ans, kws in zip(preds, keywords_list):
        if not kws:
            continue
        if any(_keyword_hit(ans, kw) for kw in kws):
            ok += 1
    scored = sum(1 for kws in keywords_list if kws)
    return ok / scored if scored else 0.0


def summarize(preds: Sequence[str], labels: Sequence[str]) -> Dict[str, float]:
    """分场景混淆矩阵摘要。"""
    from collections import Counter
    conf = Counter(zip(preds, labels))
    return {
        "n": len(preds),
        "intent_accuracy": round(intent_accuracy(preds, labels), 4),
        "confusions": {f"{p}->{l}": n for (p, l), n in conf.items() if p != l},
    }
