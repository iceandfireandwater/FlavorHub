"""GustoBot 评测体系 —— 检索评测（纯向量检索）。

评测对象：
- 候选文档池 = recipe.json 菜谱 + data/kb/data.txt 段落 + **Milvus 古籍译文**
  （古籍必须入池，否则古籍题的 relevant_ids 不在池里、指标恒为 0）
- 使用真实 Embedding API（text-embedding-v3, dim=1024）
- 指标：Recall@K / MRR@K / nDCG@K

**只用向量通道**：生产链路的 knowledge_service.search 本身就只有向量检索，
评测里再算一套 RRF 混合等于测一个没人用的东西（实测 RRF 普遍还更低）。

用法：
    python -m eval.scripts.run_retrieval_eval [--pool-size 600] [--top-k 5 10]
"""
from __future__ import annotations
from . import quiet  # noqa: F401  (import 即静音日志，只留进度条)
from .progress import make_progress

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------
def load_env() -> None:
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# 中文分词（无 jieba 依赖：按字符 bigram + 连续 ASCII 词）
# ---------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    tokens = []
    # 连续 ASCII 词
    for w in re.findall(r"[A-Za-z0-9_]+", text):
        tokens.append(w.lower())
    # 中文字符 bigram（覆盖大部分中文查询/文档）
    han = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.extend("".join(han[i:i + 2]) for i in range(len(han) - 1))
    tokens.extend(han)  # unigram
    return tokens
def keyword_score(query: str, doc: str, index: Dict[str, Dict[str, int]]) -> float:
    """简化 BM25：query 词在文档中的 TF 加权求和（文档长度归一）。"""
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0
    doc_tokens = index.get(doc, {})
    score = 0.0
    for t in q_tokens:
        tf = doc_tokens.get(t, 0)
        if tf > 0:
            score += (1 + math.log(1 + tf))
    dl = sum(doc_tokens.values()) or 1
    return score / math.sqrt(dl)


# ---------------------------------------------------------------------------
# 知识库文档构建（真实数据）
# ---------------------------------------------------------------------------
def load_classic_documents(needed: set, extra: int = 60) -> List[dict]:
    """从 Milvus 拉「古籍译文」进候选池。

    为什么必须加：原实现只从 `recipe.json` + `data.txt` 建池（实测 504 条），
    而 8 本古籍的 2454 条译文**只存在 Milvus 里**。于是古籍题的 relevant_ids
    （`classics_*`）根本不在池中，`recall@k / MRR / nDCG` 恒为 0 —— 指标是假的。

    - 评测用到的古籍全拿（保证 relevant_ids 在池里）
    - 另外随机取 `extra` 条无关古籍当干扰项，避免"池子里只有正确答案"导致虚高
    """
    try:
        from pymilvus import MilvusClient
    except Exception as exc:  # noqa: BLE001
        print(f"  !! 未安装 pymilvus，古籍无法入池: {exc}", flush=True)
        return []

    uri = os.getenv("MILVUS_URI") or ("http://milvus:19530" if os.path.exists("/.dockerenv") else "http://localhost:19530")
    try:
        cli = MilvusClient(uri=uri)
        cli.load_collection("recipes")
        rows = cli.query(
            "recipes", filter="category == '古籍译文'",
            output_fields=["recipe_id", "name", "content"], limit=16000,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  !! 拉取古籍失败（{uri}）: {exc}", flush=True)
        return []

    import random
    by_id = {r["recipe_id"]: r for r in rows if r.get("recipe_id")}
    hit = [by_id[k] for k in needed if k in by_id]
    miss = sorted(k for k in needed if k not in by_id)
    if miss:
        print(f"  !! 有 {len(miss)} 个 relevant_id 在 Milvus 里找不到: {miss[:5]}", flush=True)

    rest = [r for r in rows if r["recipe_id"] not in needed]
    random.seed(7)
    distractors = random.sample(rest, min(extra, len(rest))) if rest else []

    out = []
    for r in hit + distractors:
        out.append({
            "id": r["recipe_id"],
            "text": (r.get("content") or "").strip(),
            "kind": "classic",
            "topic": (r.get("name") or "").strip(),
        })
    return out


def build_kb_documents(recipe_db: Dict[str, dict], kb_paras: List[str]) -> List[dict]:
    docs = []
    for name, entry in recipe_db.items():
        if not isinstance(entry, dict):
            continue
        parts = [name]
        for grp in ("主食材", "辅料"):
            for item in entry.get(grp, []):
                if isinstance(item, list) and item:
                    parts.append(str(item[0]))
        parts.extend([entry.get("口味", ""), entry.get("工艺", ""), entry.get("做法", "")])
        text = " ".join(p for p in parts if p)
        # topic = 菜名本身：query 中的菜名可精确命中置顶（FAQ/菜名短关键词对齐）
        docs.append({"id": name, "text": text, "kind": "recipe", "topic": name})
    for i, p in enumerate(kb_paras):
        topic = p.split("。")[0][:30]
        docs.append({"id": f"kb_para_{i}", "text": p, "kind": "kb", "topic": topic})
    return docs


# ---------------------------------------------------------------------------
# 短关键词评测子集
# ---------------------------------------------------------------------------
def _build_short_query_cases(retrieval_cases) -> List[dict]:
    """为每条检索样本生成短关键词 query（模拟 FAQ 短词/口语化输入）：
    - recipe 类：直接用菜名（如 '宫保鸡丁'）
    - history_faq：用段落 topic 核心词（如 '明朝蔬菜'）
    """
    out = []
    for c in retrieval_cases:
        if c.scenario in ("recipe_search", "recipe_detail", "recipe_compare"):
            names = []
            for k in ("dish", "dish_a", "dish_b"):
                v = c.expected_slots.get(k)
                if v:
                    names.append(v)
            q = "".join(names[:2])
        else:  # history_faq
            topic = c.topic or ""
            q = topic[:10] if topic else c.turns[0][:8]
        out.append({
            "id": f"{c.id}_short",
            "scenario": c.scenario,
            "question": q,
            "relevant_ids": c.relevant_ids,
        })
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
async def main_async(pool_size: int, top_ks: List[int], eval_set_path: str):
    from .schema import load_eval_set

    load_env()
    import json as _json
    recipe_db = _json.load(open(os.path.join(PROJECT_ROOT, "data/recipe.json"), encoding="utf-8"))
    kb_text = open(os.path.join(PROJECT_ROOT, "data/kb/data.txt"), encoding="utf-8").read()
    from .dataset import split_kb_paragraphs
    kb_paras = split_kb_paragraphs(kb_text)

    cases = load_eval_set(eval_set_path)
    # 需要检索的样本：recipe_search / recipe_detail / recipe_compare / history_faq
    retrieval_cases = [
        c for c in cases
        if c.scenario in ("recipe_search", "recipe_detail", "recipe_compare", "history_faq")
    ]
    # 相关文档集合（必须包含在候选池中）
    needed = set()
    for c in retrieval_cases:
        needed.update(c.relevant_ids)

    # 候选池：相关文档 + 随机抽样其他菜谱 + 全部 kb 段落
    import random
    random.seed(7)
    other_names = [n for n in recipe_db if n not in needed]
    random.shuffle(other_names)
    pool_names = list(needed & set(recipe_db.keys())) + other_names[:max(0, pool_size - len(needed) - len(kb_paras))]
    pool_ids = set(pool_names) | set(f"kb_para_{i}" for i in range(len(kb_paras)))
    docs = [d for d in build_kb_documents(
        {n: recipe_db[n] for n in pool_names},
        kb_paras,
    ) if d["id"] in pool_ids]
    # 确保 kb 段落都在
    kb_docs = [d for d in docs if d["kind"] == "kb"]

    # 古籍（Milvus）：原实现漏掉了它们，导致古籍题的检索指标恒为 0
    classic_docs = load_classic_documents(needed)
    docs = docs + classic_docs

    n_recipe = sum(1 for d in docs if d["kind"] == "recipe")
    print(f"候选文档池: {len(docs)} 条（菜谱 {n_recipe} + kb 段落 {len(kb_docs)} "
          f"+ 古籍 {len(classic_docs)}）", flush=True)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    # 向量通道：优先真实 Embedding API（text-embedding-v3）；不可用（欠费/未配置）
    # 时自动降级为本地 TF-IDF 向量（sklearn，无外部依赖，结果可复现）。
    # ------------------------------------------------------------------
    from sklearn.feature_extraction.text import TfidfVectorizer

    vector_backend = "local-tfidf"
    doc_texts = [d["text"][:1500] for d in docs]

    try:
        from openai import OpenAI
        _client = OpenAI(
            api_key=os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("EMBEDDING_BASE_URL") or os.environ.get("OPENAI_API_BASE"),
        )
        _model = os.environ.get("EMBEDDING_MODEL", "text-embedding-v3")

        def _api_embed(texts: List[str]) -> List[List[float]]:
            out = []
            for i in range(0, len(texts), 10):  # 该 Embedding API 限制 batch <= 10
                batch = texts[i:i + 10]
                resp = _client.embeddings.create(model=_model, input=batch)
                out.extend(r.embedding for r in resp.data)
            return out

        probe = _api_embed([doc_texts[0]])
        vector_backend = f"api:{_model}"

        def embed(texts: List[str]) -> List[List[float]]:
            out = []
            for i in range(0, len(texts), 10):
                batch = texts[i:i + 10]
                for attempt in range(3):
                    try:
                        resp = _client.embeddings.create(model=_model, input=batch)
                        out.extend(r.embedding for r in resp.data)
                        break
                    except Exception as e:  # noqa
                        print(f"embedding 失败(重试 {attempt + 1}/3): {e}", flush=True)
                        time.sleep(2)
                else:
                    raise RuntimeError("embedding 调用失败")
            return out

    except Exception as e:  # noqa  API 不可用 → 本地 TF-IDF
        print(f"Embedding API 不可用（{type(e).__name__}），降级为本地 TF-IDF 向量", flush=True)
        vector_backend = "local-tfidf"
        _tfidf = TfidfVectorizer(analyzer=tokenize)

        def embed(texts: List[str]) -> List[List[float]]:
            return _tfidf.transform(texts).toarray().tolist()

    print("编码文档向量 ...", flush=True)
    t0 = time.time()
    if vector_backend == "local-tfidf":
        _tfidf = TfidfVectorizer(analyzer=tokenize)
        doc_vecs = _tfidf.fit_transform(doc_texts).toarray().tolist()
    else:
        doc_vecs = embed(doc_texts)
    print(f"文档编码完成: {len(docs)} 条 (backend={vector_backend}), 耗时 {time.time() - t0:.1f}s", flush=True)

    # keyword 索引（TF）

    # topic 索引（菜名 + kb 段落主题，用于精确置顶）
    topic_index = {}
    for d in docs:
        if d["topic"]:
            for t in set(tokenize(d["topic"])):
                topic_index.setdefault(t, set()).add(d["id"])

    import numpy as np
    mat = np.array(doc_vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms == 0, 1, norms)

    # ------------------------------------------------------------------
    # 逐 query 检索
    # ------------------------------------------------------------------
    # 检索评测（完整问句）
    # ------------------------------------------------------------------
    print("检索评测（完整问句） ...", flush=True)
    q_texts = [c.turns[0] for c in retrieval_cases]
    q_vecs = embed(q_texts)
    q_mat = np.array(q_vecs, dtype=np.float32)
    q_norms = np.linalg.norm(q_mat, axis=1, keepdims=True)
    q_mat = q_mat / np.where(q_norms == 0, 1, q_norms)
    sims = q_mat @ mat.T  # (nq, ndocs)

    records = []
    prog = make_progress(len(retrieval_cases), label="检索(完整问句)")
    for i, case in enumerate(retrieval_cases):
        q = case.turns[0]
        rels = case.relevant_ids
        # 纯向量检索（RRF 已移除：生产链路的 knowledge_service.search 本来就只有向量通道）
        order = np.argsort(-sims[i])
        vec_top = [docs[j]["id"] for j in order]

        records.append({
            "id": case.id,
            "scenario": case.scenario,
            "question": q,
            "relevant_ids": rels,
            "vec_top5": vec_top[:5],
            "hit_vec": any(r in vec_top[:max(top_ks)] for r in rels),
        })
        prog.update(note=case.id)

    # ------------------------------------------------------------------
    # 检索评测（短关键词子集）：模拟 FAQ 短关键词/口语化 query，
    # 短关键词子集：模拟 FAQ 里用户只打几个词的情况。
    # ------------------------------------------------------------------
    print("检索评测（短关键词子集） ...", flush=True)
    short_cases = _build_short_query_cases(retrieval_cases)
    short_q = [c["question"] for c in short_cases]
    sq_vecs = embed(short_q)
    sq_mat = np.array(sq_vecs, dtype=np.float32)
    sq_norms = np.linalg.norm(sq_mat, axis=1, keepdims=True)
    sq_mat = sq_mat / np.where(sq_norms == 0, 1, sq_norms)
    sq_sims = sq_mat @ mat.T

    short_records = []
    prog2 = make_progress(len(short_cases), label="检索(短关键词)")
    for i, sc in enumerate(short_cases):
        q = sc["question"]
        rels = sc["relevant_ids"]
        order = np.argsort(-sq_sims[i])
        vec_top = [docs[j]["id"] for j in order]
        short_records.append({
            "id": sc["id"], "scenario": sc["scenario"], "question": q,
            "relevant_ids": rels, "vec_top5": vec_top[:5],
        })
        prog2.update(note=sc["id"])

    def _recall(recs, key, k):
        scored = [r for r in recs if r["relevant_ids"]]
        hits = sum(any(rel in r[key][:k] for rel in r["relevant_ids"]) for r in scored)
        return round(hits / len(scored), 4) if scored else 0.0

    def _retrieval_metrics(recs, key, k):
        """统一算 Recall/Precision/MRR/nDCG@K（仅对相关标注样本）。"""
        scored = [r for r in recs if r["relevant_ids"]]
        if not scored:
            return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}
        preds = [r[key] for r in scored]
        rels = [r["relevant_ids"] for r in scored]
        return {
            "recall": round(metrics.recall_at_k(preds, rels, k), 4),
            "precision": round(metrics.precision_at_k(preds, rels, k), 4),
            "mrr": round(metrics.mrr(preds, rels, k), 4),
            "ndcg": round(metrics.ndcg_at_k(preds, rels, k), 4),
        }

    # ------------------------------------------------------------------
    # 指标
    # ------------------------------------------------------------------
    from . import metrics

    report = {
        "n": len(retrieval_cases), "pool_size": len(docs),
        "vector_backend": vector_backend,
        "top_k": {},
        "short_keyword": {"n": len(short_records), "top_k": {}},
    }
    for k in top_ks:
        m = _retrieval_metrics(records, "vec_top5", k)
        ms = _retrieval_metrics(short_records, "vec_top5", k)
        report["top_k"][str(k)] = {
            "recall@k": _recall(records, "vec_top5", k),
            "mrr": m.get("mrr"),
            "ndcg": m.get("ndcg"),
        }
        report["short_keyword"]["top_k"][str(k)] = {
            "recall@k": _recall(short_records, "vec_top5", k),
            "mrr": ms.get("mrr"),
            "ndcg": ms.get("ndcg"),
        }
    # 分场景（FAQ vs 菜谱，完整问句）
    by_sc = {}
    for sc in ("history_faq", "recipe_search", "recipe_detail", "recipe_compare"):
        rows = [r for r in records if r["scenario"] == sc]
        if not rows:
            continue
        k = top_ks[0]
        by_sc[sc] = {
            "n": len(rows),
            "recall@k": _recall(rows, "vec_top5", k),
        }
    report["per_scenario"] = by_sc
    # 短关键词分场景
    short_by_sc = {}
    for sc in ("history_faq", "recipe_search", "recipe_detail", "recipe_compare"):
        rows = [r for r in short_records if r["scenario"] == sc]
        if not rows:
            continue
        k = top_ks[0]
        short_by_sc[sc] = {
            "n": len(rows),
            "recall@k": _recall(rows, "vec_top5", k),
        }
    report["short_keyword"]["per_scenario"] = short_by_sc

    out_path = "eval/results/retrieval_eval_report.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open("eval/results/retrieval_eval_details.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    prog.finish()
    prog2.finish()

    print("\n============== 检索评测报告 ==============")
    print(f"样本: {report['n']}  |  候选池: {report['pool_size']} 条知识")
    for k, v in report["top_k"].items():
        print(f"Recall@{k}: {v['recall@k']:.4f}   MRR@{k}: {v['mrr']}   nDCG@{k}: {v['ndcg']}")
    print("")
    print("-- 分场景（Recall@%d）--" % top_ks[0])
    for sc, v in report["per_scenario"].items():
        print(f"  {sc:16s} n={v['n']:3d}  recall={v['recall@k']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-size", type=int, default=600)
    ap.add_argument("--top-k", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--eval-set", default="eval/cases/eval_set.jsonl")
    args = ap.parse_args()

    import asyncio
    asyncio.run(main_async(args.pool_size, args.top_k, args.eval_set))


if __name__ == "__main__":
    main()
