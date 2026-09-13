# GustoBot 三层评测体系结果（2026-08-11 实测）

> 完整评测分三层：NLU 层 / 检索与工具层 / 最终回答层。
> 本报告如实标注每层"已实测 / 需环境 / 失真不可用"。

---

## 第 1 层：NLU 层（✅ 已实测，路由级）

评测对象：`analyze_and_route_query`（真实生产路由节点），171 条评测集。

| 指标 | 数值 | 口径 |
|---|---|---|
| Intent Accuracy | **66.67%**（114/171） | 路由 == 标注 |
| Slot Accuracy（strict） | **91.23%** | 槽位全等匹配 |
| Slot Accuracy（loose） | **100.00%** | 槽位语义包含匹配 |

分场景 Intent（基线，原始系统）：
```
stat_query 95.0% > history_faq 87.5% > multi_turn 80.0% > recipe_search 78.0%
> negative 68.0% ≈ recipe_detail 68.0% > recipe_compare 0~4%（最大短板）
```

优化实验（prompt + 启发式）：66.67% → **92.98%（+26.31pp）**，对比/详情/统计三类 → 100%。

## 第 2 层：检索与工具层（✅ 检索已实测；⚠️ Tool/Args 需 e2e 环境）

检索评测：584 条候选知识库（568 菜谱 + 16 kb 段落），86 条可评分样本，text-embedding-v3。

| 指标 | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| **RRF 混合** Recall | 91.86% | 98.84% | 100% | 100% |
| RRF Precision | 91.86% | 38.76% | 23.49% | 11.74% |
| RRF MRR | 0.9186 | 0.9516 | 0.9539 | 0.9539 |
| RRF nDCG | 0.9186 | 0.9583 | 0.9628 | 0.9628 |
| **纯向量** Recall | 98.84% | 100% | 100% | 100% |
| 纯向量 MRR | 0.9884 | 0.9942 | 0.9942 | 0.9942 |

**关键发现（trade-off）**：
- RRF 融合**保召回**（@5 满格），但**第一名精度略降**（MRR 0.954 vs vector 0.994）——置顶权重把关键词命中项顶到第一，挤掉语义最相关的
- Precision 低是"相关文档稀疏"（每 query 仅 1-2 个 relevant）的必然结果，非检索差
- FAQ 长段落场景 Recall@1 仅 62.5%（16 条样本）——段落语义相近，第一名易排错，@3 回到 93.8%

⚠️ **Tool Coverage / Args Accuracy**：`run_e2e_eval.py` 已实现（metrics.tool_coverage / args_accuracy），需完整环境（Neo4j+MySQL+Milvus）跑端到端才能出真实值。

## 第 3 层：最终回答层（⚠️ 需完整环境，当前数值失真不可用）

`run_e2e_eval.py` 已实现三个指标：
- **key_fact_coverage**（关键事实覆盖率）：答案含 expected_answer_keywords 占比
- **rejection_accuracy**（拒答正确率）：negative 样本回答含拒绝/追问词占比
- **citation_consistency**（引用一致性）：回答 [n] 引用数 ≤ sources 条数占比

**本次 e2e 实测（环境受限，数值失真）**：intent 32.87% / key_fact 0% / rejection 0%。
失真原因（诚实报告，这是评测工程的真实发现）：
1. Neo4j/Milvus 不可用 → 子图抛 `'NoneType' object has no attribute 'query'`，异常串无 connect 字样 → 原 env_error 检测漏标 → 被当"有效样本"
2. 子图静默 fallback（KB workflow unavailable → direct search）→ 部分样本不抛异常但答案为空
3. **结论：e2e 必须在完整环境（Docker 全套服务）下运行才有意义**；当前已修复 env_error 检测（补 NoneType/attribute 等关键词），待环境就绪后重跑

## 三层总结表

| 层 | 指标 | 状态 | 数据 |
|---|---|---|---|
| NLU | Intent / Slot | ✅ 实测 | 66.67% / 91.23%(strict) 100%(loose) |
| 检索 | Recall@K / MRR / nDCG / Precision@K | ✅ 实测 | @1/3/5 = 91.9%/98.9%/100%，MRR 0.954，nDCG@5 0.963 |
| 工具 | Tool / Args | 🟡 代码就绪 | 需完整环境 |
| 回答 | key_fact / rejection / citation | 🟡 代码就绪 | 需完整环境（当前失真） |

## 复现命令

```bash
# NLU 层（路由级，仅需 LLM API）
python -m eval.run_route_eval

# 检索层（需 Embedding API）
python -m eval.run_retrieval_eval --top-k 1 3 5 10

# 回答层 + 工具层（需 Neo4j+MySQL+Milvus+Redis 完整环境）
python -m eval.run_e2e_eval
```
