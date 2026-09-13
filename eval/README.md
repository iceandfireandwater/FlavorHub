# GustoBot 评测体系（eval/）

对**原始系统**（未做任何优化改动）构建的端到端评测体系：评测集 + 指标 + 分层评测器。

## 评测集

`eval/cases/eval_set.jsonl` — 171 条评测集，覆盖 7 类任务（含 50 条难样本：描述性查找、同义改写、领域边界、复杂统计、混淆对比）：

| scenario | 说明 | 期望路由 | 数量 |
|---|---|---|---|
| recipe_search | 菜谱搜索（做法/步骤/描述性/同义改写） | graphrag-query | 50 |
| recipe_detail | 菜谱详情（食材/耗时/口味/工艺/步骤） | graphrag-query | 25 |
| recipe_compare | 菜谱对比（辣度/耗时/混淆） | graphrag-query | 25 |
| stat_query | 统计查询（含复杂统计/最值） | text2sql-query | 20 |
| history_faq | 历史文化典故（FAQ） | kb-query | 16 |
| multi_turn | 多轮记忆 | 混合 | 10 |
| negative | 越界/模糊（含领域边界难样本） | general/additional-query | 25 |

每条样本标注结构（对标业界评测集）：
`intent(expected_route) / tool(expected_tool) / args / slots / relevant_ids / topic / ground_truth / gt_source`

**关键设计：ground truth 全部可追溯** —— 问题由知识库条目生成或按真实用户问法构造，答案回溯到
`recipe.json` 具体字段或知识库段落，评测结论可当面演示验证。

## 实测基线（原始系统，无优化）

| 指标 | 数值 |
|---|---|
| Intent Accuracy | 66.67%（171 条，含难样本） |
| Slot Accuracy | 91.23%（严格全等）/ 100.00%（语义） |
| 分场景短板 | recipe_compare 4.00%、negative 68.00%、recipe_detail 64.00% |
| NLU 优化实验（仅实验，代码已还原） | 66.67% → 92.98%（+26.31pp） |
| Recall@K | 100.00%（text-embedding-v3，584 条候选知识库） |

## 指标口径

| 指标 | 定义 |
|---|---|
| Intent Accuracy | 系统路由类型 == 标注 intent 占比 |
| Slot Accuracy | LLM 槽位抽取与标注槽位匹配率（strict/loose） |
| Effective Tool Coverage | 选中标注工具且执行成功占比 |
| Effective Args Accuracy | 传给工具的参数与标注 args 匹配率 |
| Recall@K | 检索结果 Top-K 含 relevant_ids 占比 |
| Pipeline Success Rate | 最终回答含标注关键信息点占比 |

## 使用

```bash
# 1. 重新生成评测集（history_faq 用真实 LLM 出题）
python -m eval.generate --use-llm

# 2. 路由评测（Intent / Slot Accuracy，仅需 LLM API，无需数据库）
python -m eval.scripts.run_route_eval
# 产出 eval/results/route_eval_report.json + *_details.jsonl

# 3. 检索评测（Recall@K：vector vs RRF 融合 + topic 置顶，需 Embedding API）
python -m eval.scripts.run_retrieval_eval
# 产出 eval/results/retrieval_eval_report.json

# 4. 端到端评测（Tool/Args/Pipeline Success，需完整环境：Neo4j+MySQL+Milvus+Redis）
python -m eval.scripts.run_e2e_eval
# 产出 eval/results/e2e_eval_report.json

# 5. 幻觉率评测（factuality 口径：回答断言 vs ground truth，需 LLM API）
python -m eval.scripts.run_hallucination_eval --concurrency 6
# 产出 eval/results/hallucination_eval_report.json + *_details.jsonl
```

## 幻觉率评测（eval/run_hallucination_eval.py）

**动机**：项目内置 self-RAG 式幻觉检测（check_hallucinations 节点 + GradeHallucinations 二值评分 + 专用 prompt），
但**未接入图（dead code）**；评测侧独立补上幻觉率评估，不改动原始系统代码。

**口径**（本机 e2e 明细无 sources 字段，故采用 factuality 而非 faithfulness）：
- 输入：e2e_eval_report_details.jsonl（系统实测回答）+ eval_set.jsonl（ground truth）
- 判定：独立 LLM-as-Judge（不走被测系统），**断言级**——把回答拆成原子事实断言，逐条判定
  supported 被 GT 支撑 / contradicted 与 GT 矛盾 / unsupported 无据断言
- 指标：
  claim_hallucination_rate = (contradicted + unsupported) / 总断言（严格口径）
  claim_contradiction_rate = contradicted / 总断言（保守口径，确定幻觉）
  groundedness = supported / 总断言
  sample_hallucination_rate = 含 >=1 条幻觉断言的样本占比
- 范围：仅对**有 ground truth 的场景**判定（recipe_search/detail/compare、history_faq、stat_query）；
  negative / multi_turn 无标准答案，跳过

**实测（96 条可判定样本，617 条断言）**：
| 指标 | 值 |
|---|---|
| claim_hallucination_rate（严格） | 46.52% |
| claim_contradiction_rate（保守） | 5.67% |
| groundedness | 53.48% |
| sample_hallucination_rate | 81.25% |

**分场景**（严格口径 / 样本级）：recipe_search 31.2%/73.3%、recipe_compare 51.7%/93.3%、
recipe_detail 57.2%/88%、stat_query 68%/90%、history_faq 72.6%/68.8%

**洞察**：保守口径仅 5.67% 明确矛盾，严格口径 46.5% 说明主要问题是**"无据扩写"**
（回答补充了 GT 未提及的细节），而非捏造事实——与 key_fact 覆盖率 50.55% 互补：
一个测"信息够不够"，一个测"信息真不真"。

## 混合检索 RRF（run_retrieval_eval.py 内置）

- `keyword_score()`：简化 BM25（TF + 文档长度归一）
- `rrf_fuse()`：Reciprocal Rank Fusion 融合 keyword + vector 双路排序
- topic/keyword 精确置顶：菜名 + kb 段落主题建索引，query 词命中 topic 时强加权
- 候选文档池 = 相关文档 + 随机抽样知识库子集 + 全部 kb 段落，保证 Recall@K 有区分度
- 向量通道：优先真实 Embedding API（text-embedding-v3），不可用时自动降级本地 TF-IDF

## 评测驱动迭代（方法论）

基线 66.67% → 失败样本分析 → NLU 模板/启发式优化实验 92.98%（+26.31pp）。
按"保留原始系统"要求，优化代码已还原；`eval/results/route_eval_optimized.json` 保留实验数据，
失败样本清单在 `route_eval_report_details.jsonl`（pred≠expect 行），可随时复现优化。

---

# 统一评测入口（2026-09 新增）

## 一键运行

```bash
# 推荐：在 backend 容器内跑（容器里有全部依赖，且 eval/ 已挂载）
MSYS_NO_PATHCONV=1 docker exec -w /app -e PYTHONPATH=/app gustobot-backend-1 python -m eval.scripts.run_all

# 常用参数
python -m eval.scripts.run_all --quick                              # 每维度只跑 3 条（冒烟，~3 分钟）
python -m eval.scripts.run_all --only memory,guardrail,streaming    # 只跑指定维度
python -m eval.scripts.run_all --skip e2e,answer_quality            # 跳过指定维度
python -m eval.scripts.run_all --baseline eval/results/eval_summary.json  # 与上次结果对比
```

产物：
- `eval/results/EVAL_SUMMARY.md` —— 人读的总报告（含基线对比表）
- `eval/results/eval_summary.json` —— 机读版

> **为什么要在容器里跑**：老的评测器直接调用被测代码（需要 pydantic / langchain-openai / pymilvus 等），
> 宿主机没有这些依赖；容器里 `./eval` 已挂载（`docker-compose.yml`），且 `eval/client.py` 默认
> `GUSTOBOT_BASE_URL=http://localhost:8000`（同容器直连）。

## 维度总览

| 维度 | 评测器 | 类型 | 测什么 |
|---|---|---|---|
| 意图路由 | `run_route_eval.py` | 原有 | 意图分类 + slot 抽取准确率 |
| 知识检索 | `run_retrieval_eval.py` | 原有 | Recall@K / MRR、RRF 融合 |
| 工具选择 | `run_tool_eval.py` | 原有 | 选对数据源的概率 |
| 幻觉控制 | `run_hallucination_eval.py` | 原有 | claim 级幻觉率 / groundedness |
| 答案质量 | `run_answer_quality.py` | 原有 | key fact 覆盖 + LLM judge 打分 |
| 端到端 | `run_e2e_eval.py` | 原有 | 全链路 intent/tool/key-fact |
| **跨会话记忆** | `run_memory_eval.py` | **新增** | 会话 A 自述 → 新会话 B 追问；含防污染回归（50 条） |
| **会话内多轮记忆** | `run_intra_session_memory_eval.py` | **新增** | 同一会话里前面说的话，滑出窗口后还能否记住（50 条） |
| **拒答正确性** | `run_guardrail_eval.py` | **新增** | 该答的必须答（反向回归）、该拒的必须拒 |
| **流式输出** | `run_streaming_eval.py` | **新增** | chunk 数 / TTFT / 整段重复检测 |
| **文件上传** | `run_upload_eval.py` | **新增** | 上传文件内容能否被回答引用 |
| **外部搜索** | `run_external_eval.py` | **新增** | 时效性问题不该回"暂无记载" |

## 新增评测集的用例格式

放在 `eval/cases/*.jsonl`，每行一条 JSON：

```jsonc
// memory_eval.jsonl —— 先在不同会话里说 turns，再用新会话问 probe
{"id":"mem_fact_01","kind":"facts","turns":["我叫阿强，住在成都","推荐一道菜"],
 "probe":"阿强来自哪里？","expect_any":["成都"],"forbid_any":["没有提到"]}

// guardrail_eval.jsonl —— must_answer=false 表示"应当拒答"
{"id":"gr_should_answer_01","query":"...","must_answer":true}

// streaming_eval.jsonl —— 流式门槛
{"id":"st_short","query":"推荐一道素菜","min_chunks":15,"min_len":60}
```

## 共用客户端

`eval/client.py` 封装了 `chat_stream` / `upload_file` / `list_sessions` / `delete_session` / `health`。
新增评测器**统一通过 HTTP 打真实后端**（而不是 import 内部函数），这样：
- 测的是"用户实际会经历的行为"；
- 评测代码不依赖被测代码的实现细节，重构后不会一起失效。


---

# 记忆评测集（各 50 条，用生成脚本产出以保证可复现）

```bash
python -m eval.gen_memory_cases    # 重新生成两份评测集
```

| 文件 | 条数 | 类型分布 |
|---|---|---|
| `eval/cases/memory_eval.jsonl` | 50 | facts 20 / constraint 15 / preference 8 / no_poison 7 |
| `eval/cases/intra_memory_eval.jsonl` | 50 | intra_constraint 15 / intra_facts 15 / intra_reference 12 / intra_preference 8 |

**两者的区别（重要）**：

- **跨会话记忆**（`memory_eval.jsonl`）：会话 A 里自述 → **另开一个会话 B** 追问。
  测的是 **L3 长期记忆**（`user_memories` 表，按 user_id 存）。
- **会话内多轮记忆**（`intra_memory_eval.jsonl`）：**全程同一个会话**，setup 轮 → 5 轮 filler
  （撑过 `GUSTOBOT_MEMORY_TURNS=5` 的窗口）→ probe 追问。
  测的是 **L2 会话摘要压缩**（`state.memory`）——即"前面说过的话滑出窗口后还在不在"。

会话内用例的字段与跨会话不同：

```jsonc
{"id":"intra_constraint_01","kind":"intra_constraint",
 "setup":["我不吃辣，另外对花生过敏"],          // 前几轮说的（会被挤出窗口）
 "fillers":["今天想喝点汤","随便问一句，谢谢",…],  // 撑过窗口
 "probe":"我前面说的饮食要求，你还记得吗？",
 "expect_any":["不吃辣"]}
```
