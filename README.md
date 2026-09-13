<div align="center">

<img src="docs/images/chef.png" width="220" alt="GustoBot — 会记住你口味的美食助手" />

# GustoBot · 中华美食智能助手

**一个会记住你口味的多轮对话式菜谱 Agent**

把「菜谱问答」做成一套**有记忆、可评测、可复现**的工程系统。

<br/>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.60-1C3C3C)
![Milvus](https://img.shields.io/badge/Milvus-2.x-00A1EA)
![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1?logo=neo4j&logoColor=white)
![Vue](https://img.shields.io/badge/Vue-3-4FC08D?logo=vuedotjs&logoColor=white)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

</div>

---

## ⭐ 关于本项目：它从哪里来

> **本项目 fork 自 [@skygazer42](https://github.com/skygazer42) 的开源项目 [GustoBot](https://github.com/skygazer42/GustoBot)。**
>
> 上游作者搭起了一套**相当扎实**的基础设施 —— 多源 RAG 检索（Neo4j 知识图谱 + Milvus 向量库 + PostgreSQL）、
> LangGraph 多智能体编排、FastAPI 服务端、完整 Docker 编排、以及一套覆盖菜谱全场景的知识库。
>
> **没有这个起点，就没有后面这一切。在此向上游作者致以诚挚的感谢。** 🙏
>
> 原项目的完整说明保留在 [`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md)，上游仓库仍在本 README
> 末尾保留引用 —— **如果你只想用原版，请直接去上游仓库。**

### 那我在这个基础上做了什么？

一句话概括：**把「能跑通」推进到「能被量化和复现」。**

上游已经解决了「**怎么答**」；我主要解决的是三个它还没有回答的问题：

| 问题 | 我的回答 |
|---|---|
| **它记得住我吗？** | 三层记忆架构（窗口 / 会话摘要 / 跨会话长期记忆） |
| **它答得好不好，谁说了算？** | 一套 **251 条、9 个维度**的评测体系，指标可复现 |
| **换个模型/改句提示词，是变好还是变坏？** | 同上的评测体系 + 基线对比 |

> 顺带说明：**下面每一条改动都附了「为什么」** —— 我认为对一个开源项目来说，
> 讲清楚**决策依据**比罗列功能更有价值。这里面有不少是我踩了坑之后才想明白的。

---

## 🎯 我做的改动

### 1️⃣ 三层记忆架构 —— 让它真正「记得住」

上游的 `state.messages` 是只增不减的，配上持久化后**每轮 prompt 都会膨胀**，最终撑爆上下文；而且**换个会话就彻底失忆**。我重构成三层：

```
┌──────────────────────────────────────────────────────────────┐
│  L1  工作记忆   state.messages        窗口内最近 N 轮原文    │
│  L2  会话记忆   state.memory          超窗历史压成结构化 JSON │
│  L3  长期记忆   user_memories 表      按 user_id 跨会话累积   │
└──────────────────────────────────────────────────────────────┘
```

**关键设计：LLM 只负责「抽增量」，合并规则由代码确定性执行。**

早期我让 LLM 把「旧摘要 + 新对话」**重写**成新摘要 —— 结果是**越压越糊**：

```
第 3 轮：不吃辣、对花生过敏、家里有老人
第 5 轮重写后：不吃辣、对花生过敏          ← 老人丢了
第 8 轮重写后：口味清淡                     ← 更多丢了
```

每轮都是一次有损压缩，**旧字段会被悄悄丢掉，而且不可复现**。现在改成：

```python
async def extract_delta(overflow) -> dict:     # LLM：只从这段对话里抽增量
def merge_memory(previous, delta) -> dict:     # 代码：确定性合并（纯函数，可单测）
```

| 字段 | 合并语义 | 为什么 |
|---|---|---|
| `constraints` | **只增去重** | 硬约束（忌口/过敏）丢了是事故 |
| `relaxed` | 并集 | 用户**主动解除**的限制（见下） |
| `preferences` | **键覆盖** | 用户改口味要生效 |
| `dishes` / `facts` | **并集** | 信息累积 |
| `answered` | **按问题去重** | 同一问题不重复记 |

**这样最坏情况只丢一轮的增量，绝不会把历史整个毁掉。**

#### 三个踩过坑才想明白的语义问题

**① 「撤回」不能做成删除。** 用户说「花生那个限制不用管了」——我先做成**真删**，结果 `constraints` 空了，跨会话再问时模型**什么都不知道**，只能反问用户。改成 **`relaxed` 字段（记成"已解除"）**，渲染成：

```
硬性约束（必须遵守，不得违背）：不吃辣
用户已主动解除的限制（不要再拿它当禁忌）：对花生过敏
```

**② 记忆最危险的失败模式不是「记不住」，而是「把错的记进去」。** 我们真踩过：

```json
"answered": [{"q": "我之前提到了阿强，他来自哪里？",
              "a": "之前的对话中并没有提到阿强的信息"}]   ← 错误结论进了库
```

这条**每轮都注入**，模型看到这个"自我印证的记录"，就**推翻了自己刚看到的 facts**，反复说"没提到"。所以抽取规则里现在**明令禁止**把「没找到 / 无法回答 / 不在范围内」写进结论字段。

**③ 「用户是谁」需要一个专门的字段。** 「我叫阿强，在成都」既不是约束、不是偏好、不是菜名、也不是提问 —— **早期的抽取器直接把它丢了**。加了 `facts` 字段之后才记得住。

> 抽取时机也有讲究：原本只在**下一轮请求开始时**跑，导致**「只说一句就结束」的会话永远抽不到**。
> 现在在**流末尾再抽一次**（顺带不占首字延迟）。

### 2️⃣ 完整评测体系 —— 让「好不好」可量化

上游没有成体系的评测。我建了一套 **251 条 × 9 个维度**的评测流水线，一条命令出全部指标。

#### 测评集：251 条，覆盖 7 类任务

| 场景 | 条数 | 说明 |
|---|---|---|
| `recipe_search` | 90 | 菜谱查询 |
| `history_faq` | 56 | 饮食典故 / 古籍专项 |
| `recipe_detail` | 25 | 菜品细节 |
| `recipe_compare` | 25 | 对比查询 |
| `negative` | 25 | **负例**（该拒答的） |
| `stat_query` | 20 | 结构化统计 |
| `multi_turn` | 10 | 多轮对话 |

每条都标了 `expected_route` / `expected_tool` / `expected_slots` / `relevant_ids` / `expected_answer_keywords` / `ground_truth`，
**判据可回溯到原始数据**。

**其中 80 条是我新加的古籍题** —— 从 8 本中国饮食古籍（《随园食单》《山家清供》《饮膳正要》《易牙遗意》
《本心斋疏食谱》《云林堂饮食制度集》《饮食须知》《清异录》，共 2454 条译文）中**由 LLM 生成 + 脚本校验**：
`relevant_ids` 必须是 Milvus 里真实存在的 id、关键词必须真在原文里，**不满足就丢弃**。

#### 9 个评测维度

| 维度 | 指标 | 命令 |
|---|---|---|
| 意图路由 | `intent_accuracy` / `slot_accuracy` | `run_route_eval` |
| 知识检索 | `recall@k` / `MRR@k` / `nDCG@k` | `run_retrieval_eval` |
| 工具选择 | `tool_accuracy` / `args_accuracy` | `run_tool_eval` |
| 幻觉控制 | `claim_hallucination_rate` / `groundedness` | `run_hallucination_eval` |
| 答案质量 | `key_fact_coverage` / LLM-Judge | `run_answer_quality` |
| 端到端 | `intent` / `key_fact` / `rejection` | `run_e2e_eval` |
| 拒答正确性 | `reject_rate` / `answer_rate` | `run_guardrail_eval` |
| **跨会话记忆** | `recall` / `pass_rate` / `poison_rate` | `run_memory_eval` |
| **会话内记忆** | 同上 + `window_overflow_rate` | `run_intra_session_memory_eval` |

**全部评测器都有进度条 + 剩余时间估计**，跑 200+ 条不再是一屏刷不完的日志。

#### 一个我认为值得强调的教训：**先怀疑评测，再怀疑模型**

这一路下来，**80% 的时间花在修评测口径上，而不是修系统**。几个真实例子：

| 现象 | 真相 |
|---|---|
| `update` 类 10 条全错 | 判据里 `probe` 含「忌口」，而 `forbid_any` 又禁「忌口」——**模型复述问题就必挂** |
| `multi` 类 7/10 错 | 我给 `expect_all` 加了同义词展开，**"必须说出所有同义词"不可能满足** |
| `tool_accuracy = 0.05` | 模型选了 `predefined_cypher`（**更稳的实践**），判据却只认 `cypher_query`（让 LLM 现场生成），**把更优解判成了错** |
| `args_accuracy = 0` | 标注是扁平业务参数，预测是节点的原始嵌套输出，**结构完全不同** |
| 检索 `Recall@k` 对古籍恒为 0 | **候选池根本没把 Milvus 的古籍放进去** |

所以现在养成了一个习惯：**任何一个指标出现显著异动，先查评测基建，再查被测系统。**
（这个习惯救过我不止一次 —— 有一次"幻觉率暴涨"其实是 **Neo4j 凭据没设**导致的假失败。）

### 3️⃣ 古籍 RAG —— 给知识库补上「文化」这一块

上游的知识库偏重**菜谱做法**，饮食文化部分较薄。我灌入了 8 本中国饮食古籍的译文：

| 书 | 条目数 | 书 | 条目数 |
|---|---|---|---|
| 《清异录》 | 747 | 《易牙遗意》 | 168 |
| 《饮膳正要》 | 612 | 《山家清供》 | 106 |
| 《随园食单》 | 377 | 《云林堂饮食制度集》 | 50 |
| 《饮食须知》 | 372 | 《本心斋疏食谱》 | 22 |

**共 2454 条 chunk → Milvus**（`category=古籍译文`，`id=classics_<书名>_<序号>`）。

排版遵循古籍原貌：**卷 / 门 / 单结构保留，条目名也译成白话**，出处写进 `content`。

**顺带修掉一个架构缺口**：古籍只灌进了 Milvus，而 `graphrag-query` 默认走 Neo4j —— 于是
问「《随园食单》里的栗子糕怎么做」会**路由判对、却答"没查到"**。现在在**图谱无命中时自动兜底转查 Milvus**：

```python
# summarize/node.py
_NO_INFO_MARKS = ("couldn't find any relevant information", ...)
if not raw_context or any(k in raw_context.lower() for k in _NO_INFO_MARKS):
    raw_context = await _milvus_fallback_context(question)   # 兜底
```

> **注意这里判的是"error 特征串"而不只是"是否为空"** —— Cypher 查不到时会返回
> `{"error": "I couldn't find any relevant information"}`，**它会被当成"数据统计"塞进上下文**，
> 于是 `raw_context` 非空但内容其实是个错误提示。我第一版兜底就栽在这个细节上。

### 4️⃣ 其它改进

| 改动 | 说明与理由 |
|---|---|
| **真实的流式输出** | 上游的"流式"是**假的** —— 先 `await` 完整回答、再按空格拆分逐个 `sleep` 发送。**中文没有空格，等于整段一次性发出**。改成消费 `graph.astream(stream_mode="messages")` 的真 token 流 + SSE 推送 |
| **前端重写** | 从单页 widget 改成 **Vue 3 + 会话列表**：支持多会话切换、历史加载、markdown 渲染（标题/列表/代码块/链接）、上传附件、调用链路标注 |
| **Checkpoint 持久化** | `MemorySaver`（进程内）→ **`AsyncPostgresSaver`**，**重启不丢上下文** |
| **外部搜索** | 接入 **DuckDuckGo**（`ddgs`，免费无需 key），让「最近流行什么菜」这类时效性问题能联网补充 |
| **修掉几个静默 bug** | 见下方「踩坑记录」 |

#### 几个值得一提的坑

**「时好时坏」几乎总是「某条规则命中与否取决于 LLM 生成的中间文本」。**
问「土豆烧茄子怎么做」有时答得完整、有时答"没查到"。日志对照发现：

```
成功那次：tool_selection → predefined_cypher        （查 Neo4j）✅
失败那次：命中关键词规则「食材」→ 强推 LightRAG → 'LightRAG' object has no attribute '_addon_params' ❌
```

规则判的是 **LLM 改写后的查询串**（不是用户原话）——改写里恰好带上「食材」二字就挂。
**而 LightRAG 一直是坏的，只是因为大部分问题不走它，没人发现。**

**「任务已取消」≠「进程已停止」。** 长任务用 `nohup &` 起，`kill` 只终止 job 本身、**不会杀掉它拉起的进程**；
残留的孤儿进程会继续打后端，下次重跑就变成双倍并发。**取消后要查进程表确认。**

**LangGraph 的 `Send` 对象无法 JSON 序列化。** 子图用 `Send` 做动态并行，而 `PostgresSaver` 的
`pending_sends` **走原生 psycopg `Jsonb`、绕过 serde**，每次收尾都报 `Object of type Send is not JSON serializable`
（**崩在回答推完之后**，表现为"回答正常、末尾多一条 `[出错]`"）。最终用 `set_json_dumps(..., default=str)` 兜住。

---

## 🚀 快速开始

### 环境要求

| 依赖 | 版本 | 说明 |
|---|---|---|
| **Docker** + Docker Compose | 20.10+ | 拉起 Milvus / Neo4j / PostgreSQL / MySQL / Redis |
| **Python** | 3.10 | 后端 |
| **Node.js** | 18+ | 前端 |

> **不需要**单独安装 Milvus / Neo4j 等中间件 —— 全部由 `docker-compose.yml` 编排。

### 第 1 步：获取代码

```bash
git clone https://github.com/iceandfireandwater/FlavorHub.git
cd GustoBot
```

> ⚠️ 仓库用了 **Git LFS** 管理大文件（`data/recipe.json`、`data/1.png`）。
> 如果 clone 后这两个文件很小（几百字节），说明没拉 LFS 内容，执行：
> ```bash
> git lfs install && git lfs pull
> ```

### 第 2 步：配置环境变量

```bash
cp .env.example .env
```

然后编辑 `.env`，**至少填这三项**：

```ini
# ① LLM（对话/意图识别/记忆抽取）
LLM_API_KEY=your_llm_api_key_here
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# ② Embedding（向量检索，必需）
EMBEDDING_API_KEY=your_embedding_api_key_here
EMBEDDING_BASE_URL=...
EMBEDDING_MODEL=text-embedding-v3

# ③ Rerank（二阶段重排，可留空则跳过）
RERANK_API_KEY=your_rerank_api_key_here
```

> 知识库侧有一套独立的 `KB_LLM_*` / `KB_EMBEDDING_*` 变量，同样要填。
> **`.env` 已被 `.gitignore` 排除，不会误提交。**

### 第 3 步：启动后端（含全部中间件）

```bash
docker compose up -d
```

首次启动会拉取镜像并初始化数据库，**大约需要 3~5 分钟**。确认各服务就绪：

```bash
docker compose ps
curl http://localhost:8000/health          # 期望返回 200
```

各服务端口：

| 服务 | 端口 | 服务 | 端口 |
|---|---|---|---|
| 后端 API | **8000** | Neo4j Browser | 17474 |
| Milvus | 19530 | PostgreSQL | 5433 |
| MinIO | 9000 | MySQL | 13306 |

### 第 4 步：初始化知识库（首次必须）

后端启动时会自动检测并灌库。若需要手动触发：

```bash
# 灌入菜谱数据集
docker exec -w /app -e PYTHONPATH=/app gustobot-backend-1 python -m scripts.init_kb_milvus

# 灌入古籍译文（8 本，2454 条）
docker exec -w /app -e PYTHONPATH=/app gustobot-backend-1 python -m scripts.ingest_classics_to_milvus --apply
```

> 灌库用 DashScope embedding，**注意 batch 上限是 10**，脚本已默认处理。

### 第 5 步：启动前端

```bash
cd web
npm install
npm run dev
```

打开 **http://localhost:5173** 即可对话。前端通过 Vite proxy 转发到后端 8000 端口。

> 想换端口：`VITE_PORT=3000 npm run dev`；后端地址：`.env` 里的 `VITE_API_BASE_URL`。

### 常见问题

<details>
<summary><b>Q: 启动后问什么都说"没查到"</b></summary>

优先查 **Neo4j 凭据**。如果 `NEO4J_USER` / `NEO4J_PASSWORD` 缺失，`Neo4jGraph` 初始化会失败，
上层工具静默返回 `None`，表现为**间歇性的 `env_error`** —— 极易被误判成"模型变差了"。

```bash
docker exec gustobot-backend-1 env | grep NEO4J
```
</details>

<details>
<summary><b>Q: 改了 <code>.env</code> 但没生效</b></summary>

`docker restart` **不会重读 `.env`**，需要：

```bash
docker compose up -d backend
```

⚠️ 另外注意：`docker-compose.yml` 里的环境变量**优先级高于 `.env`**。如果某项在 compose 里被硬编码，
改 `.env` 是无效的。
</details>

<details>
<summary><b>Q: 重建容器后报 <code>ModuleNotFoundError</code></b></summary>

`docker compose up -d` 重建容器会**丢掉只在旧容器可写层里 pip 装的包**。凡是运行时装的依赖，
都要写进 `requirements.txt` 再 `docker compose build backend`。
</details>

<details>
<summary><b>Q: 一轮对话要十几秒</b></summary>

这是**预期行为**：一轮对话要**串行调用 LLM 十几次**（路由 → planner → 工具选择 → Cypher 生成 → 摘要 → 记忆抽取 …），
每次 1~2 秒。实测**换更快的模型只能省首 token 那零点几秒，总时长基本不变** —— 瓶颈在调用次数，不在单次延迟。

优化方向：① 减少 LLM 调用次数；② 记忆抽取改成异步后台（不阻塞首字）；③ 简单问题走轻量路径。
</details>

---

## 🧪 跑评测

评测入口是 `eval/run_all.py`，也可以单维度跑。

> **所有评测器必须在 backend 容器内执行** —— 老评测器直接 import 被测代码，宿主机环境缺依赖。

### 一键全跑

```bash
docker exec -w /app -e PYTHONPATH=/app gustobot-backend-1 python -m eval.scripts.run_all
```

产物：`eval/data/EVAL_SUMMARY.md`（人读）+ `eval_summary.json`（机读）

### 按维度跑（推荐，顺序有讲究）

```bash
# ① 意图路由 + 槽位
docker exec -w /app -e PYTHONPATH=/app gustobot-backend-1 python -m eval.scripts.run_route_eval

# ② 检索（recall@k / MRR / nDCG）
docker exec -w /app -e PYTHONPATH=/app gustobot-backend-1 python -m eval.scripts.run_retrieval_eval --top-k 1 3 5

# ③ 工具选择
docker exec -w /app -e PYTHONPATH=/app gustobot-backend-1 python -m eval.scripts.run_tool_eval

# ④ 端到端（最慢，60~90 分钟；⑤⑥ 都读它的明细，必须在前面跑）
docker exec -w /app -e PYTHONPATH=/app gustobot-backend-1 python -m eval.scripts.run_e2e_eval

# ⑤ 答案质量（读 ④ 的明细）
docker exec -w /app -e PYTHONPATH=/app gustobot-backend-1 python -m eval.scripts.run_answer_quality

# ⑥ 幻觉控制（读 ④ 的明细）
docker exec -w /app -e PYTHONPATH=/app gustobot-backend-1 python -m eval.scripts.run_hallucination_eval

# ⑦ 拒答正确性
docker exec -w /app -e PYTHONPATH=/app gustobot-backend-1 python -m eval.scripts.run_guardrail_eval

# ⑧⑨ 记忆（跨会话 / 会话内，各 80 条）
docker exec -w /app -e PYTHONPATH=/app gustobot-backend-1 python -m eval.scripts.run_memory_eval
docker exec -w /app -e PYTHONPATH=/app gustobot-backend-1 python -m eval.scripts.run_intra_session_memory_eval
```

> ⚠️ **`run_all` 里 `answer_quality` 排在 `e2e` 之前**（顺序是反的），所以一键跑之后建议**再补跑一次 ⑤⑥**，
> 否则它们读的是上一轮的 e2e 明细。

### 进度条

所有评测器都会显示进度 + **剩余时间估计**：

```
路由 [#########################...] 210/251  83.7%  已用 8m00s  剩余 ~1m33s  通过 173  hgc_038
```

### `run_all` 参数

| 参数 | 作用 |
|---|---|
| `--quick` | 每个维度只跑 3 条（冒烟） |
| `--only route,retrieval` | 只跑指定维度 |
| `--skip hallucination` | 跳过指定维度 |
| `--baseline eval/data/eval_summary.json` | 与基线对比，输出变化表 |

---

## 📊 实测指标（251 条测评集）

> 以下为本机实测。**每个指标的分母不同**（过滤条件不一样），我特意标出来 ——
> 我一直认为**只写百分比不写分母，是在误导读者**。

### 核心指标

| 指标 | 数值 | 分母 | 说明 |
|---|---|---|---|
| **Intent Accuracy** | **81.67%** | 251 | 路由类型判对的比例 |
| **Slot Accuracy**（loose） | **99.60%** | 251 | 语义口径（strict 口径 81.27%，见下） |
| **Recall@1 / @3 / @5** | **92.77% / 96.99% / 98.19%** | 196 | 纯向量检索 |
| **MRR@5** | **95.08%** | 196 | 首个正确答案的平均排名倒數 |
| **nDCG@5** | **95.36%** | 196 | 整体排序质量 |
| **Tool Selection Accuracy** | **90.00%** | 80 | 查询**路径**选对的比例 |
| **Args Accuracy** | **80.63%** | 80 | 传给工具的参数匹配率 |
| **Key Fact Coverage** | **78.36%** | 171 | 回答覆盖关键信息点的比例 |
| **Rejection Accuracy** | **85.71%** | 14 | 该拒答/该回答的判断正确率 |
| **Hallucination Rate** | **51.50%** | 176 | ⚠️ 见下方口径说明 |

### 关于幻觉率的口径（重要）

`51.50%` 这个数字**需要谨慎解读**。它统计的是"**未被标准答案直接支持的声明占比**"，而其中包含大量：

- **常识补充**（"适合配粥"）
- **合理推断**（从做法推出"咸鲜口味"）
- **表述差异**（同一事实的不同说法）
- 甚至**模型自述"我没查到 X"**（这是真话）

**真正"与事实矛盾"的比例是 3.57%**（`claim_contradiction_rate`），**这才是通常意义上的幻觉率**。

| 细分 | 数值 | 含义 |
|---|---|---|
| `supported` | 48.50% | 有据可依 |
| `unsupported` | 47.93% | 标准答案未提及（含常识/推断） |
| **`contradicted`** | **3.57%** | **与事实明确矛盾 ← 真幻觉** |
| `groundedness` | 48.50% | 忠实度 |

> **这个口径差异是我在分析阶段发现的**：判定 prompt 把"标准答案里没写"简单等同于"幻觉"了。
> 我认为如实标注口径，比报一个漂亮的数字更重要。

### 分场景 Intent

| 场景 | n | Intent | 场景 | n | Intent |
|---|---|---|---|---|---|
| `stat_query` | 20 | **100%** | `history_faq` | 56 | 85.71% |
| `recipe_detail` | 25 | 92.00% | `recipe_search` | 90 | 80.00% |
| `multi_turn` | 10 | 90.00% | `negative` | 25 | 68.00% |
| | | | `recipe_compare` | 25 | **64.00%** |

**已知短板**：`recipe_compare`（对比类查询）最低 —— 需要同时抽取两个菜名，模型常只抽出一个；
`negative` 偏低是判定口径问题（混入了"边界问题"，它们本不该按拒答率评）。

### 记忆评测

| 维度 | Recall | Pass Rate | Poison Rate | 样本 |
|---|---|---|---|---|
| **跨会话记忆** | 98.75% | 98.75% | 0% | 80 |
| **会话内记忆** | 97.50% | 97.50% | 0% | 80 |

**会话内**含 17 轮的超长程用例（埋点在第 1 句、追问在第 17 轮），**深层压缩后仍能召回**；
**跨会话**验证的是"新开会话还能否记得用户身份与约束"。

---

## 📁 项目结构

```
GustoBot/
├── gustobot/                      # 后端主体
│   ├── application/agents/
│   │   ├── memory.py              # ⭐ 三层记忆核心（抽取 / 合并 / 渲染）
│   │   ├── user_memory.py         # ⭐ 跨会话长期记忆（user_memories 表）
│   │   ├── lg_builder.py          # LangGraph 图构建 + 路由
│   │   ├── lg_states.py           # 图状态定义
│   │   └── kg_sub_graph/          # 知识图谱子图（图谱 + LightRAG + SQL）
│   ├── infrastructure/            # Milvus / Neo4j / DB / 工具
│   └── interfaces/http/v1/        # FastAPI 路由（chat / sessions）
├── eval/                          # ⭐ 评测体系
│   ├── run_route_eval.py          #   意图路由 + 槽位
│   ├── run_retrieval_eval.py      #   检索（recall / MRR / nDCG）
│   ├── run_tool_eval.py           #   工具选择
│   ├── run_e2e_eval.py            #   端到端
│   ├── run_answer_quality.py      #   答案质量 + LLM-Judge
│   ├── run_hallucination_eval.py  #   幻觉控制
│   ├── run_guardrail_eval.py      #   拒答正确性
│   ├── run_memory_eval.py         #   ⭐ 跨会话记忆
│   ├── run_intra_session_memory_eval.py  # ⭐ 会话内记忆
│   ├── gen_classics_cases.py      #   ⭐ 古籍题生成器
│   ├── progress.py                #   统一进度条（含 ETA）
│   ├── quiet.py                   #   日志静音
│   └── data/eval_set.jsonl        #   251 条测评集
├── data/
│   ├── kb/古籍/原文|译文/          # 8 本古籍（原文 + 译文）
│   └── recipe.json                # 菜品数据集（LFS）
├── scripts/                       # 灌库 / 导入 / 初始化脚本
├── web/                           # Vue 3 前端
├── docs/                          # 文档（含 UPSTREAM_README.md）
└── docker-compose.yml             # 全栈编排
```

---

## 🙏 致谢

### 上游项目

**本项目基于 [@skygazer42](https://github.com/skygazer42) 的 [GustoBot](https://github.com/skygazer42/GustoBot) 二次开发。**

上游作者完成了多源 RAG 架构、LangGraph 编排、知识库构建、Docker 全栈编排等**大量的基础工作** —— 这是一份
很有价值的开源贡献。**没有它就没有这个项目。** 如果你觉得这个方向有意思，请**先去给上游仓库点个 ⭐**。

原项目说明留存于 [`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md)，上游联系方式（原作者）：
- 上游仓库：https://github.com/skygazer42/GustoBot
- 邮箱：207829897@qq.com

### 技术栈

[FastAPI](https://fastapi.tiangolo.com/) ·
[LangChain](https://python.langchain.com/) ·
[LangGraph](https://langchain-ai.github.io/langgraph/) ·
[Milvus](https://milvus.io/) ·
[Neo4j](https://neo4j.com/) ·
[Vue 3](https://vuejs.org/) ·
[Vite](https://vitejs.dev/)

---

## 📄 License

[Apache License 2.0](LICENSE) —— 与上游项目保持一致。

> 二次开发部分同样以 Apache-2.0 发布。欢迎 issue / PR。

<div align="center">

**GustoBot** · 让 AI 成为真正懂你的厨房助手

`forked from` [skygazer42/GustoBot](https://github.com/skygazer42/GustoBot) `with ❤️`

</div>
