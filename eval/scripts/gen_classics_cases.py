# -*- coding: utf-8 -*-
"""从古籍（Milvus `category=古籍译文`）出题，追加进 eval_set.jsonl。

设计要点
- **LLM 只负责生成"问句"**；relevant_ids / ground_truth / expected_answer_keywords
  全部直接取自 Milvus 里真实存在的 chunk —— 这样判据天然可校验，不会有"题目问的东西
  库里根本没有"的假失败。
- expected_route 一律 kb-query：实测古籍菜名在 Neo4j 图谱里命中率很低
  （山家清供 0/6、易牙遗意 1/6、随园食单 2/6），因为图谱来自 recipe.json，古籍是独立数据；
  古籍内容的检索入口是 Milvus。
- 出完题会**逐条校验**：relevant_ids 必须真在库里、关键词必须真在 content 里，不过就丢弃。

用法：
    python -m eval.gen_classics_cases --per-book 5 --dry-run   # 只打印不写
    python -m eval.gen_classics_cases --per-book 5            # 真正追加
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

EVAL_SET = Path("eval/cases/eval_set.jsonl")
BOOKS = [
    "随园食单", "山家清供", "饮膳正要", "易牙遗意",
    "本心斋疏食谱", "云林堂饮食制度集", "饮食须知", "清异录",
]
def _milvus_uri() -> str:
    """容器内跑用服务名，宿主机跑用 localhost。"""
    import os as _os
    explicit = _os.getenv("MILVUS_URI")
    if explicit:
        return explicit
    # 容器内能解析 milvus 这个主机名
    if _os.path.exists("/.dockerenv"):
        return "http://milvus:19530"
    return "http://localhost:19530"
ID_RE = re.compile(r"^classics_(.+?)_(\d+)$")


def load_env(path: str = ".env") -> dict:
    env = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def fetch_classics() -> dict:
    """从 Milvus 拉全部古籍 chunk，按书名分组。"""
    from pymilvus import MilvusClient

    cli = MilvusClient(uri=_milvus_uri())
    cli.load_collection("recipes")
    rows = cli.query(
        "recipes", filter="category == '古籍译文'",
        output_fields=["recipe_id", "name", "content"], limit=16000,
    )
    by_book = defaultdict(list)
    for r in rows:
        m = ID_RE.match(r.get("recipe_id") or "")
        content = (r.get("content") or "").strip()
        if not m or len(content) < 40:
            continue          # 太短的条目信息量不足，不适合出题
        by_book[m.group(1)].append({
            "recipe_id": r["recipe_id"],
            "book": m.group(1),
            "name": (r.get("name") or "").strip(),
            "content": content,
        })
    return by_book


def pick_keyword(chunk: dict) -> str:
    """从 content 里挑一段"有信息量"的原文作为判据关键词。

    取法：去掉书名与条目前缀后，从正文第 10 个字起截 16 个字 —— 这段一定是正文，
    模型只要真的答了这条内容就必然包含它。
    """
    body = chunk["content"]
    # 去掉 "《书名》门类·条目名：" 这段前缀
    body = re.sub(r"^《[^》]+》[^：]*：", "", body).strip()
    if len(body) < 26:
        return body[:16]
    return body[10:26]


QUESTION_PROMPT = """你是菜谱/饮食评测集的出题人。下面是一段中国古籍的饮食内容，请生成**一个普通用户会真实问出口的中文问题**。

硬性要求：
- 这个问题必须**只能靠上面这段内容**回答（不能是常识题）
- 自然口语化，像真实用户随手问的，不要写成"请简述…"这种考试腔
- 一句话，不超过 30 个字
- **只输出问题本身**，不要引号、不要书名号包裹整句、不要任何解释或前缀

【出处】《{book}》
【条目】{name}
【内容】{content}
"""

HISTORY_FOCUS = """出题方向：让用户**明确询问这道菜/这个食材的做法、来历或典故**（可以带上书名，体现"我在查古籍"）。
例：「随园食单里的茄子是怎么做的？」「山家清供里提到的青精饭是什么？」
"""

SEARCH_FOCUS = """出题方向：像普通用户那样**直接问这道菜怎么做 / 这个食材有什么讲究**，**不要提书名**（考察检索能否命中古籍内容）。
例：「栗子鸡怎么做？」「海藻吃了有什么禁忌？」
"""


def call_llm(env: dict, prompt: str, retries: int = 2) -> str:
    """直接打 chat/completions（脚本在宿主机跑，不经后端）。"""
    import urllib.request

    base = (env.get("LLM_BASE_URL") or "").rstrip("/")
    key = env.get("LLM_API_KEY") or ""
    model = env.get("LLM_MODEL") or "deepseek-chat"
    body = json.dumps({
        "model": model, "temperature": 0.8,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    last = None
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(
                base + "/chat/completions", data=body,
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
            )
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read().decode("utf-8"))
            return (d["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"LLM 调用失败: {last}")


def clean_question(raw: str) -> str:
    q = (raw or "").strip()
    q = re.sub(r"^(问题[:：]\s*)", "", q)
    q = q.strip().strip('"').strip("“”").strip("'")
    q = q.split("\n")[0].strip()
    return q


_KNOW_PAT = re.compile(
    r"(是什么|什么典故|为什么|什么来历|怎么来的|什么名字|什么东西|干什么用|是茶吗|还有别的名字|"
    r"哪些|怎么回事|指的是|能治什么|什么毛病|有什么讲究|有什么禁忌|会怎么样|有什么坏处|"
    r"什么人不能|有多少颗|都有哪|能吃吗|有毒|叫什么|都是啥)"
)
_STEP_PAT = re.compile(
    r"(怎么做|怎样做|如何做|怎么弄|怎么保存|怎么做才|做法|怎么烧|怎么炖|怎么蒸|怎么吃|"
    r"怎么操作|能用什么|用什么|留着干嘛|要放哪些|放哪些料)"
)


def route_of(question: str) -> str:
    """按问句语义决定期望路由（不是按数据在哪个库）。"""
    q = question or ""
    if _KNOW_PAT.search(q):
        return "kb-query"          # 问知识/典故/功效 → 文本检索
    if _STEP_PAT.search(q):
        return "graphrag-query"    # 问做法 → 图谱查结构化步骤
    return "kb-query"              # 兜底


def build_case(chunk: dict, question: str, idx: int, kind: str) -> dict:
    prefix = "hgc" if kind == "history_faq" else "gcs"
    return {
        "id": f"{prefix}_{idx:03d}",
        "scenario": kind,
        "turns": [question],
        # 路由按**语义**定，不是按数据在哪：
        #   「XX怎么做」→ graphrag-query（要结构化菜谱步骤）
        #   「XX是什么/什么典故/为什么/有什么禁忌」→ kb-query（要文本知识）
        # 所以不能一律写 kb-query —— 之前那样写导致 80 条古籍题的 intent 只有 0.44。
        "expected_route": route_of(question),
        # 古籍译文灌在 Milvus 的 recipes 集合里（Postgres 只有 data.txt 的历史文化段落，
        # 那些题的 relevant_ids 是 kb_para_*）。这里必须写 milvus，不能照抄旧的 postgres。
        "expected_tool": "milvus",
        "expected_args": {"query": question},
        "expected_slots": {},
        "relevant_ids": [chunk["recipe_id"]],
        "topic": chunk["book"],
        "expected_answer_keywords": [pick_keyword(chunk)],
        "ground_truth": chunk["content"],
        "gt_source": chunk["recipe_id"],
        "negative": False,
        "note": f"古籍：{chunk['book']}·{chunk['name']}",
    }


def validate(case: dict, chunk: dict) -> tuple:
    """校验题目自洽：关键词必须在 content 里、id 必须匹配、问句不能为空。"""
    kw = (case.get("expected_answer_keywords") or [""])[0]
    q = (case.get("turns") or [""])[0]
    if not q or len(q) < 4:
        return False, "问句过短/为空"
    if not kw or kw not in chunk["content"]:
        return False, "关键词不在 content 里"
    if case["relevant_ids"] != [chunk["recipe_id"]]:
        return False, "relevant_ids 不匹配"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-book", type=int, default=5, help="每本书出几题（每类）")
    ap.add_argument("--out", default=str(EVAL_SET))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    env = load_env()
    by_book = fetch_classics()
    missing = [b for b in BOOKS if b not in by_book]
    if missing:
        print("!! 库里缺这些书:", missing)

    rnd = random.Random(args.seed)
    new_cases, dropped = [], []

    for kind, focus in [("history_faq", HISTORY_FOCUS), ("recipe_search", SEARCH_FOCUS)]:
        idx = 0
        for book in BOOKS:
            items = by_book.get(book) or []
            if not items:
                continue
            for chunk in rnd.sample(items, min(args.per_book, len(items))):
                prompt = QUESTION_PROMPT.format(
                    book=book, name=chunk["name"], content=chunk["content"][:600]
                ) + focus
                try:
                    q = clean_question(call_llm(env, prompt))
                except Exception as exc:  # noqa: BLE001
                    dropped.append((book, "LLM失败: %s" % str(exc)[:50]))
                    continue
                case = build_case(chunk, q, idx, kind)
                ok, why = validate(case, chunk)
                if ok:
                    new_cases.append(case)
                    idx += 1
                    print("  [%s] %-10s %s" % (kind, book, q))
                else:
                    dropped.append((book, why))

    print("\n生成 %d 条，丢弃 %d 条" % (len(new_cases), len(dropped)))
    for b, w in dropped[:8]:
        print("   丢弃 %-10s %s" % (b, w))

    if args.dry_run:
        print("\n(dry-run，未写盘)")
        return

    with open(args.out, "a", encoding="utf-8") as f:
        for c in new_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("\n已追加到 %s" % args.out)


if __name__ == "__main__":
    main()
