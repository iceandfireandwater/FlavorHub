# -*- coding: utf-8 -*-
"""古籍白话翻译：读 原文/<书>.txt，逐条调用 LLM 译成现代汉语，写 译文/<书>.txt。

用法：
    python scripts/translate_classics.py --book 本心斋疏食谱            # 全书
    python scripts/translate_classics.py --book 本心斋疏食谱 --limit 3  # 只译前 N 个块
    python scripts/translate_classics.py --book 本心斋疏食谱 --workers 4
"""
import argparse
import io
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = r"E:\code\python\project\GustoBot"
KB = os.path.join(BASE, "data", "kb", "古籍")
CR, LF = chr(13), chr(10)

SYS_PROMPT = (
    "你是中国古代饮食古籍的翻译专家。把用户给你的文言文片段翻译成现代汉语白话。"
    "硬性要求："
    "1) 逐句翻译，不省略、不概括、不合并、不添加原文没有的解释或评价；"
    "2) 食材名、药名、书名、人名、地名等专有名词保留原样（必要时可在括号内用今天的说法点一下）；"
    "3) 涉及做法的必须把步骤、用量、火候译准，不得臆造；"
    "4) 遇到底本缺字（如方框、问号、●）保留原样，不要猜补；"
    "5) 只输出译文本身，不要任何前缀、后缀、标题、引号或“译文：”字样。"
)


TITLE_SYS_PROMPT = (
    "你是中国古代饮食古籍的翻译专家。用户会给你一个古籍的**条目标题**（菜名/食名），"
    "请把它译成今天读者一看就懂的白话名称。硬性要求："
    "1) 只输出一个名词性短语，2–8 个字，不要成句；"
    "2) 不要解释，不要写“的意思是”“指的是”，不要任何标点；"
    "3) 若该名称今天仍通用（如“雪藕”“绿粉”“烧猪肉”）就原样输出；"
    "4) 只输出结果本身，不要引号、前后缀。"
)


def load_env(path):
    env = {}
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


ENV = load_env(os.path.join(BASE, ".env"))
API_URL = ENV["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
API_KEY = ENV["LLM_API_KEY"]
MODEL = ENV.get("TRANSLATE_MODEL", "claude-sonnet-5")


def translate(text, model=None, kind="text", retries=4, temperature=0.2):
    payload = {
        "model": model or MODEL,
        "messages": [
            {"role": "system", "content": TITLE_SYS_PROMPT if kind == "title" else SYS_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                API_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + API_KEY,
                    "User-Agent": "OpenAI/Python 1.0.0",
                },
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read().decode("utf-8"))
            out = d["choices"][0]["message"]["content"].strip()
            if out:
                # 每个待译片段对应原文的一行，译文必须压成单行，
                # 否则 LLM 插入的换行/空行会把条目的块结构撑开、导致全文错位
                out = out.replace(chr(13) + chr(10), " ").replace(chr(10), " ").replace(chr(13), " ")
                out = re.sub(r"[ 　]{2,}", " ", out).strip()
                return out
            last = "empty content"
        except Exception as e:  # noqa: BLE001
            last = "%s: %s" % (type(e).__name__, str(e)[:200])
            time.sleep(2 * (i + 1))
    raise RuntimeError("translate failed after %d retries (%s)" % (retries, last))


TITLE_RE = re.compile(r"^卷(第)?[一二三四五六七八九十百\d]+$")
CATEGORY_RE = re.compile(r"^[一-鿿]{1,5}(门|單|单|品|类|類|鲊|鮓|造)$")


def is_keep_title(line):
    """这类行是全书结构标题（卷号/门类/单/品），保留原文不翻译。"""
    s = line.strip()
    if TITLE_RE.match(s) or CATEGORY_RE.match(s):
        return True
    return False


def is_struct_title(line):
    s = line.strip()
    if not s:
        return False
    if TITLE_RE.match(s):
        return True
    # 短且不带句读的行，视为结构标题（书名/卷名/门类名/条目名）
    if len(s) <= 8 and not re.search(r"[，。；：、！？…—\u3001\u3002]", s):
        return True
    return False


def split_blocks(lines):
    """按空行切块，返回 [(start_index, [lines...]), ...]"""
    blocks = []
    cur = []
    for i, l in enumerate(lines):
        if l.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(l)
    if cur:
        blocks.append(cur)
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    src = os.path.join(KB, "原文", args.book + ".txt")
    dst = args.out or os.path.join(KB, "译文", args.book + ".txt")
    text = io.open(src, encoding="utf-8").read().replace(CR + LF, LF)
    blocks = split_blocks(text.split(LF))
    if args.limit:
        blocks = blocks[: args.limit]

    # 收集需要翻译的片段
    jobs = []  # (block_idx, line_idx, text)
    for bi, blk in enumerate(blocks):
        if bi == 0:
            continue  # 第 0 块是书名 + 撰者行，整块原样保留
        if len(blk) == 1:
            if not is_keep_title(blk[0]):
                jobs.append((bi, 0, blk[0], "text"))
        else:
            # 多行块：第 0 行是条目名（也译，走 title 模式）；其余是正文
            if is_keep_title(blk[0]):
                for li in range(1, len(blk)):
                    jobs.append((bi, li, blk[li], "text"))
            else:
                jobs.append((bi, 0, blk[0], "title"))
                for li in range(1, len(blk)):
                    jobs.append((bi, li, blk[li], "text"))

    print("blocks=%d, lines_to_translate=%d, model=%s" % (len(blocks), len(jobs), args.model))

    # 断点缓存
    cache_path = os.path.join(KB, "译文", ".cache_%s_%s.json" % (args.book, args.model))
    cache = {}
    if os.path.exists(cache_path) and not args.no_cache:
        cache = json.loads(io.open(cache_path, encoding="utf-8").read())
        print("cache loaded:", len(cache), "entries")

    results = {}
    todo = []
    for bi, li, t, kind in jobs:
        ck = "%s|%d:%d:%s" % (kind, bi, li, t[:16])
        if ck in cache:
            results[(bi, li)] = cache[ck]
        else:
            todo.append((bi, li, t, kind, ck))

    print("to translate: %d (cached: %d)" % (len(todo), len(jobs) - len(todo)))
    lock = __import__("threading").Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(translate, t, args.model, kind): (bi, li, ck) for bi, li, t, kind, ck in todo}
        done = 0
        for fut in futs:
            bi, li, ck = futs[fut]
            try:
                out = fut.result()
            except Exception as e:  # noqa: BLE001
                out = "[翻译失败] " + str(e)[:120]
            results[(bi, li)] = out
            with lock:
                cache[ck] = out
                done += 1
                if done % 5 == 0:
                    io.open(cache_path, "w", encoding="utf-8").write(json.dumps(cache, ensure_ascii=False, indent=0))
                print("  %d/%d done" % (done, len(todo)), flush=True)
    io.open(cache_path, "w", encoding="utf-8").write(json.dumps(cache, ensure_ascii=False, indent=0))

    out_lines = list(blocks[0])  # 书名 + 撰者行，原样
    for bi, blk in enumerate(blocks):
        if bi == 0:
            continue
        out_lines.append("")
        for li, line in enumerate(blk):
            out_lines.append(results.get((bi, li), line))
    out_lines.append("")

    data = (LF.join(out_lines)).replace(LF, CR + LF).encode("utf-8")
    io.open(dst, "wb").write(data)
    print("written:", dst, len(data), "bytes")


if __name__ == "__main__":
    main()
