# -*- coding: utf-8 -*-
"""把古籍译文（data/kb/古籍/译文/*.txt）按条目灌入 Milvus 向量库 (collection=recipes)。

在 backend 容器内运行（主机没有 pymilvus，必须在容器里跑）：
    MSYS_NO_PATHCONV=1 docker exec -i gustobot-backend-1 python - --dry-run < scripts/ingest_classics_to_milvus.py
    MSYS_NO_PATHCONV=1 docker exec -i gustobot-backend-1 python - --apply   < scripts/ingest_classics_to_milvus.py
单独重灌某本（先删旧数据）：
    --book 清异录
"""
import argparse
import io
import os
import re
import sys

CR, LF = chr(13), chr(10)
KB_DIR = "/app/data/kb" if os.path.isdir("/app/data/kb") else r"E:\code\python\project\GustoBot\data\kb"
YI_DIR = os.path.join(KB_DIR, "古籍", "译文")
CATEGORY = "古籍译文"

TITLE_RE = re.compile(r"^卷(第)?[一二三四五六七八九十百\d]+$")
CATEGORY_RE = re.compile(r"^[\u4e00-\u9fff]{1,5}(门|單|单|品|类|類|鲊|鮓|造)$")
ID_RE = re.compile(r"^classics_")


def is_struct(line):
    s = line.strip()
    return bool(TITLE_RE.match(s) or CATEGORY_RE.match(s))


def to_blocks(text):
    out, cur = [], []
    for l in text.split(LF):
        if l.strip() == "":
            if cur:
                out.append(cur); cur = []
        else:
            cur.append(l)
    if cur:
        out.append(cur)
    return out


def parse_book(path):
    """译文文件 -> [(section, title, body)]"""
    t = io.open(path, encoding="utf-8").read().replace(CR + LF, LF)
    bs = to_blocks(t)
    items, section = [], ""
    for blk in bs[1:]:                      # 跳过第 0 块（书名 + 撰者行）
        if len(blk) == 1:
            if is_struct(blk[0]):
                section = blk[0]
            else:
                items.append((section, "", blk[0]))
        else:
            if is_struct(blk[0]):
                section = blk[0]
                items.append((section, "", " ".join(blk[1:])))
            else:
                items.append((section, blk[0], " ".join(blk[1:])))
    return items


def build_docs(only_book=""):
    docs, per_book = [], []
    seq = 0
    for fname in sorted(os.listdir(YI_DIR)):
        if not fname.endswith(".txt") or fname.lower().startswith("readme"):
            continue
        book = fname[:-4]
        if only_book and book != only_book:
            continue
        n = 0
        for section, title, body in parse_book(os.path.join(YI_DIR, fname)):
            body = body.strip()
            if not body:
                continue
            head = "《%s》" % book
            if section:
                head += section
            if title:
                head += "·%s" % title
            content = "%s：%s" % (head, body)
            if len(content) > 60000:
                content = content[:60000]
            meta = {
                "recipe_id": "classics_%s_%d" % (book, n),   # 每条唯一，避免 chunk_id 撞车
                "name": (title or section or book)[:120],
                "category": CATEGORY,
                "difficulty": "",
            }
            docs.append(("classics_%s_%d" % (book, n), content, meta))
            n += 1
            seq += 1
        per_book.append((book, n))
    return docs, per_book


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--book", default="")
    args = ap.parse_args()

    docs, per_book = build_docs(args.book)
    total_chars = sum(len(c) for _, c, _ in docs)
    print("=== 待灌入条目 ===")
    for b, n in per_book:
        print("  %-12s %4d 条" % (b, n))
    print("  合计 %d 条，content 总字符 %d" % (len(docs), total_chars))
    print("\n=== 前 3 条样例 ===")
    for _id, c, m in docs[:3]:
        print("  id=%s" % _id)
        print("  name=%s | recipe_id=%s" % (m["name"], m["recipe_id"]))
        print("  content=%s" % c[:160])
        print("  ---")

    if not args.apply:
        print("\n[dry-run] 未写库。加 --apply 执行。")
        return

    from gustobot.config import settings
    from gustobot.infrastructure.knowledge.embeddings import OpenAICompatibleEmbeddings
    from gustobot.infrastructure.knowledge.vector_store import VectorStore

    emb = OpenAICompatibleEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.EMBEDDING_API_KEY or settings.LLM_API_KEY,
        base_url=settings.EMBEDDING_BASE_URL,
        dimension=settings.EMBEDDING_DIMENSION,
    )
    vs = VectorStore(
        collection_name=settings.MILVUS_COLLECTION,
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
        dimension=settings.EMBEDDING_DIMENSION,
        index_type=settings.MILVUS_INDEX_TYPE,
        metric_type=settings.MILVUS_METRIC_TYPE,
    )
    print("\n=== 开始灌入（batch=%d）===" % args.batch)
    ok = 0
    for i in range(0, len(docs), args.batch):
        chunk = docs[i:i + args.batch]
        vecs = emb.embed_documents([c for _, c, _ in chunk])
        if vs.add_documents(ids=[d[0] for d in chunk], embeddings=vecs,
                            documents=[d[1] for d in chunk], metadatas=[d[2] for d in chunk]):
            ok += len(chunk)
        print("  %d/%d" % (ok, len(docs)), flush=True)
    print("=== 完成：写入 %d 条 ===" % ok)


if __name__ == "__main__":
    main()
