"""
Utility script to seed the Milvus vector store with initial knowledge base text.
"""

import asyncio
import os
from pathlib import Path
from typing import Iterable

from loguru import logger

from gustobot.infrastructure.knowledge import KnowledgeService


async def _ingest_chunk(service: KnowledgeService, chunk: str, source_name: str, index: int) -> int:
    if not chunk.strip():
        return 0

    metadata = {
        "source": source_name, # 文件路径名
        "ingest_mode": "bootstrap",
        "chunk_index": index,
    }
    result = await service.ingest_text(chunk, metadata=metadata)
    added = int(result.get("add_count", 0))
    logger.info("Milvus chunk {} ingested: {} segments", index, added)
    return added

# 把按\n\n分割出来的段落，按照每个chunk最大1600的长度，组合成多个chunk
def _chunk_paragraphs(paragraphs: Iterable[str], max_chars: int = 1600) -> list[str]:
    chunks: list[str] = []
    buffer: list[str] = []
    length = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if length + len(para) > max_chars and buffer:  # 按1600长度分割
            chunks.append("\n\n".join(buffer))
            buffer = []
            length = 0
        buffer.append(para)
        length += len(para) + 2

    if buffer:
        chunks.append("\n\n".join(buffer))

    return chunks


async def ingest_file(path: Path) -> int:
    if not path.exists():
        logger.warning("KB data file not found: {}", path)
        return 0

    # 清洗一道，去除空格
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        logger.warning("KB data file is empty: {}", path)
        return 0

    # 按\n\n分割
    paragraphs = [p for p in content.split("\n\n") if p.strip()]
    chunks = _chunk_paragraphs(paragraphs)
    if not chunks:
        logger.warning("No valid chunks generated from: {}", path)
        return 0

    service = KnowledgeService()
    total_added = 0
    try:
        for idx, chunk in enumerate(chunks, start=1):
            try:
                total_added += await _ingest_chunk(service, chunk, path.name, idx)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to ingest chunk {}: {}", idx, exc)
        logger.info("Milvus ingestion finished: total {} segments", total_added)
        return total_added
    finally:
        await service.close()


async def main() -> int:
    # data_path = Path(os.getenv("KB_DATA_FILE", "E:\\code\\python\\project\\GustoBot\\data\\kb\\data.txt"))
    data_path = Path(os.getenv("KB_DATA_FILE", "E:\\code\\python\\project\\GustoBot\\data\\kb\\历史菜谱源头.txt"))
    return await ingest_file(data_path)


if __name__ == "__main__":
    asyncio.run(main())
