"""跨会话的长期记忆：把结构化 memory 按 user_id 持久化。

与 session 级的 memory.py 配合：
- memory.py 负责会话内的窗口裁剪 + 结构化抽取（本轮记忆）；
- 本模块负责把这份记忆**按用户**存进数据库，让新会话也能继承。

存储上就是一个用户一行、memory 存 JSON 文本，结构与 state.memory 完全一致：
{"constraints": [...], "preferences": {...}, "dishes": [...], "answered": [...], "notes": [...]}
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from sqlalchemy import text

from gustobot.infrastructure.core.database import engine

logger = logging.getLogger(__name__)

TABLE_DDL = (
    "CREATE TABLE IF NOT EXISTS user_memories ("
    "  user_id VARCHAR(128) NOT NULL,"
    "  memory LONGTEXT NOT NULL,"
    "  created_at DATETIME NULL,"
    "  updated_at DATETIME NULL,"
    "  PRIMARY KEY (user_id)"
    ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
)


def ensure_table() -> None:
    """幂等地建表（服务启动或首次使用时调用）。"""
    try:
        with engine.begin() as conn:
            conn.execute(text(TABLE_DDL))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"创建 user_memories 表失败: {exc}")


def load_user_memory(user_id: str) -> Dict[str, Any]:
    """读取该用户的长期记忆；不存在或出错时返回空 dict。"""
    if not user_id:
        return {}
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT memory FROM user_memories WHERE user_id = :u"), {"u": user_id}
            ).fetchone()
        if row and row[0]:
            data = json.loads(row[0])
            return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"读取长期记忆失败: {exc}")
    return {}


def save_user_memory(user_id: str, memory: Dict[str, Any]) -> None:
    """把记忆 upsert 进库（按 user_id 覆盖）。"""
    if not user_id or not memory:
        return
    try:
        payload = json.dumps(memory, ensure_ascii=False)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO user_memories (user_id, memory, created_at, updated_at) "
                    "VALUES (:u, :m, NOW(), NOW()) "
                    "ON DUPLICATE KEY UPDATE memory = :m, updated_at = NOW()"
                ),
                {"u": user_id, "m": payload},
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"写入长期记忆失败: {exc}")
