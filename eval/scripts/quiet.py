# -*- coding: utf-8 -*-
"""评测时静音日志，只留进度条。

背景：跑一次检索/端到端评测是 200+ 条请求，每条都会打一堆 loguru INFO、
sqlalchemy 引擎日志、httpx 请求日志，把进度条冲得看不见。

用法——**import 即生效**：
    from . import quiet  # noqa: F401

想临时放开看调试信息就设环境变量：
    EVAL_LOG_LEVEL=INFO python -m eval.scripts.run_route_eval
"""
from __future__ import annotations

import logging
import os
import warnings

# 这些 logger 在评测里纯属噪音，一律压到 WARNING 以上
_NOISY_LOGGERS = (
    "sqlalchemy", "sqlalchemy.engine", "sqlalchemy.engine.Engine",
    "sqlalchemy.pool", "sqlalchemy.orm", "sqlalchemy.dialects",
    "httpx", "httpcore", "urllib3", "requests",
    "neo4j", "pymilvus", "pymilvus.client",
    "langchain", "langchain_core", "langchain_openai", "langchain_neo4j",
    "openai", "LiteLLM", "litellm",
    "matplotlib", "PIL", "asyncio",
)


def silence(level: str | None = None) -> None:
    """把日志压到 `level`（默认 ERROR）。设 EVAL_LOG_LEVEL 可覆盖。"""
    lvl_name = (level or os.getenv("EVAL_LOG_LEVEL") or "ERROR").upper()
    lvl = getattr(logging, lvl_name, logging.ERROR)

    # ① 标准 logging（sqlalchemy / httpx 等走这里）
    logging.basicConfig(level=lvl, force=True)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(lvl, logging.WARNING))
        logging.getLogger(name).propagate = False

    # ② loguru（项目自身大量 logger.info）
    try:
        import sys

        from loguru import logger

        logger.remove()
        logger.add(sys.stderr, level=lvl_name)
    except Exception:  # noqa: BLE001
        pass

    # ③ 告警
    warnings.filterwarnings("ignore")


silence()
