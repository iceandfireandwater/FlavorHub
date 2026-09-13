"""评测用的后端客户端：统一封装对话/上传/会话接口。

所有新增评测器都通过 HTTP 打真实后端（而不是直接调内部函数），
这样测的是"用户实际会经历的行为"，也避免评测代码依赖被测代码的实现细节。
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import requests

import os as _os

BASE_URL = _os.getenv("GUSTOBOT_BASE_URL", "http://localhost:8000")


def _iter_sse(resp):
    """把 SSE 流拆成事件 dict。"""
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        try:
            yield json.loads(line[5:].strip())
        except Exception:  # noqa: BLE001
            continue


def chat_stream(
    message: str,
    *,
    session_id: Optional[str] = None,
    file_path: Optional[str] = None,
    user_id: Optional[str] = None,
    timeout: int = 300,
    base_url: str = BASE_URL,
) -> Dict[str, Any]:
    """走流式接口发一条消息，返回观测到的完整指标。

    返回字段：text / route / session_id / chunks / ttft / elapsed / errors
    """
    payload: Dict[str, Any] = {"message": message, "stream": True}
    if session_id:
        payload["session_id"] = session_id
    if file_path:
        payload["file_path"] = file_path
    if user_id:
        payload["user_id"] = user_id

    t0 = time.time()
    ttft: Optional[float] = None
    text, route, sid = "", None, session_id
    chunks, errors = 0, []

    with requests.post(base_url + "/api/v1/chat/stream", json=payload, stream=True, timeout=timeout) as resp:
        for evt in _iter_sse(resp):
            if evt.get("session_id"):
                sid = evt["session_id"]
            etype = evt.get("type")
            if etype == "message" and evt.get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                chunks += 1
                text += evt["content"]
            elif etype == "replace":
                text = evt.get("content") or ""
            elif evt.get("route"):
                route = evt["route"]
            elif etype == "error":
                errors.append(str(evt.get("content") or ""))

    return {
        "text": text,
        "route": route,
        "session_id": sid,
        "chunks": chunks,
        "ttft": ttft if ttft is not None else -1.0,
        "elapsed": time.time() - t0,
        "errors": errors,
    }


def upload_file(file_name: str, lines: List[str], *, base_url: str = BASE_URL) -> Optional[str]:
    """上传一个文本文件，返回 file_path。"""
    content = "\n".join(lines).encode("utf-8")
    resp = requests.post(
        base_url + "/api/v1/upload/file",
        files={"file": (file_name, content, "text/plain")},
        timeout=60,
    )
    data = resp.json()
    return data.get("file_path") if data.get("success") else None


def list_sessions(user_id: str = "default_user", *, base_url: str = BASE_URL) -> List[Dict[str, Any]]:
    resp = requests.get(f"{base_url}/api/v1/sessions/", params={"user_id": user_id}, timeout=30)
    return resp.json() if resp.ok else []


def delete_session(session_id: str, *, base_url: str = BASE_URL) -> bool:
    return requests.delete(f"{base_url}/api/v1/sessions/{session_id}", timeout=30).status_code == 204


def health(*, base_url: str = BASE_URL) -> bool:
    try:
        return requests.get(base_url + "/health", timeout=10).status_code == 200
    except Exception:  # noqa: BLE001
        return False
