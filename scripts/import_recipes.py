#!/usr/bin/env python3
"""
Batch import recipes into the KB via the HTTP API.

Example:
  python scripts/import_recipes.py --file data/recipe.json --batch-size 100
"""


import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gustobot.infrastructure.knowledge.recipe_import import recipe_json_entry_to_recipe


def _iter_payloads_from_json(
    raw: Any,
    *,
    limit: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Yield KB recipe payloads from either:
    - dict keyed by recipe name (bundled data/recipe.json format)
    - list of recipe dicts (already-normalized)
    """

    emitted = 0

    # 使用生成器 (yield) 的好处是即使 JSON 里有 10 万道菜，它也不会在内存中创建所有格式化后的数据，而是处理一个吐出一个，节省内存
    # 如果raw是字典，则遍历其键值对
    if isinstance(raw, dict):
        for name, entry in raw.items():
            if limit is not None and emitted >= limit:  # 限制导入的数量
                return
            if not isinstance(entry, dict):
                continue
            emitted += 1
            yield recipe_json_entry_to_recipe(str(name), entry)
        return

    # 如果raw是列表，则遍历其中的每个元素
    if isinstance(raw, list):
        for item in raw:
            if limit is not None and emitted >= limit:
                return
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("title") or item.get("菜名")
            if not name:
                continue
            emitted += 1
            # If already matches API fields, keep as-is, otherwise attempt a light mapping.
            if "主食材" in item or "辅料" in item or "做法" in item:
                yield recipe_json_entry_to_recipe(str(name), item)
            else:
                payload = {
                    "name": str(name),
                    "category": item.get("category") or item.get("类型"),
                    "time": item.get("time") or item.get("耗时"),
                    "ingredients": item.get("ingredients"),
                    "steps": item.get("steps"),
                    "tips": item.get("tips"),
                }
                yield payload
        return

    raise TypeError(f"Unsupported JSON root type: {type(raw)!r}")

# 把前面吐出的数据按照指定的 --batch-size（默认 100 条）打包成一个小列表
def _chunked(iterable: Iterable[Dict[str, Any]], size: int) -> Iterator[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:  # 如果批量处理的大小达到指定值
            yield batch  # 直接吐出
            batch = []
    if batch:  # 如果还有剩余的数据
        yield batch


def _post_batch(
    client: httpx.Client,
    *,
    api_base_url: str,
    recipes: List[Dict[str, Any]],
) -> Tuple[int, Dict[str, Any]]:
    url = api_base_url.rstrip("/") + "/api/v1/knowledge/recipes/batch"

    # ----------------------------POST----------------------------------
    response = client.post(url, json=recipes)
    # ----------------------------POST-----------------------------------

    response.raise_for_status()

    # payload={'status': 'success', 'message': 'Inserted 3 recipes', 'statistics': {'success': 3, 'error': 0, 'total': 3}}
    payload = response.json() if response.content else {}
    # inserted=3
    inserted = int(payload.get("statistics", {}).get("success", len(recipes)))

    print(f"[POST {url}] recipes={recipes}")
    print(f"[POST {url}] payload={payload}")
    print(f"[POST {url}] inserted={inserted}")

    # 返回插入的数量 和 POST的状态
    return inserted, payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Batch import recipes into KB")
    parser.add_argument("--file", required=True, help="Path to recipe JSON file")
    parser.add_argument("--batch-size", type=int, default=100, help="Recipes per request")
    parser.add_argument(
        "--api-base-url",
        default="http://localhost:8000",
        help="Backend base URL (default: http://localhost:8000)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only import first N recipes")
    parser.add_argument("--dry-run", action="store_true", help="Convert only, do not POST")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds")
    args = parser.parse_args(argv)

    json_path = Path(args.file)

    # Namespace(file='data\\recipe.json', batch_size=100, api_base_url='http://localhost:8000', limit=3, dry_run=False, timeout=60.0)
    print(args)  

    if not json_path.exists():
        print(f"File not found: {json_path}", file=sys.stderr)
        return 2

    with json_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)  # 将 JSON 文本解析为 Python 的字典 (dict) 或列表 (list)


    payload_iter = _iter_payloads_from_json(raw, limit=args.limit)

    # {'name': '香肠炒菜干', 'category': '热菜', 'time': '十分钟', 'ingredients': ['香肠 2根', '菜干 200g', '豆豉 2匙', '蒜 少许', '葱 1颗', '酱油 2匙', '蚝油 1匙', '食用油 适量'], 'steps': ['准备的食材。', '香肉肠切片。', '爆香蒜末、豆豉。', '倒入香肉肠，中火翻炒。', '炒两分钟后倒入菜干翻炒。', '加入酱油、蚝油调味，再加入葱段。', '炒均匀即可出锅。', '成品。'], 'tips': '口味：酱香；工艺：炒'}
    # for idx, item in enumerate(payload_iter):
    #     print(item)
    #     print('\n')

    if args.dry_run:
        sample = []
        for idx, item in enumerate(payload_iter):
            if idx >= min(args.batch_size, 5):
                break
            sample.append(item)
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        return 0

    total = 0
    started = time.time()

    # 默认设定 60s 超时
    # 创建了一个 HTTP 客户端，设置超时时间为 args.timeout 秒，不使用环境变量中的代理
    with httpx.Client(timeout=args.timeout, trust_env=False) as client:
        for batch_idx, batch in enumerate(_chunked(payload_iter, args.batch_size), start=1):
            inserted, _payload = _post_batch(client, api_base_url=args.api_base_url, recipes=batch)

            # 计算已插入的总数和平均插入速度
            total += inserted
            elapsed = time.time() - started
            rate = total / elapsed if elapsed > 0 else 0.0
            print(
                f"[batch {batch_idx}] sent={len(batch)} inserted={inserted} total={total} rate={rate:.1f}/s"
            )

    print(f"Done. Imported {total} recipes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
