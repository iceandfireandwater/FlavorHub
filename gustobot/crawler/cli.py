"""
Crawler CLI entrypoint.

Docs reference:
  python -m gustobot.crawler.cli wikipedia --query "川菜" --import-kb --limit 10
"""


import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import httpx

from .wikipedia import fetch_wikipedia_pages, wikipedia_page_to_recipe


def _post_recipes_batch(
    *,
    api_base_url: str,
    recipes: List[Dict[str, Any]],
    timeout: float,
) -> Dict[str, Any]:
    url = api_base_url.rstrip("/") + "/api/v1/knowledge/recipes/batch"
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        response = client.post(url, json=recipes)
        response.raise_for_status()
        return response.json() if response.content else {}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gustobot.crawler.cli")
    subparsers = parser.add_subparsers(dest="source", required=True)

    wikipedia = subparsers.add_parser("wikipedia", help="Import Wikipedia page extracts")
    wikipedia.add_argument("--query", required=True, help="Search query")
    wikipedia.add_argument("--limit", type=int, default=10, help="Number of pages to fetch")
    wikipedia.add_argument("--lang", default="zh", help="Wikipedia language (default: zh)")
    wikipedia.add_argument("--import-kb", action="store_true", help="Insert into KB via API")
    wikipedia.add_argument(
        "--api-base-url",
        default="http://localhost:8000",
        help="Backend base URL (default: http://localhost:8000)",
    )
    wikipedia.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds")
    wikipedia.add_argument("--output", default=None, help="Write harvested pages as JSON to file")
    wikipedia.add_argument("--dry-run", action="store_true", help="Do not POST, only print/emit JSON")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.source == "wikipedia":
        try:
            pages = fetch_wikipedia_pages(
                args.query, limit=args.limit, lang=args.lang, timeout=args.timeout
            )
        except httpx.HTTPError as exc:
            print(f"Failed to fetch Wikipedia pages: {exc}", file=sys.stderr)
            return 1

        if args.output:
            with open(args.output, "w", encoding="utf-8") as fp:
                json.dump(pages, fp, ensure_ascii=False, indent=2)

        recipes = [wikipedia_page_to_recipe(page, query=args.query) for page in pages]

        if args.dry_run or not args.import_kb:
            print(json.dumps(recipes, ensure_ascii=False, indent=2))
            return 0

        try:
            result = _post_recipes_batch(
                api_base_url=args.api_base_url, recipes=recipes, timeout=args.timeout
            )
        except httpx.HTTPError as exc:
            print(f"Failed to import into KB API: {exc}", file=sys.stderr)
            return 1

        inserted = result.get("statistics", {}).get("success")
        print(f"Imported {inserted if inserted is not None else len(recipes)} items into KB.")
        return 0

    parser.error(f"Unknown source: {args.source}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
