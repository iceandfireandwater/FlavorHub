"""
Wikipedia crawler using the public MediaWiki API.

Design goals:
- Small surface area (used by CLI + tests)
- No hard dependency on the backend server; ingestion happens via HTTP API client in CLI
- Network calls are isolated so mapping can be unit-tested offline
"""


from typing import Any, Dict, List, Optional

import httpx


def wikipedia_page_to_recipe(page: Dict[str, Any], *, query: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert a Wikipedia page summary into a recipe-like payload accepted by the KB API.

    We store the extract as a single "step" so it remains searchable/retrievable in the recipe KB.
    """

    title = (page.get("title") or page.get("name") or "").strip()
    extract = (page.get("extract") or page.get("content") or "").strip()
    fullurl = (page.get("fullurl") or page.get("url") or "").strip()

    category = "Wikipedia"
    if query:
        category = f"Wikipedia/{query}"

    tips_parts: List[str] = []
    if fullurl:
        tips_parts.append(f"来源：{fullurl}")

    payload: Dict[str, Any] = {
        "name": title or (query or "Wikipedia"),
        "category": category,
        "ingredients": None,
        "steps": [extract] if extract else None,
        "tips": "；".join(tips_parts) if tips_parts else None,
    }
    return payload


def _api_url(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


def search_wikipedia_titles(
    query: str,
    *,
    limit: int = 10,
    lang: str = "zh",
    timeout: float = 20.0,
) -> List[str]:
    """Return a list of Wikipedia page titles matching the query."""

    if not query or not query.strip():
        return []

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": max(1, min(int(limit), 50)),
        "format": "json",
    }

    response = httpx.get(_api_url(lang), params=params, timeout=timeout, trust_env=False)
    response.raise_for_status()
    payload = response.json()

    results = payload.get("query", {}).get("search", []) or []
    titles = [item.get("title") for item in results if isinstance(item, dict) and item.get("title")]
    return [str(t) for t in titles]


def fetch_wikipedia_pages(
    query: str,
    *,
    limit: int = 10,
    lang: str = "zh",
    timeout: float = 20.0,
) -> List[Dict[str, Any]]:
    """
    Fetch Wikipedia page intro extracts for a search query.

    Returns items with keys: title, extract, fullurl
    """

    titles = search_wikipedia_titles(query, limit=limit, lang=lang, timeout=timeout)
    if not titles:
        return []

    # Fetch extracts + canonical urls in one request.
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": 1,
        "exintro": 1,
        "inprop": "url",
        "titles": "|".join(titles),
        "format": "json",
    }

    response = httpx.get(_api_url(lang), params=params, timeout=timeout, trust_env=False)
    response.raise_for_status()
    payload = response.json()

    pages = payload.get("query", {}).get("pages", {}) or {}
    collected: List[Dict[str, Any]] = []
    for _page_id, page in pages.items():
        if not isinstance(page, dict):
            continue
        title = page.get("title")
        if not title:
            continue
        collected.append(
            {
                "title": str(title),
                "extract": (page.get("extract") or "").strip(),
                "fullurl": (page.get("fullurl") or "").strip(),
            }
        )

    # Keep order stable (search order).
    by_title = {item["title"]: item for item in collected if item.get("title")}
    ordered = [by_title[title] for title in titles if title in by_title]
    # Append any leftover pages (unlikely, but safe)
    leftover = [item for item in collected if item.get("title") not in set(titles)]
    return ordered + leftover
