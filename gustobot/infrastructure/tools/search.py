from typing import Dict, List, Optional
import httpx
from gustobot.config import settings
from gustobot.infrastructure.core import get_logger

logger = get_logger(service="tool.search")


class SearchTool:
    """Wrapper around SerpAPI Google search results."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: Optional[float] = None,
        default_results: Optional[int] = None,
    ) -> None:
        self.allow_network = settings.ENABLE_EXTERNAL_SEARCH
        self.api_key = api_key or settings.SERPAPI_KEY
        # 默认走 DuckDuckGo：免费、不需要 key；只有显式配了 provider=serpapi 且给了 key 才用 SerpAPI
        self.provider = (getattr(settings, "SEARCH_PROVIDER", "") or "duckduckgo").lower()
        if self.provider == "serpapi" and not self.api_key:
            logger.warning("SEARCH_PROVIDER=serpapi 但未配置 SERPAPI_KEY，回退 DuckDuckGo")
            self.provider = "duckduckgo"
        if not self.allow_network:
            logger.info("External search disabled; SearchTool will operate in no-op mode.")

        self.endpoint = (endpoint or settings.SERPAPI_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.SERPAPI_TIMEOUT
        self.default_results = default_results or settings.SEARCH_RESULT_COUNT

    def search(self, query: str, *, num_results: Optional[int] = None) -> List[Dict]:
        """执行搜索并返回结构化结果"""
        if not query:
            raise ValueError("Search query must not be empty.")

        if not self.allow_network:
            logger.info("External search disabled via configuration; returning empty result set.")
            return []

        result_count = num_results or self.default_results

        if self.provider == "serpapi" and self.api_key:
            return self._search_serpapi(query, result_count)
        return self._search_duckduckgo(query, result_count)

    def _search_duckduckgo(self, query: str, count: int) -> List[Dict]:
        """DuckDuckGo 检索（ddgs 库，免费无需 key）。"""
        try:
            from ddgs import DDGS
        except ImportError:
            logger.warning("ddgs 未安装，外部搜索不可用")
            return []
        try:
            with DDGS() as client:
                rows = list(client.text(query, max_results=count))
        except Exception as exc:  # noqa: BLE001
            logger.error("DuckDuckGo 搜索失败: {}", exc)
            return []
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href") or r.get("url", ""),
                "snippet": r.get("body", ""),
            }
            for r in rows
        ][:count]

    def _search_serpapi(self, query: str, result_count: int) -> List[Dict]:
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "num": result_count,
            "hl": "zh-CN",
            "gl": "cn",
        }

        try:
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                response = client.get(self.endpoint, params=params)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Search request failed: {}", exc)
            return []

        try:
            payload = response.json()
        except ValueError as exc:
            logger.error("Failed to decode search response: {}", exc)
            return []

        return self._parse_results(payload, limit=result_count)

    def _parse_results(self, data: Dict, *, limit: int) -> List[Dict]:
        """Normalize SerpAPI payload."""
        organic_results = data.get("organic_results") or []
        results: List[Dict] = []

        for item in organic_results:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
            )
            if len(results) >= limit:
                break

        return results
