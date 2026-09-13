"""
Data crawlers / importers for bootstrapping the knowledge base.

Currently supported sources:
- Wikipedia (via MediaWiki API)
"""

from .wikipedia import fetch_wikipedia_pages, wikipedia_page_to_recipe

__all__ = ["fetch_wikipedia_pages", "wikipedia_page_to_recipe"]

