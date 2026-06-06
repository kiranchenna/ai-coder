"""
rag/research.py — Web research → knowledge-base ingestion
==========================================================
The single "research a topic / fetch a URL into the knowledge base" pipeline,
shared by the agent's `research`/`fetch_url` tools and the `/knowledge learn`
REPL command (so the logic isn't duplicated across layers).
"""

from __future__ import annotations

from tools.web_tools import (
    fetch_url,
    format_search_results,
    is_fetch_error,
    search_web,
)

SEARCH_TTL_HOURS = 12.0
PAGE_TTL_HOURS = 48.0
PAGE_EXCERPT_CHARS = 2500
MIN_PAGE_CHARS = 200


def cache_url(url: str, project: str = "", max_chars: int = 20000) -> tuple[int, str]:
    """Fetch a URL and cache it. Returns (chunks_stored, page_text_or_error).

    Fetches up to ``max_chars`` so a docs page the agent asked to read isn't
    silently clipped to the small web-research excerpt size.
    """
    from rag.store import KnowledgeBase

    page = fetch_url(url, max_chars=max_chars)
    if is_fetch_error(page):
        return 0, page
    n = KnowledgeBase.get().add(page, source=url, title=url,
                                ttl_hours=PAGE_TTL_HOURS, project=project)
    return n, page


def research_topic(topic: str, project: str = "", fetch_pages: int = 2) -> dict:
    """
    Web-search a topic, cache the result summary plus the top pages, and return
    {count, sources, text}. `count` is chunks stored, `sources` is the human-
    readable source list, `text` is the combined research for the model to read.
    """
    from rag.store import KnowledgeBase

    kb = KnowledgeBase.get()
    results = search_web(topic)
    if not results:
        return {"count": 0, "sources": [], "text": ""}

    summary = format_search_results(results)
    count = kb.add(summary, source="web-search", title=topic,
                   ttl_hours=SEARCH_TTL_HOURS, project=project)
    sources = ["web-search results"]
    extra = ""

    for r in results[:fetch_pages]:
        url = r.get("href") or r.get("url", "")
        if not url:
            continue
        page = fetch_url(url)
        if page and not is_fetch_error(page) and len(page) > MIN_PAGE_CHARS:
            count += kb.add(page, source=url, title=r.get("title", topic),
                            ttl_hours=PAGE_TTL_HOURS, project=project)
            sources.append(url)
            extra += f"\n\n---\n\n**Source: {url}**\n\n{page[:PAGE_EXCERPT_CHARS]}"

    return {"count": count, "sources": sources, "text": summary + extra}
