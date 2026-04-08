"""
tools/web_tools.py — Web search and URL fetching
=================================================
Uses DuckDuckGo (free, no API key) for search.
Uses httpx + BeautifulSoup for clean page text extraction.
"""

from __future__ import annotations

import re
import httpx
from bs4 import BeautifulSoup
from rich.console import Console

console = Console()


# ─── Search ───────────────────────────────────────────────────────────────────

def search_web(query: str, max_results: int | None = None) -> list[dict]:
    """
    Search the web using DuckDuckGo and return a list of results.

    Each result dict has: title, url, body (snippet)

    Args:
        query:       Search query string
        max_results: Number of results (defaults to config value)

    Returns:
        List of result dicts with 'title', 'url', 'body' keys
    """
    from core.config import get_config
    cfg = get_config()
    n = max_results or cfg.search_max_results

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ddgs import DDGS
            
        with DDGS() as ddgs:
            # backend='lite' is sometimes much more accurate for technical keywords
            try:
                results = list(ddgs.text(query, max_results=n))
            except Exception:
                results = list(ddgs.text(query, backend="lite", max_results=n))
        return results or []
    except Exception as e:
        console.print(f"[yellow]⚠ Search failed: {e}[/yellow]")
        return []


def format_search_results(results: list[dict]) -> str:
    """Format search results as a readable string for AI context."""
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url   = r.get("href") or r.get("url", "")
        body  = r.get("body", "")
        lines.append(f"{i}. **{title}**\n   URL: {url}\n   {body}\n")

    return "\n".join(lines)


# ─── Fetch URL ────────────────────────────────────────────────────────────────

def fetch_url(url: str, timeout: int | None = None) -> str:
    """
    Fetch a URL and return its content as clean plain text (markdown-ish).

    Strips nav, footer, script, style and other noise — leaves the main
    readable content. Ideal for feeding documentation pages to the AI.

    Args:
        url:     The URL to fetch
        timeout: Request timeout in seconds (defaults to config value)

    Returns:
        Cleaned page text (may be truncated for very large pages)
    """
    from core.config import get_config
    cfg = get_config()
    t = timeout or cfg.search_timeout

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/122.0 Safari/537.36"
        )
    }

    try:
        resp = httpx.get(url, headers=headers, timeout=t, follow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        return f"[Error fetching {url}: {e}]"

    content_type = resp.headers.get("content-type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        return f"[Non-text content ({content_type}) at {url} — skipped]"

    if "text/plain" in content_type:
        return resp.text[:8000]

    return _html_to_text(resp.text, max_chars=8000)


def _html_to_text(html: str, max_chars: int = 8000) -> str:
    """Parse HTML and extract meaningful text content."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove noisy tags
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "svg", "iframe"]):
        tag.decompose()

    # Get text with reasonable spacing
    text = soup.get_text(separator="\n", strip=True)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... content truncated ...]"

    return text


# ─── Documentation fetch ──────────────────────────────────────────────────────

def fetch_docs(library: str) -> str:
    """
    Try to find and fetch the official documentation for a library.
    First searches DuckDuckGo, then fetches the top documentation page.

    Returns:
        Cleaned documentation text or an error message
    """
    query = f"{library} official documentation site"
    results = search_web(query, max_results=3)

    if not results:
        return f"Could not find docs for '{library}'."

    # Try each result until we get useful content
    for result in results:
        url = result.get("href") or result.get("url", "")
        if not url:
            continue
        text = fetch_url(url)
        if len(text) > 200 and not text.startswith("[Error"):
            return f"# Documentation: {library}\nSource: {url}\n\n{text}"

    return f"Could not fetch docs for '{library}'."
