"""
core/knowledge.py — Vector RAG knowledge base
===============================================
Persistent vector store using ChromaDB + Ollama embeddings.
Fetched web content is cached with TTL and retrieved via semantic search.

Storage location: ~/.aicoder/knowledge/chroma/
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

console = Console()

# ── Paths ─────────────────────────────────────────────────────────────────────
KNOWLEDGE_DIR = Path.home() / ".aicoder" / "knowledge"
CHROMA_DIR    = KNOWLEDGE_DIR / "chroma"





# ── Knowledge Base ─────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    Persistent vector knowledge base using ChromaDB.

    Usage:
        kb = KnowledgeBase()
        text = kb.fetch_and_store("fastapi authentication tutorial")
        results = kb.search("how to add JWT auth", n=3)
    """

    _instance: "KnowledgeBase | None" = None

    def __init__(self, force_new: bool = False):
        self._client     = None
        self._collection = None
        self._embedding_fn: OllamaEmbeddingFunction | None = None

    @classmethod
    def get(cls) -> "KnowledgeBase":
        """Return the shared singleton instance."""
        if cls._instance is None:
            cls._instance = KnowledgeBase()
        return cls._instance

    def _init(self) -> None:
        """Lazily initialise ChromaDB client and collection."""
        if self._client is not None:
            return

        try:
            import chromadb
            from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
        except ImportError:
            raise ImportError(
                "chromadb is not installed. Run: pip install chromadb"
            )

        from core.config import get_config
        cfg = get_config()

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        self._embedding_fn = OllamaEmbeddingFunction(
            url = f"{cfg.model_base_url.rstrip('/')}/api/embeddings",
            model_name = cfg.embedding_model,
        )

        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = self._client.get_or_create_collection(
            name="aicoder_knowledge",
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Core operations ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n: int = 5,
        ttl_hours: float = 24.0,
        project: str | None = None,
    ) -> list[dict]:
        """
        Search the knowledge base for semantically similar content.
        Filters out entries whose TTL has expired.

        Returns list of {content, metadata, distance} dicts.
        """
        self._init()
        cutoff = time.time() - (ttl_hours * 3600)

        # Over-fetch to account for TTL filtering
        try:
            where: dict = {}
            if project:
                where["project"] = project

            results = self._collection.query(
                query_texts=[query],
                n_results=min(n * 3, max(self._collection.count(), 1)),
                where=where if where else None,
            )
        except Exception:
            return []

        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        valid = []
        for doc, meta, dist in zip(docs, metas, distances):
            if meta.get("fetched_at", 0) >= cutoff:
                valid.append({
                    "content":  doc,
                    "metadata": meta,
                    "distance": dist,
                })
            if len(valid) >= n:
                break

        return valid

    def store(
        self,
        query:    str,
        content:  str,
        url:      str     = "",
        ttl_hours: float  = 24.0,
        source:   str     = "search",
        project:  str     = "",
        phase:    str     = "",
    ) -> str:
        """
        Store content in the vector DB. Returns the document ID.
        Upserts — same query+url combination is always overwritten.
        """
        self._init()

        doc_id = hashlib.md5(f"{query}::{url}".encode()).hexdigest()
        self._collection.upsert(
            ids=[doc_id],
            documents=[content],
            metadatas=[{
                "query":      query,
                "url":        url,
                "fetched_at": time.time(),
                "ttl_hours":  ttl_hours,
                "source":     source,
                "project":    project,
                "phase":      phase,
            }],
        )
        return doc_id

    def store_document(
        self,
        url:      str,
        content:  str,
        title:    str     = "",
        ttl_hours: float  = 48.0,
        project:  str     = "",
        phase:    str     = "",
    ) -> str:
        """Convenience: store a fetched URL document."""
        return self.store(
            query     = title or url,
            content   = content,
            url       = url,
            ttl_hours = ttl_hours,
            source    = "document",
            project   = project,
            phase     = phase,
        )

    # ── High-level fetch + cache ──────────────────────────────────────────────

    def fetch_and_store(
        self,
        query:          str,
        ttl_hours:      float = 6.0,
        max_results:    int   = 3,
        fetch_top_url:  bool  = True,
        project:        str   = "",
        phase:          str   = "",
    ) -> str:
        """
        Check cache first. If stale/missing, search the web and cache results.
        Returns combined research text.
        """
        # 1. Check cache
        cached = self.search(query, n=max_results, ttl_hours=ttl_hours, project=project)
        if cached:
            return "\n\n---\n\n".join(r["content"] for r in cached)

        # 2. Fetch fresh
        from tools.web_tools import search_web, format_search_results, fetch_url

        console.print(f"  [dim]🌐 Searching: {query}[/dim]")
        results = search_web(query)

        if not results:
            return f"[No search results for: {query}]"

        combined = format_search_results(results)

        # Store search results
        self.store(
            query     = query,
            content   = combined,
            ttl_hours = ttl_hours,
            source    = "search",
            project   = project,
            phase     = phase,
        )

        # 3. Fetch top URL for richer content
        if fetch_top_url and results:
            url = results[0].get("href") or results[0].get("url", "")
            if url:
                console.print(f"  [dim]📄 Fetching: {url[:80]}[/dim]")
                doc_content = fetch_url(url)
                if doc_content and len(doc_content) > 200:
                    self.store_document(
                        url       = url,
                        content   = doc_content[:6000],
                        title     = results[0].get("title", query),
                        ttl_hours = ttl_hours * 2,
                        project   = project,
                        phase     = phase,
                    )
                    combined += f"\n\n---\n\n**Source: {url}**\n\n{doc_content[:3000]}"

        return combined

    def count(self) -> int:
        """Return total number of items in the knowledge base."""
        self._init()
        return self._collection.count()

    def clear_project(self, project: str) -> None:
        """Delete all entries for a specific project."""
        self._init()
        try:
            self._collection.delete(where={"project": project})
        except Exception:
            pass

    def info(self) -> dict:
        """Return knowledge base stats."""
        self._init()
        return {
            "total_documents": self._collection.count(),
            "storage_path":    str(CHROMA_DIR),
        }
