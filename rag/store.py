"""
rag/store.py — Vector knowledge base with chunking
===================================================
Persistent ChromaDB store + Ollama embeddings, with real overlapping text
chunking and per-item TTL.

This is what lets the agent "learn / stay current": researched web pages and
documents are chunked, embedded, and retrieved semantically at query time —
RAG as the practical substitute for retraining the model's weights.

Storage: ~/.aicoder/rag/chroma/   (separate from the legacy knowledge store)
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

RAG_DIR = Path.home() / ".aicoder" / "rag"
CHROMA_DIR = RAG_DIR / "chroma"
COLLECTION = "aicoder_rag"

# Chunking defaults (characters). Small enough to keep retrieved context lean
# for a local model, with overlap so facts aren't split across a boundary.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
DEFAULT_TTL_HOURS = 168.0  # 1 week


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring to break on newlines."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Prefer a newline boundary in the last `overlap` chars of the window
            nl = text.rfind("\n", end - overlap, end)
            if nl > start:
                end = nl
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


class KnowledgeBase:
    """Persistent semantic knowledge base. Singleton via ``KnowledgeBase.get()``."""

    _instance: "KnowledgeBase | None" = None

    def __init__(self) -> None:
        self._client = None
        self._collection = None

    @classmethod
    def get(cls) -> "KnowledgeBase":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Lazy init ──────────────────────────────────────────────────────────────

    def _init(self) -> None:
        if self._client is not None:
            return
        try:
            import chromadb
            from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
        except ImportError as e:  # pragma: no cover
            raise ImportError("chromadb is not installed. Run: pip install chromadb") from e

        from core.config import get_config

        cfg = get_config()
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        # The current OllamaEmbeddingFunction strips any /api/embeddings suffix
        # and uses the modern /api/embed endpoint, so pass the base URL.
        embedding_fn = OllamaEmbeddingFunction(
            url=cfg.model_base_url.rstrip("/"),
            model_name=cfg.embedding_model,
        )
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ──────────────────────────────────────────────────────────────────

    def add(
        self,
        text: str,
        source: str = "",
        title: str = "",
        ttl_hours: float = DEFAULT_TTL_HOURS,
        project: str = "",
    ) -> int:
        """Chunk, embed, and upsert text. Returns the number of chunks stored."""
        self._init()
        chunks = chunk_text(text)
        if not chunks:
            return 0

        now = time.time()
        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            cid = hashlib.md5(f"{source}::{title}::{i}::{chunk[:80]}".encode()).hexdigest()
            ids.append(cid)
            docs.append(chunk)
            metas.append({
                "source": source,
                "title": title,
                "chunk": i,
                "fetched_at": now,
                "ttl_hours": ttl_hours,
                "project": project,
            })
        self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
        return len(chunks)

    # ── Read ───────────────────────────────────────────────────────────────────

    def search(self, query: str, n: int = 5, max_distance: float = 0.5) -> list[dict]:
        """
        Semantic search. Returns up to n live (non-expired) results as
        {content, metadata, distance}, filtered by relevance.

        ``max_distance`` is the cosine-distance cutoff (0 = identical, 2 =
        opposite). Results above it are dropped, so an unrelated query against a
        sparse store returns nothing instead of the nearest irrelevant chunk.
        """
        self._init()
        try:
            count = self._collection.count()
            if count == 0:
                return []
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n * 3, count),
            )
        except Exception:
            return []

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        now = time.time()
        out: list[dict] = []
        for doc, meta, dist in zip(docs, metas, dists):
            if dist is not None and dist > max_distance:
                continue
            ttl = float(meta.get("ttl_hours", DEFAULT_TTL_HOURS))
            if meta.get("fetched_at", 0) >= now - ttl * 3600:
                out.append({"content": doc, "metadata": meta, "distance": dist})
            if len(out) >= n:
                break
        return out

    def count(self) -> int:
        self._init()
        return self._collection.count()

    def info(self) -> dict:
        self._init()
        return {"total_chunks": self._collection.count(), "storage_path": str(CHROMA_DIR)}
