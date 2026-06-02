"""
memory/project.py — Persistent, structured per-project memory
=============================================================
Durable facts the agent records about a project — decisions, conventions, key
facts, TODOs — kept separately from raw conversation history and from the RAG
knowledge base. Auto-loaded into the system prompt at the start of each session
so the agent "remembers where we left off" days later.

Storage: ~/.aicoder/memory/<project_id>/project_memory.json  (keyed by path)
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from core.config import MEMORY_DIR


def _project_id(root: Path) -> str:
    """Stable identifier for a project path (name + short hash)."""
    digest = hashlib.md5(str(root.resolve()).encode()).hexdigest()[:8]
    return f"{root.resolve().name}_{digest}"


class ProjectMemory:
    """Append-only store of durable project facts, grouped by category."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self._dir = MEMORY_DIR / _project_id(self.workspace)
        self.path = self._dir / "project_memory.json"

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save(self, items: list[dict]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── API ────────────────────────────────────────────────────────────────────

    def all(self) -> list[dict]:
        return self._load()

    def add(self, text: str, category: str = "note") -> dict:
        """Add a durable fact. Idempotent on identical text (case-insensitive)."""
        text = (text or "").strip()
        category = (category or "note").strip().lower()
        items = self._load()
        for it in items:
            if it.get("text", "").strip().lower() == text.lower():
                return it  # already remembered
        entry = {
            "id": hashlib.md5(f"{text}::{len(items)}".encode()).hexdigest()[:8],
            "text": text,
            "category": category,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        items.append(entry)
        self._save(items)
        return entry

    def search(self, query: str) -> list[dict]:
        q = (query or "").lower().strip()
        if not q:
            return self._load()
        return [
            it for it in self._load()
            if q in it.get("text", "").lower() or q in it.get("category", "").lower()
        ]

    def remove(self, entry_id: str) -> bool:
        items = self._load()
        kept = [it for it in items if it.get("id") != entry_id]
        if len(kept) != len(items):
            self._save(kept)
            return True
        return False

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def render(self, limit: int = 60) -> str:
        """Render memory as markdown grouped by category, for the system prompt."""
        items = self._load()
        if not items:
            return ""
        groups: "OrderedDict[str, list[dict]]" = OrderedDict()
        for it in items[-limit:]:
            groups.setdefault(it.get("category", "note"), []).append(it)
        lines: list[str] = []
        for category, entries in groups.items():
            lines.append(f"## {category.title()}")
            for it in entries:
                lines.append(f"- {it['text']}")
        return "\n".join(lines)
