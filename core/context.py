"""
core/context.py — Workspace scanner and context builder
========================================================
Reads the current workspace and builds a compact summary for the AI.
This makes the AI aware of the project structure, language, and key files
without needing to load every file into the conversation.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.file_tools import file_tree


# Files that are always worth reading for project context
_KEY_FILES = [
    "package.json", "pyproject.toml", "requirements.txt", "Gemfile",
    "go.mod", "Cargo.toml", "pom.xml", "build.gradle",
    "README.md", "readme.md", "README.rst",
    ".env.example", "docker-compose.yml", "Dockerfile",
    "tsconfig.json", ".eslintrc.json", "vite.config.ts", "vite.config.js",
    "next.config.js", "next.config.ts",
]

_LANGUAGE_MAP = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".tsx": "TypeScript/React", ".jsx": "JavaScript/React",
    ".go": "Go", ".rs": "Rust", ".java": "Java", ".cs": "C#",
    ".cpp": "C++", ".c": "C", ".rb": "Ruby", ".php": "PHP",
    ".swift": "Swift", ".kt": "Kotlin", ".vue": "Vue",
    ".html": "HTML", ".css": "CSS", ".scss": "SCSS",
    ".sh": "Shell", ".bat": "Batch", ".ps1": "PowerShell",
    ".sql": "SQL", ".md": "Markdown",
}


def _load_aicoderignore(root: Path):
    """
    Load .aicoderignore patterns from the workspace root.
    Uses gitignore-style matching via pathspec.

    Returns a pathspec.PathSpec instance or None if no file found.
    """
    ignore_file = root / ".aicoderignore"
    if not ignore_file.exists():
        return None
    try:
        import pathspec
        lines = [
            l.strip() for l in ignore_file.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        return pathspec.PathSpec.from_lines("gitwildmatch", lines)
    except Exception:
        return None


class WorkspaceContext:
    """Lazily built snapshot of the current workspace."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._summary: str = ""
        self._key_files: dict[str, str] = {}
        self._file_count: int = 0
        self._languages: dict[str, int] = {}

    def build(self) -> "WorkspaceContext":
        """Scan the workspace and populate context fields."""
        from core.config import get_config
        cfg = get_config()

        ignore_dirs = set(cfg.ignore_dirs)
        ignore_exts = set(cfg.ignore_extensions)
        ai_ignore   = _load_aicoderignore(self.root)

        # ── Count files & detect languages ────────────────────────────────────
        lang_counts: dict[str, int] = {}
        for path in self.root.rglob("*"):
            if path.is_file():
                if any(part in ignore_dirs for part in path.parts):
                    continue
                if path.suffix in ignore_exts:
                    continue
                if ai_ignore:
                    rel = str(path.relative_to(self.root)).replace("\\", "/")
                    if ai_ignore.match_file(rel):
                        continue
                lang = _LANGUAGE_MAP.get(path.suffix.lower())
                if lang:
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                self._file_count += 1

        self._languages = dict(
            sorted(lang_counts.items(), key=lambda x: -x[1])
        )

        # ── Read key files ─────────────────────────────────────────────────────
        for name in _KEY_FILES:
            candidate = self.root / name
            if candidate.exists() and candidate.is_file():
                try:
                    text = candidate.read_text(encoding="utf-8", errors="replace")
                    # Truncate very large files
                    if len(text) > 3000:
                        text = text[:3000] + "\n... [truncated]"
                    self._key_files[name] = text
                except Exception:
                    pass

        # ── Build tree ─────────────────────────────────────────────────────────
        tree = file_tree(
            self.root,
            ignore_dirs=cfg.ignore_dirs,
            ignore_extensions=cfg.ignore_extensions,
            max_depth=3,
        )

        # ── Assemble summary ──────────────────────────────────────────────────
        parts: list[str] = [
            f"# Workspace: {self.root.name}",
            f"Path: {self.root}",
            f"Total source files: {self._file_count}",
        ]

        if self._languages:
            lang_str = ", ".join(f"{l} ({n})" for l, n in list(self._languages.items())[:6])
            parts.append(f"Languages: {lang_str}")

        if tree:
            parts.append(f"\n## Directory structure:\n```\n{self.root.name}/\n{tree}\n```")

        if self._key_files:
            parts.append("\n## Key project files:")
            for name, content in self._key_files.items():
                parts.append(f"\n### {name}\n```\n{content}\n```")

        self._summary = "\n".join(parts)
        return self

    @property
    def summary(self) -> str:
        """Return the workspace summary string (builds on first call)."""
        if not self._summary:
            self.build()
        return self._summary

    def overview(self, max_depth: int = 3) -> str:
        """
        A compact repo overview for agent orientation — languages, file count,
        and directory tree, WITHOUT dumping file contents (the agent reads files
        on demand via tools). Keeps the system prompt lean.
        """
        from core.config import get_config

        cfg = get_config()

        if not self._languages:
            ignore_dirs = set(cfg.ignore_dirs)
            ignore_exts = set(cfg.ignore_extensions)
            lang_counts: dict[str, int] = {}
            count = 0
            for path in self.root.rglob("*"):
                if not path.is_file():
                    continue
                if any(part in ignore_dirs for part in path.parts):
                    continue
                if path.suffix in ignore_exts:
                    continue
                lang = _LANGUAGE_MAP.get(path.suffix.lower())
                if lang:
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                count += 1
            self._languages = dict(sorted(lang_counts.items(), key=lambda x: -x[1]))
            self._file_count = count

        parts = [f"Project: {self.root.name}", f"Source files: {self._file_count}"]
        if self._languages:
            parts.append(
                "Languages: "
                + ", ".join(f"{l} ({n})" for l, n in list(self._languages.items())[:6])
            )
        tree = file_tree(
            self.root,
            ignore_dirs=cfg.ignore_dirs,
            ignore_extensions=cfg.ignore_extensions,
            max_depth=max_depth,
        )
        if tree:
            parts.append("Structure:\n" + tree)
        return "\n".join(parts)

    @property
    def primary_language(self) -> str:
        """Return the most common language detected in the workspace."""
        if not self._languages and not self._summary:
            self.build()
        if self._languages:
            return next(iter(self._languages))
        return "Unknown"

    def read_file_for_context(self, rel_path: str, max_kb: int = 200) -> str:
        """Read a specific file and return its content for AI context."""
        path = self.root / rel_path
        if not path.exists():
            return f"[File not found: {rel_path}]"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            max_chars = max_kb * 1024
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... [truncated]"
            return text
        except Exception as e:
            return f"[Cannot read {rel_path}: {e}]"

    def collect_files_for_ai(self, max_files: int | None = None) -> str:
        """
        Collect all source files in ===FILE: path=== format for AI context.
        Respects .aicoderignore if present.
        """
        from core.config import get_config
        cfg = get_config()
        max_files = max_files or cfg.max_context_files

        ignore_dirs = set(cfg.ignore_dirs)
        ignore_exts = set(cfg.ignore_extensions)
        max_bytes   = cfg.max_file_size_kb * 1024
        ai_ignore   = _load_aicoderignore(self.root)

        blocks: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in ignore_dirs for part in path.parts):
                continue
            if path.suffix in ignore_exts:
                continue
            if path.stat().st_size > max_bytes:
                continue
            # Apply .aicoderignore
            if ai_ignore:
                rel_str = str(path.relative_to(self.root)).replace("\\", "/")
                if ai_ignore.match_file(rel_str):
                    continue
            if len(blocks) >= max_files:
                break

            rel = path.relative_to(self.root)
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                blocks.append(f"===FILE: {rel}===\n{content}\n===END===")
            except Exception:
                pass

        return "\n\n".join(blocks)
