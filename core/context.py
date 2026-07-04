"""
core/context.py — Workspace scanner / repo overview
====================================================
Builds a compact orientation summary of the workspace (languages, file count,
directory tree) that is injected into the agent's system prompt. The agent reads
individual files on demand via tools, so this stays lightweight — no file
contents are dumped here.
"""

from __future__ import annotations

from pathlib import Path

from tools.file_tools import file_tree


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
    Load .aicoderignore patterns from the workspace root (gitignore syntax via
    pathspec). Returns a PathSpec or None if absent.
    """
    ignore_file = root / ".aicoderignore"
    if not ignore_file.exists():
        return None
    try:
        import pathspec
        lines = [
            line.strip() for line in ignore_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return pathspec.PathSpec.from_lines("gitwildmatch", lines)
    except Exception:
        return None


class WorkspaceContext:
    """Lazily computed orientation snapshot of a workspace."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._file_count: int = 0
        self._languages: dict[str, int] = {}

    def overview(self, max_depth: int = 3) -> str:
        """
        Compact repo overview for agent orientation — languages, file count, and
        directory tree, WITHOUT file contents. Respects config ignore rules and
        the workspace's .aicoderignore.
        """
        from core.config import get_config

        cfg = get_config()
        ignore_dirs = set(cfg.ignore_dirs)
        ignore_exts = set(cfg.ignore_extensions)
        ai_ignore = _load_aicoderignore(self.root)

        if not self._languages:
            lang_counts: dict[str, int] = {}
            count = 0
            for path in self.root.rglob("*"):
                if not path.is_file():
                    continue
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
                count += 1
            self._languages = dict(sorted(lang_counts.items(), key=lambda x: -x[1]))
            self._file_count = count

        parts = [f"Project: {self.root.name}", f"Source files: {self._file_count}"]
        if self._languages:
            parts.append(
                "Languages: "
                + ", ".join(f"{lang} ({n})" for lang, n in list(self._languages.items())[:6])
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
