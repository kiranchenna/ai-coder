"""
tools/file_tools.py — File system operations for aicoder
=========================================================
All paths are resolved relative to the active workspace root.
Includes backup, diff preview, and reviewed-write support.
"""

import re
import difflib
from pathlib import Path
from typing import Iterator

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def resolve(workspace: Path, rel_path: str) -> Path:
    """Resolve a relative path against the workspace root safely."""
    target = (workspace / rel_path).resolve()
    try:
        target.relative_to(workspace.resolve())
    except ValueError:
        raise PermissionError(f"Path escapes workspace: {rel_path}")
    return target


# ─── Read ─────────────────────────────────────────────────────────────────────

def read_file(workspace: Path, rel_path: str) -> str:
    """Read a file and return its text content."""
    path = resolve(workspace, rel_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {rel_path}")
    if not path.is_file():
        raise IsADirectoryError(f"Not a file: {rel_path}")
    return path.read_text(encoding="utf-8", errors="replace")


# ─── Write ────────────────────────────────────────────────────────────────────

def write_file(workspace: Path, rel_path: str, content: str) -> Path:
    """Write content to a file, creating parent directories as needed."""
    path = resolve(workspace, rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ─── Backup ───────────────────────────────────────────────────────────────────

def backup_file(workspace: Path, rel_path: str) -> Path | None:
    """
    Create a .bak copy of an existing file before it gets overwritten.

    Returns:
        The backup path if a backup was made, None if the file didn't exist.
    """
    path = resolve(workspace, rel_path)
    if not path.exists():
        return None
    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_bytes(path.read_bytes())
    return backup


# ─── Diff ─────────────────────────────────────────────────────────────────────

def generate_diff(old_content: str, new_content: str, filename: str = "file") -> str:
    """Return a unified diff string between old and new content."""
    old_lines = (old_content or "").splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff_iter = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm="",
    )
    return "\n".join(diff_iter)


def show_diff(diff_text: str, filename: str) -> None:
    """Render a coloured unified diff in the terminal using Rich Syntax."""
    if not diff_text.strip():
        console.print(f"  [dim](no changes in {filename})[/dim]")
        return
    syntax = Syntax(
        diff_text, "diff", theme="monokai",
        line_numbers=False, word_wrap=True,
    )
    console.print(
        Panel(
            syntax,
            title=f"[bold yellow]~ {filename}[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        )
    )


# ─── List ─────────────────────────────────────────────────────────────────────

def list_files(
    workspace: Path,
    rel_path: str = ".",
    pattern: str = "*",
) -> list[Path]:
    """Return files matching a glob pattern under rel_path."""
    base = resolve(workspace, rel_path)
    if not base.exists():
        return []
    if base.is_file():
        return [base]
    return sorted(p for p in base.rglob(pattern) if p.is_file())


# ─── File tree ────────────────────────────────────────────────────────────────

def file_tree(
    root: Path,
    ignore_dirs: list[str] | None = None,
    ignore_extensions: list[str] | None = None,
    max_depth: int = 4,
    _depth: int = 0,
    _prefix: str = "",
) -> str:
    """Generate a compact ASCII directory tree string."""
    if ignore_dirs is None:
        ignore_dirs = []
    if ignore_extensions is None:
        ignore_extensions = []

    if _depth > max_depth:
        return _prefix + "  ...\n"

    lines = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return _prefix + "  [permission denied]\n"

    visible = [
        e for e in entries
        if not (e.is_dir() and e.name in ignore_dirs)
        and not (e.is_file() and e.suffix in ignore_extensions)
        and not e.name.startswith(".")
    ]

    for i, entry in enumerate(visible):
        is_last  = i == len(visible) - 1
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "

        if entry.is_dir():
            lines.append(f"{_prefix}{connector}{entry.name}/")
            lines.append(
                file_tree(
                    entry,
                    ignore_dirs=ignore_dirs,
                    ignore_extensions=ignore_extensions,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    _prefix=_prefix + extension,
                )
            )
        else:
            lines.append(f"{_prefix}{connector}{entry.name}")

    return "\n".join(lines)


# ─── Search in files ──────────────────────────────────────────────────────────

def search_in_files(
    workspace: Path,
    query: str,
    rel_path: str = ".",
    case_sensitive: bool = False,
    max_results: int = 30,
    ignore_dirs: list[str] | None = None,
    ignore_extensions: list[str] | None = None,
) -> list[dict]:
    """
    Simple grep-like search. Returns list of {file, line_number, line}.

    Skips ignored directories (e.g. .venv, node_modules, __pycache__) and binary
    extensions. When the ignore lists are omitted they default to the workspace
    config, so callers don't accidentally scan virtualenvs or dependencies.
    """
    base = resolve(workspace, rel_path)
    results: list[dict] = []
    flags = 0 if case_sensitive else re.IGNORECASE

    if ignore_dirs is None or ignore_extensions is None:
        from core.config import get_config
        cfg = get_config()
        if ignore_dirs is None:
            ignore_dirs = cfg.ignore_dirs
        if ignore_extensions is None:
            ignore_extensions = cfg.ignore_extensions
    ignore_dir_set = set(ignore_dirs)
    ignore_ext_set = set(ignore_extensions)

    try:
        pattern = re.compile(re.escape(query), flags)
    except re.error:
        return []

    paths: Iterator[Path] = base.rglob("*") if base.is_dir() else iter([base])

    for file_path in paths:
        if not file_path.is_file():
            continue
        if any(part in ignore_dir_set for part in file_path.parts):
            continue
        if file_path.suffix in ignore_ext_set:
            continue
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue

        for i, line in enumerate(lines, start=1):
            if pattern.search(line):
                results.append(
                    {
                        "file": str(file_path.relative_to(workspace)),
                        "line_number": i,
                        "line": line.strip(),
                    }
                )
                if len(results) >= max_results:
                    return results

    return results
