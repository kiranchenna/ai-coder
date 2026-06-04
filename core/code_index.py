"""
core/code_index.py — Lightweight symbol index
==============================================
A fast, dependency-free "where is X defined?" index. Scans source files and
extracts top-level/nested definitions (functions, classes, types, …) per
language with simple regexes — like a minimal ctags. Built on demand so it's
never stale.
"""

from __future__ import annotations

import re
from pathlib import Path

# Per-extension list of (compiled pattern, kind). Each pattern's group(1) is the
# symbol name. Patterns are matched against the start of each line.
_JS_PATTERNS = [
    (r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)", "function"),
    (r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)", "class"),
    (r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|\w+)\s*=>", "function"),
    (r"(?:export\s+)?(?:interface|type|enum)\s+(\w+)", "type"),
]

_RAW_PATTERNS: dict[str, list[tuple[str, str]]] = {
    ".py": [
        (r"(?:async\s+)?def\s+(\w+)", "function"),
        (r"class\s+(\w+)", "class"),
    ],
    ".go": [
        (r"func\s+(?:\([^)]*\)\s*)?(\w+)", "func"),
        (r"type\s+(\w+)", "type"),
    ],
    ".rs": [
        (r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", "fn"),
        (r"(?:pub\s+)?struct\s+(\w+)", "struct"),
        (r"(?:pub\s+)?enum\s+(\w+)", "enum"),
        (r"(?:pub\s+)?trait\s+(\w+)", "trait"),
    ],
    ".rb": [
        (r"def\s+(\w+)", "method"),
        (r"class\s+(\w+)", "class"),
        (r"module\s+(\w+)", "module"),
    ],
    ".java": [
        (r"(?:public|private|protected)?\s*(?:final\s+|abstract\s+)?class\s+(\w+)", "class"),
        (r"(?:public|private|protected)?\s*interface\s+(\w+)", "interface"),
    ],
}
for _ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue"):
    _RAW_PATTERNS[_ext] = _JS_PATTERNS

# Compile once.
_PATTERNS = {
    ext: [(re.compile(r"^\s*" + pat), kind) for pat, kind in pats]
    for ext, pats in _RAW_PATTERNS.items()
}


def build_symbol_index(
    workspace: Path,
    ignore_dirs: set[str] | None = None,
    max_files: int = 3000,
) -> dict[str, list[dict]]:
    """
    Return {symbol_name: [{file, line, kind}, ...]} for definitions across the
    workspace's source files. Best-effort, regex-based.
    """
    ignore_dirs = ignore_dirs or set()
    index: dict[str, list[dict]] = {}
    scanned = 0

    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignore_dirs for part in path.parts):
            continue
        patterns = _PATTERNS.get(path.suffix.lower())
        if not patterns:
            continue
        scanned += 1
        if scanned > max_files:
            break
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        rel = str(path.relative_to(workspace))
        for i, line in enumerate(lines, start=1):
            for pattern, kind in patterns:
                m = pattern.match(line)
                if m:
                    index.setdefault(m.group(1), []).append(
                        {"file": rel, "line": i, "kind": kind}
                    )
                    break

    return index
