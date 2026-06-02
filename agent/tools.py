"""
agent/tools.py — Tool definitions for the agentic loop
=======================================================
Each tool is a LangChain tool (schema inferred from type hints + docstring)
bound to a specific workspace via a factory closure. The agent loop binds
these to the model for native tool calling and executes the model's requests.

Phase 1 toolset: list_files, read_file, write_file, edit_file, run_shell.
All file paths are relative to the workspace root and sandboxed to it.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from rich.console import Console
from rich.prompt import Confirm

import tools.file_tools as ft
from tools.shell_tools import run_with_confirmation

console = Console()

MAX_READ_CHARS = 20_000
MAX_SHELL_OUTPUT = 6_000


def _apply_write(workspace: Path, path: str, new_content: str, existing: str | None = None) -> str:
    """Shared write path: diff preview, config-driven confirmation, backup, write."""
    from core.config import get_config

    cfg = get_config()
    try:
        target = ft.resolve(workspace, path)
    except PermissionError as e:
        return f"ERROR: {e}"

    is_new = not target.exists()
    if is_new:
        old = ""
    elif existing is not None:
        old = existing
    else:
        old = target.read_text(encoding="utf-8", errors="replace")

    if not is_new and old == new_content:
        return f"No changes — {path} already has this exact content."

    console.print(
        f"  [green bold]+ NEW[/green bold] {path}" if is_new
        else f"  [yellow bold]~ MODIFIED[/yellow bold] {path}"
    )

    diff = ft.generate_diff(old, new_content, path)
    mode = cfg.file_confirmation  # always | auto | never
    if mode != "never" and diff.strip():
        ft.show_diff(diff, path)

    if mode == "always":
        if not Confirm.ask(f"  Apply changes to {path}?", default=True):
            return f"User declined the change to {path}."

    if not is_new and cfg.file_backup and old:
        backup = target.with_suffix(target.suffix + ".bak")
        backup.write_text(old, encoding="utf-8")

    ft.write_file(workspace, path, new_content)
    verb = "Created" if is_new else "Updated"
    return f"{verb} {path} ({len(new_content)} chars)."


def build_tools(workspace: Path) -> list:
    """Return the list of LangChain tools bound to this workspace."""

    @tool
    def list_files(path: str = ".") -> str:
        """List files and directories under a path (relative to the project root) as a tree.
        Use this to explore the project structure before reading or editing files."""
        from core.config import get_config

        cfg = get_config()
        try:
            base = ft.resolve(workspace, path)
        except PermissionError as e:
            return f"ERROR: {e}"
        if not base.exists():
            return f"Path not found: {path}"
        if base.is_file():
            return path
        tree = ft.file_tree(
            base,
            ignore_dirs=cfg.ignore_dirs,
            ignore_extensions=cfg.ignore_extensions,
            max_depth=3,
        )
        return f"{path.rstrip('/')}/\n{tree}" if tree else f"(empty directory: {path})"

    @tool
    def read_file(path: str) -> str:
        """Read and return the full text of a file (relative to the project root).
        Always read a file with this tool before editing it."""
        try:
            content = ft.read_file(workspace, path)
        except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
            return f"ERROR: {e}"
        if len(content) > MAX_READ_CHARS:
            return (
                content[:MAX_READ_CHARS]
                + f"\n\n[... truncated — file is {len(content)} chars total ...]"
            )
        return content

    @tool
    def write_file(path: str, content: str) -> str:
        """Create a new file, or completely overwrite an existing one, with the given content.
        Shows a diff and (per settings) asks the user to confirm. Prefer edit_file for small
        changes to existing files."""
        return _apply_write(workspace, path, content)

    @tool
    def edit_file(path: str, old_string: str, new_string: str) -> str:
        """Replace one exact occurrence of old_string with new_string in an existing file.
        old_string must match the file exactly (including whitespace/indentation) and must be
        unique. Include enough surrounding context to make it unique. Shows a diff to confirm."""
        try:
            original = ft.read_file(workspace, path)
        except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
            return f"ERROR: {e}"

        if old_string == new_string:
            return "ERROR: old_string and new_string are identical — nothing to change."
        occurrences = original.count(old_string)
        if occurrences == 0:
            return ("ERROR: old_string was not found in the file. Read the file again and "
                    "copy the exact text (including indentation) you want to replace.")
        if occurrences > 1:
            return (f"ERROR: old_string appears {occurrences} times — it must be unique. "
                    "Include more surrounding lines so it matches exactly one location.")

        updated = original.replace(old_string, new_string)
        return _apply_write(workspace, path, updated, existing=original)

    @tool
    def search_code(query: str, path: str = ".") -> str:
        """Search file CONTENTS across the project for a literal text/substring (like grep).
        Returns matching 'file:line: text' results. Use this to locate where something is
        defined or used when you don't know which file it's in. The query is matched
        literally, not as a regex. Optionally restrict to a subdirectory with `path`."""
        try:
            results = ft.search_in_files(workspace, query, rel_path=path, max_results=40)
        except PermissionError as e:
            return f"ERROR: {e}"
        if not results:
            return f"No matches for '{query}'."
        lines = [f"{r['file']}:{r['line_number']}: {r['line']}" for r in results]
        out = "\n".join(lines)
        if len(results) >= 40:
            out += "\n[... more matches — refine your query to narrow it down ...]"
        return out

    @tool
    def find_files(name_pattern: str, path: str = ".") -> str:
        """Find files by NAME using a glob pattern (e.g. '*.py', 'test_*.py', '*config*').
        Returns matching file paths relative to the project root. Use this to locate a file
        by its name. To search file contents instead, use search_code."""
        from core.config import get_config

        cfg = get_config()
        try:
            base = ft.resolve(workspace, path)
        except PermissionError as e:
            return f"ERROR: {e}"
        if not base.exists():
            return f"Path not found: {path}"

        ignore_dirs = set(cfg.ignore_dirs)
        matches: list[str] = []
        for p in sorted(base.rglob(name_pattern)):
            if not p.is_file():
                continue
            if any(part in ignore_dirs for part in p.parts):
                continue
            matches.append(str(p.relative_to(workspace)))
            if len(matches) >= 100:
                break
        if not matches:
            return f"No files matching '{name_pattern}'."
        out = "\n".join(matches)
        if len(matches) >= 100:
            out += "\n[... truncated at 100 — narrow the pattern ...]"
        return out

    @tool
    def run_shell(command: str) -> str:
        """Run a shell command in the project root and return its combined output and exit code.
        Depending on settings the user may be asked to confirm first. Use for installing
        dependencies, running tests, git, builds, etc."""
        result = run_with_confirmation(command, cwd=workspace)
        if result is None:
            return "The user declined to run this command."
        stdout, stderr, code = result
        out = stdout or ""
        if stderr:
            out += f"\n[stderr]\n{stderr}"
        out = out.strip()
        if len(out) > MAX_SHELL_OUTPUT:
            out = out[:MAX_SHELL_OUTPUT] + "\n[... output truncated ...]"
        return f"exit code: {code}\n{out}".strip()

    return [list_files, find_files, search_code, read_file, write_file, edit_file, run_shell]
