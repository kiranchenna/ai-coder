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

    @tool
    def research(query: str) -> str:
        """Research a topic on the web for CURRENT, up-to-date information — library
        versions, new APIs, recent best practices the model may not know. Checks the local
        knowledge base first; if nothing fresh is cached, searches the web, caches the
        findings, and returns them WITH their source URLs. Use this whenever you are unsure
        about current/external facts instead of guessing."""
        from rag.store import KnowledgeBase
        from tools.web_tools import search_web, format_search_results
        from tools.web_tools import fetch_url as _fetch

        kb = KnowledgeBase.get()
        try:
            cached = kb.search(query, n=4)
        except Exception:
            cached = []
        if cached:
            body = "\n\n---\n\n".join(
                f"[source: {c['metadata'].get('source') or c['metadata'].get('title') or 'cached'}]\n"
                f"{c['content']}"
                for c in cached
            )
            return f"(from cached knowledge)\n\n{body}"

        results = search_web(query)
        if not results:
            return f"No web results found for '{query}'."
        summary = format_search_results(results)
        try:
            kb.add(summary, source="web-search", title=query, ttl_hours=12)
        except Exception:
            pass

        extra = ""
        url = results[0].get("href") or results[0].get("url", "")
        if url:
            page = _fetch(url)
            if page and not page.startswith("[Error") and not page.startswith("[Non-text") \
                    and len(page) > 200:
                try:
                    kb.add(page, source=url, title=results[0].get("title", query), ttl_hours=48)
                except Exception:
                    pass
                extra = f"\n\n---\n\n**Source: {url}**\n\n{page[:2500]}"
        return f"{summary}{extra}"

    @tool
    def fetch_url(url: str) -> str:
        """Fetch a specific web page (e.g. an official docs page), extract its readable text,
        cache it in the knowledge base, and return the content. Use when you already have a
        specific URL to read."""
        from rag.store import KnowledgeBase
        from tools.web_tools import fetch_url as _fetch

        page = _fetch(url)
        if page.startswith("[Error") or page.startswith("[Non-text"):
            return page
        try:
            KnowledgeBase.get().add(page, source=url, title=url, ttl_hours=48)
        except Exception:
            pass
        if len(page) > MAX_READ_CHARS:
            page = page[:MAX_READ_CHARS] + "\n[... truncated ...]"
        return page

    @tool
    def remember(note: str, category: str = "note") -> str:
        """Save a durable fact about THIS project to long-term memory so you and future
        sessions recall it later — an architectural decision, a convention to follow, an
        important fact, or a TODO. `category` is a short tag like 'decision', 'convention',
        'fact', or 'todo'. Use this whenever something is worth remembering across sessions."""
        from memory.project import ProjectMemory

        entry = ProjectMemory(workspace).add(note, category)
        return f"Remembered [{entry['category']}]: {entry['text']}"

    @tool
    def recall(query: str = "") -> str:
        """Recall durable facts you've saved about this project. With no query, returns all
        saved project memory; with a query, returns matching entries. Use this at the start of
        a task to remember prior decisions and conventions."""
        from memory.project import ProjectMemory

        pm = ProjectMemory(workspace)
        items = pm.search(query) if query else pm.all()
        if not items:
            return "No project memory has been saved yet."
        return "\n".join(f"[{it['category']}] {it['text']}" for it in items)

    @tool
    def run_tests() -> str:
        """Run the project's test suite to verify your changes actually work. Auto-detects the
        test command (pytest, npm test, cargo test, go test, make test, etc.). Returns whether
        tests passed or failed plus the output. Use this after editing code, and again after a
        fix, to confirm the change is correct."""
        from core.project import detect_test_command

        detected = detect_test_command(workspace)
        if detected is None:
            return ("Could not auto-detect a test command for this project. If you know it, "
                    "run it with run_shell (e.g. 'pytest -q' or 'npm test').")
        command, label = detected
        result = run_with_confirmation(command, cwd=workspace, timeout=300)
        if result is None:
            return "The user declined to run the tests."
        stdout, stderr, code = result
        out = (stdout or "")
        if stderr:
            out += "\n[stderr]\n" + stderr
        out = out.strip()
        # Failures/summaries land at the END of test output — keep the tail.
        if len(out) > MAX_SHELL_OUTPUT:
            out = "[... earlier output truncated ...]\n" + out[-MAX_SHELL_OUTPUT:]
        status = "PASSED" if code == 0 else "FAILED"
        return f"[{label}] tests {status} (exit code {code}).\n{out}"

    @tool
    def read_document(path: str) -> str:
        """Read a document (PDF, Word .docx, Markdown, .txt, .rst, or HTML) from the project,
        extract its text, ingest it into the knowledge base for later recall, and return the
        text. Use this for PRDs, TDDs, specs, or any product document the user points you at —
        NOT for source code files (use read_file for those)."""
        from rag.ingest import load_document, is_supported
        from rag.store import KnowledgeBase

        try:
            target = ft.resolve(workspace, path)
        except PermissionError as e:
            return f"ERROR: {e}"
        if not target.exists() or not target.is_file():
            return f"ERROR: document not found: {path}"
        if not is_supported(target):
            return (f"ERROR: unsupported document type '{target.suffix}'. "
                    "Supported: PDF, .docx, .md, .txt, .rst, .html.")
        try:
            text = load_document(target)
        except Exception as e:
            return f"ERROR reading document: {e}"
        if not text.strip():
            return f"The document '{path}' contains no extractable text."

        try:
            chunks = KnowledgeBase.get().add(
                text, source=f"document:{path}", title=path, ttl_hours=24 * 365
            )
        except Exception:
            chunks = 0

        note = f"[Ingested '{path}' into the knowledge base as {chunks} chunk(s).]\n\n"
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS] + "\n[... truncated — full text is searchable via rag_search ...]"
        return note + text

    @tool
    def rag_search(query: str) -> str:
        """Search your local knowledge base — web pages and documents you have already
        researched or ingested — for information relevant to the query. Use this to recall
        things you've learned before before searching the web again."""
        from rag.store import KnowledgeBase

        try:
            # Looser cutoff for recall than for the research cache-hit decision.
            results = KnowledgeBase.get().search(query, n=5, max_distance=0.65)
        except Exception as e:
            return f"ERROR: knowledge base unavailable: {e}"
        if not results:
            return "Nothing relevant found in the knowledge base yet."
        return "\n\n---\n\n".join(
            f"[source: {r['metadata'].get('source') or r['metadata'].get('title') or 'cached'}]\n"
            f"{r['content']}"
            for r in results
        )

    return [
        list_files, find_files, search_code, read_file,
        write_file, edit_file, run_shell,
        research, fetch_url, rag_search, read_document, run_tests,
        remember, recall,
    ]
