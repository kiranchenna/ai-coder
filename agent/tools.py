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

import shlex
import sys
from pathlib import Path

from langchain_core.tools import tool
from rich.prompt import Confirm

import tools.file_tools as ft
from core.console import SafeConsole
from tools.shell_tools import run_with_confirmation

console = SafeConsole()

# Diffs from actual (applied, not declined/no-op) writes this turn — drained
# by AgentSession._exec() right after each tool call so the session log (see
# agent/loop.py) can attach the real diff to that specific action without
# _apply_write needing a reference to the session, and without changing what
# any tool actually returns to the model (which stays a terse status string).
_pending_diffs: list[tuple[str, str]] = []


def _shell_quote(s: str) -> str:
    """Quote a string for the active platform's shell (subprocess shell=True)."""
    if sys.platform == "win32":
        # cmd.exe doesn't treat single quotes as quoting — use double quotes.
        return '"' + str(s).replace('"', '""') + '"'
    return shlex.quote(str(s))

MAX_READ_CHARS = 20_000
MAX_SHELL_OUTPUT = 6_000
READ_DEFAULT_LINES = 500


def locate_edit(content: str, old: str) -> tuple[int, int, str] | tuple[None, str]:
    """
    Find where `old` occurs in `content`, tolerating whitespace differences that
    trip up small models. Returns (start, end, how) for a single match, or
    (None, reason) where reason is "ambiguous" or "not_found".

    Match precedence: exact → trailing-whitespace-insensitive (per line) →
    indentation/whitespace-insensitive (per line, stripped). Looser tiers only
    apply when the exact tier finds nothing, and a tier matching >1 place is
    reported ambiguous rather than guessed.
    """
    content_lines = content.split("\n")
    old_lines = old.split("\n")
    n = len(old_lines)

    # char offset where each content line starts
    offsets, pos = [], 0
    for ln in content_lines:
        offsets.append(pos)
        pos += len(ln) + 1  # + newline

    def line_span(i: int) -> tuple[int, int]:
        last = i + n - 1
        return offsets[i], offsets[last] + len(content_lines[last])

    # Line-aligned tiers first (so a whole logical line is replaced, never a
    # fragment mid-indentation): exact lines → trailing-ws-insensitive →
    # indentation/whitespace-insensitive.
    if 0 < n <= len(content_lines):
        for label, norm in (("exact", lambda s: s),
                            ("fuzzy", lambda s: s.rstrip()),
                            ("fuzzy", lambda s: s.strip())):
            target = [norm(line) for line in old_lines]
            cnorm = [norm(line) for line in content_lines]
            hits = [i for i in range(len(content_lines) - n + 1) if cnorm[i:i + n] == target]
            if len(hits) == 1:
                start, end = line_span(hits[0])
                return start, end, label
            if len(hits) > 1:
                return None, "ambiguous"

    # Fallback: within-line substring (sub-line edits / renames).
    sub = content.count(old)
    if sub == 1:
        i = content.index(old)
        return i, i + len(old), "exact"
    if sub > 1:
        return None, "ambiguous"
    return None, "not_found"


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _reindent_to_match(matched: str, new_string: str) -> str:
    """
    Re-base new_string's indentation to the matched region's actual indentation.
    Used after a fuzzy (whitespace-insensitive) match so the model getting the
    leading indentation wrong doesn't corrupt the file. Relative indentation
    within new_string is preserved; a no-op when it already matches.
    """
    file_indent = _leading_ws(matched.split("\n", 1)[0])
    lines = new_string.split("\n")
    first_nonempty = next((line for line in lines if line.strip()), "")
    new_base = _leading_ws(first_nonempty)
    if new_base == file_indent:
        return new_string
    rebased = []
    for line in lines:
        if not line.strip():
            rebased.append("")
        elif line.startswith(new_base):
            rebased.append(file_indent + line[len(new_base):])
        else:
            rebased.append(file_indent + line.lstrip())
    return "\n".join(rebased)


def _format_hits(hits: list[dict]) -> str:
    """Render knowledge-base search results as sourced, separated blocks."""
    return "\n\n---\n\n".join(
        f"[source: {h['metadata'].get('source') or h['metadata'].get('title') or 'cached'}]\n"
        f"{h['content']}"
        for h in hits
    )


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
        ft.backup_file(workspace, path)

    ft.write_file(workspace, path, new_content)
    if diff.strip():
        _pending_diffs.append((path, diff))
    verb = "Created" if is_new else "Updated"
    return f"{verb} {path} ({len(new_content)} chars)."


def build_tools(workspace: Path) -> list:
    """Return the list of LangChain tools bound to this workspace."""
    from core.config import project_id

    proj = project_id(workspace)  # scope per-project documents; web stays global

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
    def read_file(path: str, offset: int = 1, limit: int = 0) -> str:
        """Read a file's text (relative to the project root). Always read a file before
        editing it. For LARGE files, page through them: `offset` is the 1-based start line and
        `limit` is the number of lines to read (0 = to the end, or a default window if the file
        is large). The returned text is raw (no line numbers), so it can be copied into
        edit_file directly."""
        try:
            content = ft.read_file(workspace, path)
        except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
            return f"ERROR: {e}"

        lines = content.split("\n")
        total = len(lines)
        start = max(1, offset)
        if start > total:
            return f"ERROR: offset {start} is past the end of the file ({total} lines)."
        if limit and limit > 0:
            end = min(total, start + limit - 1)
        elif start == 1 and total > READ_DEFAULT_LINES:
            end = READ_DEFAULT_LINES            # large file, no range → first window
        else:
            end = total

        body = "\n".join(lines[start - 1:end])
        if len(body) > MAX_READ_CHARS:
            return (body[:MAX_READ_CHARS]
                    + f"\n[... truncated at {MAX_READ_CHARS} chars; narrow with offset/limit ...]")
        if end < total:
            body += f"\n[showing lines {start}-{end} of {total}; read more with offset/limit]"
        elif start > 1:
            body += f"\n[showing lines {start}-{total} of {total}]"
        return body

    @tool
    def write_file(path: str, content: str) -> str:
        """Create a new file, or completely overwrite an existing one, with the given content.
        Shows a diff and (per settings) asks the user to confirm. Prefer edit_file for small
        changes to existing files."""
        return _apply_write(workspace, path, content)

    @tool
    def edit_file(path: str, old_string: str, new_string: str) -> str:
        """Replace one occurrence of old_string with new_string in an existing file.
        old_string should match the file's text and be unique; matching tolerates minor
        whitespace/indentation differences, but include enough surrounding context to be
        unambiguous. Shows a diff to confirm."""
        try:
            original = ft.read_file(workspace, path)
        except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
            return f"ERROR: {e}"

        if old_string == new_string:
            return "ERROR: old_string and new_string are identical — nothing to change."

        located = locate_edit(original, old_string)
        if located[0] is None:
            if located[1] == "ambiguous":
                return ("ERROR: old_string matches more than one place — it must be unique. "
                        "Include more surrounding lines so it matches exactly one location.")
            return ("ERROR: old_string was not found in the file. Read the file again and "
                    "copy the exact text you want to replace.")

        start, end, how = located
        # On a fuzzy match, re-base the replacement to the file's real indentation
        # so a model that mis-indents old/new_string can't corrupt the file.
        applied = _reindent_to_match(original[start:end], new_string) if how == "fuzzy" else new_string
        updated = original[:start] + applied + original[end:]
        result = _apply_write(workspace, path, updated, existing=original)
        if how == "fuzzy" and not result.startswith("ERROR") and not result.startswith("User declined"):
            result += " (matched ignoring whitespace)"
        return result

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
    def find_symbol(name: str) -> str:
        """Find where a function / class / type / symbol is DEFINED across the project, using a
        fast definitions index. Returns 'kind name — file:line' for each definition. Use this to
        jump straight to a definition instead of grepping. Matches the exact name first, then
        falls back to a substring match."""
        from core.code_index import build_symbol_index
        from core.config import get_config

        index = build_symbol_index(workspace, set(get_config().ignore_dirs))
        exact = index.get(name)
        if exact:
            return "\n".join(f"{h['kind']} {name} — {h['file']}:{h['line']}" for h in exact)

        q = name.lower()
        fuzzy = [
            f"{h['kind']} {sym} — {h['file']}:{h['line']}"
            for sym, locs in sorted(index.items())
            if q in sym.lower()
            for h in locs
        ]
        if not fuzzy:
            return f"No symbol matching '{name}' found in the index."
        if len(fuzzy) > 40:
            fuzzy = fuzzy[:40] + ["[... more; refine the name ...]"]
        return "\n".join(fuzzy)

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
            matches.append(str(p.relative_to(workspace)).replace("\\", "/"))
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
        from rag.research import research_topic

        try:
            # this project's docs + global web cache
            cached = KnowledgeBase.get().search(query, n=4, project=proj)
        except Exception:
            cached = []
        if cached:
            return f"(from cached knowledge)\n\n{_format_hits(cached)}"

        # Cache miss → research and cache globally (web knowledge is shared).
        result = research_topic(query, project="")
        if result["count"] == 0:
            return f"No web results found for '{query}'."
        return result["text"]

    @tool
    def fetch_url(url: str) -> str:
        """Fetch a specific web page (e.g. an official docs page), extract its readable text,
        cache it in the knowledge base, and return the content. Use when you already have a
        specific URL to read."""
        from rag.research import cache_url
        from tools.web_tools import is_fetch_error

        n, page = cache_url(url, project="")
        if is_fetch_error(page):
            return page
        if len(page) > MAX_READ_CHARS:
            page = page[:MAX_READ_CHARS] + "\n[... truncated ...]"
        return page

    @tool
    def run_checks() -> str:
        """Run the project's linters and type checkers (e.g. ruff, mypy, eslint, tsc, clippy,
        go vet) to catch style and type errors. Auto-detected and read-only. Use this
        alongside run_tests after editing code."""
        from core.project import detect_lint_commands
        from tools.shell_tools import run_command

        checks = detect_lint_commands(workspace)
        if not checks:
            return "No linters or type checkers are configured for this project."

        parts = []
        for command, label in checks:
            stdout, stderr, code = run_command(command, cwd=workspace, timeout=180,
                                               stream_output=False)
            body = ((stdout or "") + ("\n" + stderr if stderr else "")).strip()
            if len(body) > MAX_SHELL_OUTPUT:
                body = "[... truncated ...]\n" + body[-MAX_SHELL_OUTPUT:]
            status = "OK" if code == 0 else "ISSUES"
            parts.append(f"[{label}] {status} (exit {code})" + (f"\n{body}" if body else ""))
        return "\n\n".join(parts)

    @tool
    def git_status() -> str:
        """Show the project's git status (changed and untracked files). Read-only."""
        from tools.shell_tools import run_command

        out, err, code = run_command("git status --short --branch", cwd=workspace,
                                     stream_output=False)
        if code != 0:
            return f"git status failed: {(err or out).strip()}"
        return out.strip() or "(working tree clean)"

    @tool
    def git_diff(path: str = "") -> str:
        """Show the working-tree git diff (unstaged + staged changes) for the project,
        optionally limited to one path. Read-only — use this to review what you changed."""
        from tools.shell_tools import run_command

        cmd = "git diff HEAD" + (f" -- {_shell_quote(path)}" if path else "")
        out, err, code = run_command(cmd, cwd=workspace, stream_output=False)
        if code != 0:
            return f"git diff failed: {(err or out).strip()}"
        out = out.strip()
        if not out:
            return "(no changes vs HEAD)"
        if len(out) > MAX_READ_CHARS:
            out = out[:MAX_READ_CHARS] + "\n[... diff truncated ...]"
        return out

    @tool
    def git_commit(message: str) -> str:
        """Stage all changes and create a git commit with the given message. The user may be
        asked to confirm. Use after a coherent set of edits the user is satisfied with."""
        # Stage everything except the agent's .bak backups, then commit. Double
        # quotes around the pathspec work on both POSIX (no glob expansion) and
        # cmd.exe (quotes stripped, git receives *.bak).
        result = run_with_confirmation(
            f'git add -A && git reset -q -- "*.bak" && git commit -m {_shell_quote(message)}',
            cwd=workspace,
        )
        if result is None:
            return "The user declined to commit."
        out, err, code = result
        combined = (out + ("\n" + err if err else "")).strip()
        if code != 0:
            return f"Commit failed (exit {code}):\n{combined}"
        return f"Committed.\n{combined}"

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
                text, source=f"document:{path}", title=path,
                ttl_hours=24 * 365, project=proj,
            )
        except Exception as e:
            # Don't claim success: ingestion failed (e.g. LM Studio/embedding
            # model unavailable). Return the text but make the failure explicit.
            note = (
                f"[WARNING: could not ingest '{path}' into the knowledge base "
                f"({e}); it will NOT be searchable via rag_search. Is the embedding "
                f"model downloaded and LM Studio's server running?]\n\n"
            )
            return note + (text[:MAX_READ_CHARS] if len(text) > MAX_READ_CHARS else text)

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
            # Looser cutoff for recall than for the research cache-hit decision;
            # scoped to this project's docs + global web cache.
            results = KnowledgeBase.get().search(query, n=5, max_distance=0.65, project=proj)
        except Exception as e:
            return f"ERROR: knowledge base unavailable: {e}"
        if not results:
            return "Nothing relevant found in the knowledge base yet."
        return _format_hits(results)

    return [
        list_files, find_files, find_symbol, search_code, read_file,
        write_file, edit_file, run_shell,
        research, fetch_url, rag_search, read_document, run_tests, run_checks,
        git_status, git_diff, git_commit,
        remember, recall,
    ]
