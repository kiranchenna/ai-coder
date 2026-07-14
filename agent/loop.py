"""
agent/loop.py — The agentic tool-calling loop
==============================================
The heart of AICoder v3. One conversational agent that plans, reads and edits
real code, runs commands, and verifies its work — by calling tools natively.

Flow per user turn:
    user message
      → model.invoke(history)            (may return tool calls)
      → execute each tool, append result
      → repeat until the model returns a plain text answer (or step cap hit)

The loop is deliberately small and transparent so it is easy to debug and
extend in later phases (planner, verify loop, memory).
"""

from __future__ import annotations

import random
import socket
import subprocess
import threading
import time
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text

from core.console import SafeConsole
from core.model import (
    balanced_json_objects,
    extract_text_tool_calls,
    get_chat_model,
    is_lmstudio_endpoint,
)
from agent.prompts import system_prompt
from agent.tools import build_tools

console = SafeConsole()

# Safety cap on tool-call iterations within a single user turn.
MAX_STEPS = 12

# The brand mark echoes the app icon's palette (cyan circuit-bracket, amber
# neural mark, on dark) and its bracket / ">_" terminal-cursor motifs — a
# terminal can't render the actual bitmap logo, so this is the in-terminal
# equivalent (see assets/icon.png for the real logo, used in the README).
BRAND = ("[bold cyan]⟨[/bold cyan][bold yellow1]❯[/bold yellow1][dim yellow1]_[/dim yellow1]"
         "[bold cyan]⟩[/bold cyan]  [bold cyan]AI[/bold cyan][bold yellow1]Coder[/bold yellow1]")

# Large-scale wordmark for the startup banner (see _startup_banner below) —
# a hand-built 5-row block font, "AI" in cyan / "CODER" in yellow1 to match
# BRAND's palette. Kept as a literal (not generated at runtime) since it's a
# one-time decorative asset, not something that needs to render other words.
LOGO = (
    "[bold cyan] ███  ███[/bold cyan] [bold yellow1] ████  ███  ████  █████ ████ [/bold yellow1]\n"
    "[bold cyan]█   █  █ [/bold cyan] [bold yellow1]█     █   █ █   █ █     █   █[/bold yellow1]\n"
    "[bold cyan]█████  █ [/bold cyan] [bold yellow1]█     █   █ █   █ ████  ████ [/bold yellow1]\n"
    "[bold cyan]█   █  █ [/bold cyan] [bold yellow1]█     █   █ █   █ █     █  █ [/bold yellow1]\n"
    "[bold cyan]█   █ ███[/bold cyan] [bold yellow1] ████  ███  ████  █████ █   █[/bold yellow1]"
)

# The five tools shown on the startup banner — a curated highlight reel (the
# full set is longer; see agent/tools.py), picked to sketch the core
# explore → edit → verify loop for someone seeing the app for the first time.
TOOL_HIGHLIGHTS: list[tuple[str, str]] = [
    ("read_file", "read a file before editing it"),
    ("edit_file", "replace one exact match in an existing file"),
    ("run_shell", "run a shell command in the project root"),
    ("run_tests", "run the project's test suite to verify changes"),
    ("search_code", "grep-like search across file contents"),
]

STARTUP_TIPS: list[str] = [
    "Describe a task in plain English — the agent plans, edits, and verifies on its own.",
    "'/plan <goal>' breaks a bigger goal into a resumable, multi-step build.",
    "'/init' analyzes the codebase and writes an AICODER.md so the agent ramps up faster next time.",
    "'/model' switches models mid-session without restarting.",
    "'/review' asks the agent to review the current git diff before you commit.",
    "Ctrl+V pastes a screenshot straight into the chat — handy for UI bugs.",
    "'aicoder --continue' resumes your last conversation for this workspace.",
    "'/context' shows how close you are to the compaction budget.",
]

# Every slash command's name + a one-line description, used by the TUI's "/"
# autocomplete dropdown (agent/tui.py). /help below has its own, richer
# per-command text (argument hints, the live devmode profile) and is kept
# separate rather than derived from this simplified list — update both when
# adding a command.
SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/develop", "Developer Mode: role-driven SDLC design → build (--fast = no back-and-forth)"),
    ("/dev", "resume Developer Mode (status/build/revisit <phase>/resolve)"),
    ("/plan", "decompose a goal into tasks and build it"),
    ("/resume", "continue an in-progress plan"),
    ("/init", "analyze the codebase and write/update AICODER.md"),
    ("/model", "pick a model interactively, or switch straight to <name>"),
    ("/vision", "attach an image by path (or 'model' to pick the vision-capable model)"),
    ("/history", "list past sessions for this workspace, or view one in detail"),
    ("/status", "show workspace, model, provider, and dev-mode profile"),
    ("/context", "show conversation size vs. the compaction budget"),
    ("/compact", "summarize older turns now (usually automatic)"),
    ("/permissions", "view or change shell/file confirmation modes"),
    ("/review", "ask the agent to review the current git diff"),
    ("/tools", "list the agent's tools"),
    ("/mcp", "list connected MCP servers and their tools"),
    ("/hooks", "list configured lifecycle hooks"),
    ("/diff", "show the git diff of changes so far"),
    ("/memory", "show what's remembered about this project"),
    ("/knowledge", "RAG: 'learn <topic|URL>', stats, 'clear[ all]'"),
    ("/export", "save this conversation to a markdown file"),
    ("/doctor", "diagnose the model/tool-calling setup (like --selftest)"),
    ("/bug", "where and what to report if something's wrong"),
    ("/clear", "forget this conversation (keeps saved memory)"),
    ("/help", "show commands"),
    ("/exit", "quit"),
]


def _short(value, limit: int = 60) -> str:
    """One-line preview of a tool argument for display."""
    s = str(value).replace("\n", "\\n")
    return s if len(s) <= limit else s[:limit] + "…"


def _log_safe(value, limit: int = 500):
    """Truncate a value for the session log (see AgentSession._record_action)
    — avoids bloating it with huge file contents/tool outputs; the action's
    own `diffs` already carry the meaningful summary for a write."""
    if isinstance(value, dict):
        return {k: _log_safe(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_log_safe(v, limit) for v in value]
    s = str(value)
    return s if len(s) <= limit else s[:limit] + f"…[+{len(s) - limit} chars truncated]"


def _msg_text(m) -> str:
    """A message's textual content as a string (content may be a list of blocks)."""
    return m.content if isinstance(m.content, str) else str(m.content)


def _msg_chars(m) -> int:
    return len(_msg_text(m))


def _is_actionable_tool_message(content: str) -> bool:
    """
    Decide whether text-emitted tool-call JSON is an actual call the model intends
    to execute, vs. an illustrative example inside an explanation.

    Heuristic: the JSON must *dominate* the message. If the surrounding prose is
    both substantial (> 250 chars) and longer than the JSON itself, the message
    reads as an explanation that happens to contain JSON (e.g. "you could call
    write_file like {…}") — not an actual call. This guards against executing
    hallucinated/example write_file/run_shell calls, which is a safety issue, not
    just a correctness one.
    """
    spans = balanced_json_objects(content)
    if not spans:
        return False
    remainder = content
    for span in spans:
        remainder = remainder.replace(span, "", 1)
    remainder = remainder.replace("```json", "").replace("```", "").strip()
    json_len = sum(len(s) for s in spans)
    if len(remainder) > 250 and len(remainder) > json_len:
        return False
    return True




def _repo_overview(workspace: Path) -> str:
    """Compact repo orientation for the system prompt (best-effort)."""
    try:
        from core.context import WorkspaceContext

        return WorkspaceContext(workspace).overview()
    except Exception:
        return ""


def _project_memory(workspace: Path) -> str:
    """Durable project memory saved in earlier sessions (best-effort)."""
    try:
        from memory.project import ProjectMemory

        return ProjectMemory(workspace).render()
    except Exception:
        return ""


def _has_devmode_session(workspace: Path) -> bool:
    """Whether a Developer Mode design has already been started here."""
    try:
        return (workspace / "docs" / "dev" / "state.json").exists()
    except Exception:
        return False


def _active_work_note(workspace: Path) -> str:
    """Flag an in-progress /plan or /develop session (best-effort), so the
    model points to /resume or /dev status instead of re-suggesting a fresh
    /plan or /develop, or worse, attempting the work itself. This is a
    secondary nudge only — the model doesn't reliably act on it (verified
    live), so the startup banners' deterministic checks (see
    _has_devmode_session / Planner.has_active_plan, printed unconditionally
    before any model call) are the actual reliable signal."""
    notes = []
    try:
        from agent.planner import Planner

        if Planner(workspace, None).has_active_plan():
            notes.append(
                "An unfinished /plan task list already exists for this "
                "project — tell the user to run /resume rather than "
                "starting a new /plan or attempting the work yourself."
            )
    except Exception:
        pass
    try:
        if _has_devmode_session(workspace):
            notes.append(
                "A Developer Mode design (docs/dev/) already exists for "
                "this project — tell the user to run /dev status to see "
                "progress, /dev to resume the design, or /dev build if the "
                "design is done, rather than suggesting a fresh /develop."
            )
    except Exception:
        pass
    return "\n".join(notes)


# Project-instructions files, in load order (global first, then project).
_INSTRUCTION_NAMES = ("AICODER.md", ".aicoder.md", ".aicoderrules")
_INSTRUCTIONS_MAX_CHARS = 6_000


def _load_instructions(workspace: Path) -> str:
    """
    Load user-authored project instructions (a CLAUDE.md-style file). Combines a
    global ~/.aicoder/AICODER.md with the first instructions file found in the
    workspace. Best-effort; returns "" if none.
    """
    from core.config import AICODER_HOME

    parts: list[str] = []
    try:
        global_file = AICODER_HOME / "AICODER.md"
        if global_file.is_file():
            text = global_file.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                parts.append("## Global instructions\n" + text)
    except Exception:
        pass

    for name in _INSTRUCTION_NAMES:
        try:
            f = workspace / name
            if f.is_file():
                text = f.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    parts.append(f"## From {name}\n" + text)
                break
        except Exception:
            continue

    combined = "\n\n".join(parts)
    if len(combined) > _INSTRUCTIONS_MAX_CHARS:
        combined = combined[:_INSTRUCTIONS_MAX_CHARS] + "\n[... instructions truncated ...]"
    return combined


class _TurnInterrupted(Exception):
    """Raised internally when request_interrupt() fires mid-turn — caught by
    send() so a cancelled turn never leaves a half-formed AIMessage in
    history."""


class AgentSession:
    """Holds conversation state and drives the tool-calling loop."""

    def __init__(self, workspace: Path):
        from datetime import datetime, timezone

        self.workspace = workspace
        self._interrupt = threading.Event()
        # The images from the most recent send_with_images() call — lets a
        # follow-up /vision <question> (no path) ask about the same image(s)
        # again without re-attaching (see _handle_vision_command).
        self.last_image_paths: list[Path] = []
        # One JSON file per session (never overwritten across sessions) at
        # MEMORY_DIR/<project_id>/sessions/<session_id>.json — see
        # _save_session_log/_record_turn. session_id doubles as the filename,
        # so it's filesystem-safe (no colons) rather than pure ISO-8601.
        now = datetime.now(timezone.utc)
        self.session_id = now.strftime("%Y-%m-%dT%H-%M-%S-") + f"{now.microsecond:06d}"
        self.session_started_at = now.isoformat()
        self.session_turns: list[dict] = []
        self._current_turn_actions: list[dict] = []
        self.tools = build_tools(workspace)
        # Optional MCP server tools (no-op unless configured).
        from agent.mcp_client import MCPManager

        self.mcp = MCPManager.from_config()
        self.tools += self.mcp.langchain_tools()
        self.tools_by_name = {t.name: t for t in self.tools}
        self.llm = get_chat_model(tools=self.tools)
        from agent.hooks import HookRunner

        self.hooks = HookRunner()
        # Whether the last send() reached a genuine final answer (vs. the step cap).
        self.last_turn_complete = True
        self.instructions = _load_instructions(workspace)
        self.messages = [
            SystemMessage(
                content=system_prompt(
                    workspace,
                    list(self.tools_by_name),
                    _repo_overview(workspace),
                    _project_memory(workspace),
                    self.instructions,
                    _active_work_note(workspace),
                )
            )
        ]
        # Char budget for non-system history before older turns are summarized.
        # ~2 chars/token of the window leaves room for the system prompt + reply.
        from core.config import get_config

        self._history_budget = max(8_000, get_config().model_context_length * 2)

    # ── Public API ────────────────────────────────────────────────────────────

    def send(self, user_input: str) -> str:
        """Process one user message to completion (through any tool calls).

        Sets ``self.last_turn_complete`` to True only when the model produced a
        genuine final answer; False if the step cap was hit first. Callers that
        chain turns (the planner) read this so they don't mark a task "done" when
        the agent actually ran out of steps mid-task.
        """
        self.last_turn_complete = False
        self._interrupt.clear()
        self._current_turn_actions = []
        self._compact_history_if_needed()
        self.messages.append(HumanMessage(content=user_input))

        answer = ""
        try:
            answer = self._run_steps()
            return answer
        except _TurnInterrupted:
            console.print("\n[yellow]Interrupted — back to the prompt.[/yellow]")
            return ""
        finally:
            # Best-effort — a save failure (e.g. disk full) must never mask
            # whatever this turn actually raised, or crash a successful one.
            # Records the turn even when interrupted/step-capped (answer ""),
            # so partial progress is never silently lost from the log.
            self.session_turns.append({
                "prompt": user_input,
                "actions": self._current_turn_actions,
                "answer": answer,
                "completed": self.last_turn_complete,
            })
            self._save_session_log()

    def _sessions_dir(self) -> Path:
        from core.config import MEMORY_DIR, project_id

        return MEMORY_DIR / project_id(self.workspace) / "sessions"

    def _session_log_path(self) -> Path:
        return self._sessions_dir() / f"{self.session_id}.json"

    def _save_session_log(self) -> None:
        """Persist this session — one JSON file per session, never
        overwritten across different sessions (unlike the old single
        conversation.json). Two things live in it: `turns` (prompt/actions
        incl. real file diffs/answer — the human-analyzable log `/history`
        reads) and `raw_messages` (everything after the system prompt, via
        LangChain's messages_to_dict — what `aicoder --continue` restores).
        The system prompt itself is never persisted, always rebuilt fresh."""
        import json

        from langchain_core.messages import messages_to_dict

        try:
            path = self._session_log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({
                    "session_id": self.session_id,
                    "workspace": str(self.workspace),
                    "started_at": self.session_started_at,
                    "turns": self.session_turns,
                    "raw_messages": messages_to_dict(self.messages[1:]),
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001 — best-effort persistence, never fatal
            pass

    def _latest_session_file(self) -> Path | None:
        """The most recently saved *other* session's file for this workspace
        (never this session's own, freshly-created, still-empty one) —
        session_id's timestamp format sorts correctly as plain strings."""
        sessions_dir = self._sessions_dir()
        if not sessions_dir.is_dir():
            return None
        candidates = sorted(
            (p for p in sessions_dir.glob("*.json") if p.stem != self.session_id),
            reverse=True,
        )
        return candidates[0] if candidates else None

    def load_transcript(self) -> bool:
        """Load the most recently saved *other* session for this workspace
        (if any) and append its messages after the fresh system prompt.
        Returns whether anything was actually loaded. Used by
        `aicoder --continue`."""
        import json

        from langchain_core.messages import messages_from_dict

        path = self._latest_session_file()
        if path is None:
            return False
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            restored = messages_from_dict(raw.get("raw_messages", []))
        except Exception:
            return False
        if not restored:
            return False
        self.messages = self.messages[:1] + restored
        return True

    def request_interrupt(self) -> None:
        """Ask the current turn to stop as soon as it safely can — checked
        between streamed chunks and between tool-call steps. Best-effort: a
        chunk already in flight over the network still has to arrive first."""
        self._interrupt.set()

    def describe_images(self, image_paths: list[Path], user_text: str = "") -> str:
        """The vision half of the two-model handoff: a separate vision-capable
        model looks at the given images and describes them in text. Never
        persisted as the session's default driver (unlike /model) — built
        fresh via get_chat_model(model=...) for this one call, with no tools
        bound (a vision model doesn't need the coding toolset, just eyes)."""
        import base64
        import mimetypes

        from core.config import get_config

        cfg = get_config()
        vision_model_name = cfg.vision_model
        if not vision_model_name:
            raise RuntimeError(
                "No vision model configured — set vision.model in "
                "~/.aicoder/config.yaml (e.g. qwen2.5vl:7b) to enable image understanding."
            )

        if _is_lmstudio_endpoint(cfg):
            from core.model import is_lmstudio_model_downloaded

            if is_lmstudio_model_downloaded(vision_model_name) is False:
                raise RuntimeError(
                    f"{vision_model_name} isn't downloaded in LM Studio — grab it there, "
                    "then try again, or pick a different one with /vision model."
                )

        content: list[dict] = [{
            "type": "text",
            "text": user_text or "Describe this image in detail, especially anything that "
                    "looks like a bug, error, or visual issue.",
        }]
        for path in image_paths:
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode()
            # image_url must be an object ({"url": ...}), not a bare string —
            # LM Studio's OpenAI-compatible validation rejects the bare-string
            # form with a 400. This is the actual OpenAI vision API shape.
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            })

        vision_llm = get_chat_model(model=vision_model_name)
        response = vision_llm.invoke([HumanMessage(content=content)])
        return response.content if isinstance(response.content, str) else str(response.content)

    def send_with_images(self, user_input: str, image_paths: list[Path]) -> str:
        """The two-model handoff: describe the attached images with the
        vision model, fold that description into a normal text turn, then run
        it through the ordinary agentic loop (tool calling, editing) with the
        regular coding model — completely unchanged from a plain send()."""
        self.last_image_paths = list(image_paths)
        description = self.describe_images(image_paths, user_input)
        names = ", ".join(p.name for p in image_paths)
        augmented = (
            f"{user_input}\n\n[Attached image(s): {names} — described below by the "
            f"vision model since I can't see images directly]\n{description}"
        )
        return self.send(augmented)

    def _run_steps(self) -> str:
        for _ in range(MAX_STEPS):
            if self._interrupt.is_set():
                raise _TurnInterrupted()
            ai = self._invoke()

            # 1) Native tool calls (preferred path).
            if ai.tool_calls:
                self.messages.append(ai)
                for call in ai.tool_calls:
                    self._render_call(call)
                    result = self._exec(call)
                    self.messages.append(
                        ToolMessage(content=result, tool_call_id=call.get("id", ""))
                    )
                continue

            # 2) Fallback: some local models emit tool calls as JSON text in the
            #    content instead of via native tool calling. Recover and run them.
            content = ai.content or ""
            text_calls = [
                c for c in extract_text_tool_calls(content)
                if c["name"] in self.tools_by_name
            ]
            if text_calls and _is_actionable_tool_message(content):
                # Replace the raw JSON with a short note so the model can't re-read
                # and re-emit the same call on the next turn (avoids re-execution).
                # Phrased as a past-tense fact, not a "(Requested tools: X)"-style
                # directive — a small model tends to imitate the surface FORM of
                # its own prior turn, and a terse request-shaped placeholder gets
                # copied verbatim as a fake tool call on the next turn instead of
                # a real one (observed repeatedly with qwen2.5-coder:7b).
                called = ", ".join(f"`{c['name']}`" for c in text_calls)
                self.messages.append(AIMessage(
                    content=f"I already called {called} just now — its result is below."
                ))
                results = []
                for call in text_calls:
                    self._render_call(call)
                    results.append(f"{call['name']} -> {self._exec(call)}")
                self.messages.append(
                    HumanMessage(
                        content="Tool results:\n" + "\n\n".join(results)
                        + "\n\nContinue using your tools, or give your final answer."
                    )
                )
                continue

            # 3) Genuine final answer.
            self.messages.append(ai)
            text = content.strip()
            console.print()
            console.print(Markdown(text) if text else "[dim](no further response)[/dim]")
            self.hooks.stop(self.workspace)
            self.last_turn_complete = True
            return text

        console.print(
            "[yellow]⚠ Reached the step limit for this turn. "
            "Ask me to continue if there's more to do.[/yellow]"
        )
        return ""

    # ── Internals ─────────────────────────────────────────────────────────────

    def _invoke(self) -> AIMessage:
        """
        Stream the model's response token-by-token into a transient live region
        (plain REPL) or a dedicated status indicator (the TUI — see agent/tui.py;
        rich.live.Live needs a real Console, which the TUI's console adapter
        isn't, so it takes its own path rather than sharing Live).

        The preview is erased when the stream ends, so tool-call JSON doesn't
        linger; final answers are re-rendered as Markdown by send(). Returns a
        clean AIMessage (coerced from the streamed chunk) with any tool_calls.
        Raises on a model/stream failure, or _TurnInterrupted if
        request_interrupt() fired mid-stream, so callers can react rather than
        mistaking either for an empty answer.
        """
        try:
            from agent.tui import is_tui_active, signal_turn_ended, signal_turn_started
            tui_active = is_tui_active()
        except ImportError:  # pragma: no cover — textual is a base dependency
            tui_active = False

        accumulated = None
        if tui_active:
            signal_turn_started()
            try:
                for chunk in self.llm.stream(self.messages):
                    if self._interrupt.is_set():
                        raise _TurnInterrupted()
                    accumulated = chunk if accumulated is None else accumulated + chunk
            finally:
                signal_turn_ended()
        else:
            shown = ""
            with Live(console=console, refresh_per_second=10, transient=True) as live:
                live.update(Text("💭 Thinking…", style="dim italic"))
                for chunk in self.llm.stream(self.messages):
                    if self._interrupt.is_set():
                        raise _TurnInterrupted()
                    accumulated = chunk if accumulated is None else accumulated + chunk
                    piece = chunk.content
                    if isinstance(piece, str) and piece:
                        shown += piece
                        live.update(Text(shown))

        if accumulated is None:
            raise RuntimeError("the model returned an empty response stream")
        # Coerce the streamed AIMessageChunk into a plain AIMessage for history.
        return AIMessage(content=accumulated.content, tool_calls=accumulated.tool_calls or [])

    def _render_call(self, call: dict) -> None:
        from rich.markup import escape

        name = call.get("name", "?")
        args = call.get("args", {}) or {}
        # Escape tool arg values before embedding in a markup string — code
        # snippets routinely contain `[...]` (list literals, `List[int]`
        # type hints) that would otherwise be silently swallowed as a
        # (syntactically valid but meaningless) markup tag rather than shown.
        preview = ", ".join(f"{k}={escape(_short(v))}" for k, v in args.items())
        console.print(f"[cyan]→ {escape(str(name))}[/cyan]([dim]{preview}[/dim])")

    def reset(self) -> None:
        """Forget the conversation, keeping the system prompt (and saved memory)."""
        self.messages = self.messages[:1]

    # ── Context management ──────────────────────────────────────────────────────

    def _compact_history_if_needed(self) -> None:
        """
        Keep the conversation within the context budget by summarizing older
        turns. Preserves the system prompt and the recent turns, and only splits
        at a user-message boundary so a ToolMessage is never orphaned from its
        AIMessage (which would break the next model call).
        """
        if len(self.messages) <= 2:
            return
        system, rest = self.messages[0], self.messages[1:]
        if sum(_msg_chars(m) for m in rest) <= self._history_budget:
            return

        # Smallest user-message boundary whose tail fits the budget (largest
        # recent window we can keep). Tail shrinks as the index advances.
        keep_from = None
        for i, m in enumerate(rest):
            if isinstance(m, HumanMessage):
                if sum(_msg_chars(x) for x in rest[i:]) <= self._history_budget:
                    keep_from = i
                    break
        # keep_from is None (no boundary's tail fits) or a positive index. It can
        # never be 0: rest[0:] is the whole tail, which we already know exceeds the
        # budget (guarded above), so i == 0 fails the fit test. Use an explicit
        # `is None` check rather than a falsy one so the intent is unambiguous.
        if keep_from is None:
            return

        older, recent = rest[:keep_from], rest[keep_from:]
        summary = self._summarize_messages(older)
        note = SystemMessage(content=f"[Summary of earlier conversation]\n{summary}")
        self.messages = [system, note] + recent
        console.print("[dim]📝 Compacted earlier conversation to stay within context.[/dim]")

    def _summarize_messages(self, msgs: list) -> str:
        """Summarize older messages into a compact note (model call, tail fallback)."""
        lines: list[str] = []
        for m in msgs:
            if isinstance(m, SystemMessage):
                lines.append(_msg_text(m)[:500])  # fold in a previous summary
            elif isinstance(m, HumanMessage):
                lines.append("User: " + _msg_text(m)[:400])
            elif isinstance(m, AIMessage):
                text = _msg_text(m).strip()
                if text:
                    lines.append("Assistant: " + text[:400])
                for tc in (m.tool_calls or []):
                    lines.append(f"[called {tc.get('name', '?')}]")
            elif isinstance(m, ToolMessage):
                lines.append("[tool result] " + _msg_text(m)[:200])
        transcript = "\n".join(lines)

        try:
            ai = get_chat_model(precise=True).invoke([
                SystemMessage(content=(
                    "Summarize this earlier conversation between a user and a coding "
                    "agent. Preserve key decisions, files changed, what was done, and "
                    "any unresolved tasks. Be concise (a short paragraph)."
                )),
                HumanMessage(content=transcript[:8000]),
            ])
            summary = ai.content if isinstance(ai.content, str) else str(ai.content)
            return summary.strip() or transcript[-2000:]
        except Exception:
            return transcript[-2000:]  # fallback: keep the most recent context

    def _exec(self, call: dict) -> str:
        name = call.get("name", "")
        args = call.get("args", {}) or {}
        tool = self.tools_by_name.get(name)
        if tool is None:
            result = (
                f"ERROR: unknown tool '{name}'. "
                f"Available tools: {', '.join(self.tools_by_name)}."
            )
            self._record_action(name, args, result, [])
            return result

        # PreToolUse hooks can block the call.
        block = self.hooks.pre_tool_use(name, args, self.workspace)
        if block is not None:
            console.print(f"[yellow]⛔ {name} blocked by hook: {block}[/yellow]")
            result = f"BLOCKED by a PreToolUse hook: {block}"
            self._record_action(name, args, result, [])
            return result

        from agent.tools import _pending_diffs

        del _pending_diffs[:]  # clear any leftovers before this call
        try:
            # StructuredTool validates args against the schema and raises on
            # bad input — surfaced back to the model so it can self-correct.
            result = str(tool.invoke(args))
        except Exception as e:  # noqa: BLE001 — report any tool failure to the model
            result = f"ERROR running {name}: {e}"
        diffs = list(_pending_diffs)
        del _pending_diffs[:]

        # PostToolUse hooks (auto-format, notify, …) may add a note.
        note = self.hooks.post_tool_use(name, args, result, self.workspace)
        if note:
            result += f"\n[hook] {note}"

        self._record_action(name, args, result, diffs)
        return result

    def _record_action(self, name: str, args: dict, result: str, diffs: list) -> None:
        """Append this tool call (with any real file diffs it produced) to
        the current turn's action log — see send()/_save_session_log()."""
        self._current_turn_actions.append({
            "tool": name,
            "args": _log_safe(args),
            "result": _log_safe(result),
            "diffs": [{"path": path, "diff": diff} for path, diff in diffs],
        })


# ── REPL ───────────────────────────────────────────────────────────────────────

def _knowledge_learn(topic: str) -> None:
    """Proactively research a topic (or fetch a URL) and cache it globally."""
    from rag.research import cache_url, research_topic
    from tools.web_tools import is_fetch_error

    # Direct URL → fetch and cache that page.
    if topic.lower().startswith(("http://", "https://")):
        n, page = cache_url(topic, project="")
        if is_fetch_error(page):
            console.print(f"[yellow]{page}[/yellow]")
        else:
            console.print(f"[green]Learned {n} chunk(s) from {topic}.[/green]")
        return

    # Topic → web search + fetch the top pages for depth.
    console.print(f"[dim]🌐 Researching: {topic}…[/dim]")
    result = research_topic(topic, project="")
    if result["count"] == 0:
        console.print(f"[yellow]No web results for '{topic}'.[/yellow]")
        return

    console.print(f"[green]Learned '{topic}' — cached {result['count']} chunk(s) from "
                  f"{len(result['sources'])} source(s).[/green]")
    for s in result["sources"]:
        console.print(f"  [dim]• {s}[/dim]")


def _format_size(num_bytes: int) -> str:
    """Human-readable size, e.g. ``4.7 GB``."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _is_lmstudio_endpoint(cfg) -> bool:
    """Whether the active config's base_url matches LM Studio's default
    local server — model.base_url can still point at any other
    OpenAI-compatible server (a custom local server, a hosted API), where
    LM-Studio-specific `lms` CLI shellouts wouldn't apply."""
    return is_lmstudio_endpoint(cfg.model_base_url)


def _try_lmstudio_load(name: str, cfg) -> None:
    """Best-effort `lms load` — only attempted when the active config
    actually looks like LM Studio's default endpoint (see
    _is_lmstudio_endpoint). Silent if it's not a LM Studio-shaped config;
    surfaced as a warning if it is and the load still fails (almost
    certainly means `name` isn't downloaded)."""
    if not _is_lmstudio_endpoint(cfg):
        return
    from core.model import switch_lmstudio_model

    try:
        switch_lmstudio_model(name)
    except Exception as e:
        console.print(f"[yellow]⚠ Couldn't load '{name}' in LM Studio.[/yellow] [dim]{e}[/dim]")


def _switch_model(name: str, session: "AgentSession") -> None:
    """Switch the active model, persist it as the new default, and rebind tools."""
    from core.config import get_config, save_config
    from core.model import get_chat_model

    cfg = get_config()
    cfg.raw()["model"]["name"] = name
    save_config(cfg.raw())
    session.llm = get_chat_model(tools=session.tools)
    console.print(f"[green]✓ Set model to [bold]{name}[/bold] and saved as your default for "
                  f"new sessions.[/green]")

    # Explicitly unload the old model and load the new one — this one drives
    # every turn, so leaving stale models resident (LM Studio has no default
    # TTL on LLMs, confirmed live) would stack up RAM with every /model
    # switch in a session.
    _try_lmstudio_load(name, cfg)


def _switch_vision_model(name: str) -> None:
    """Persist a new default vision model. Unlike _switch_model, there's no
    session/tools rebind step — the vision model is never bound to a session,
    it's built fresh per call in AgentSession.describe_images."""
    from core.config import get_config, save_config

    cfg = get_config()
    cfg.raw()["vision"]["model"] = name
    save_config(cfg.raw())
    console.print(f"[green]✓ Set vision model to [bold]{name}[/bold] and saved as your "
                  "default.[/green]")

    if _is_lmstudio_endpoint(cfg):
        # No explicit load here, deliberately — LM Studio auto-loads on first
        # request (confirmed live: a chat request to an idle-but-downloaded
        # model loaded it *alongside* the already-loaded coding model, not
        # instead of it). Pre-loading via _try_lmstudio_load would be wrong
        # here: that unloads other LLMs first, which would evict the coding
        # model this vision handoff is supposed to run alongside.
        from core.model import is_lmstudio_model_downloaded

        if is_lmstudio_model_downloaded(name) is False:
            console.print(f"[yellow]⚠ '{name}' doesn't look like it's downloaded in LM "
                          "Studio yet.[/yellow]")


def _handle_custom_model_entry(session: "AgentSession") -> None:
    """'Other…' in the /model picker — an exact model id already downloaded
    in LM Studio (no pull flow — see switch_lmstudio_model's docstring)."""
    name = Prompt.ask("Model id (see `lms ls` for exact ids)", default="").strip()
    if not name:
        console.print("[dim]No change made.[/dim]")
        return
    _switch_model(name, session)


def _build_model_menu(current_name: str, installed: list[dict]) -> list[dict]:
    """Installed models as picker entries, marked against the active model.
    Shared by /model and /vision model."""
    return [
        {"tag": m["name"], "size_bytes": m["size"], "current": m["name"] == current_name}
        for m in installed
    ]


def _model_menu_label(e: dict) -> str:
    """One-line label for a single model-menu entry, used by the TUI's
    arrow-key OptionList (see ask_choice in agent/tui.py). The plain-text
    Rich panel below has its own column-aligned formatting and doesn't share
    this — kept separate to avoid touching its already-tested rendering."""
    check = "[green]✓[/green] " if e["current"] else "  "
    return f"{check}{e['tag']} [dim]({_format_size(e['size_bytes'])})[/dim]"


def _render_model_menu(entries: list[dict], title: str) -> None:
    lines: list[str] = []
    for i, e in enumerate(entries, 1):
        marker = " [dim](current)[/dim]" if e["current"] else ""
        lines.append(f"  [cyan]{i:>2}[/cyan]  {e['tag']:<24} {_format_size(e['size_bytes']):>8}{marker}")
    console.print(Panel("\n".join(lines), title=title, border_style="cyan"))


def _run_model_picker(
    picker_title: str, entries: list[dict], current_name: str,
    other_label: str = "Other… (type an exact LM Studio model id)",
    menu_title: str = "[cyan]Available models (via LM Studio)[/cyan]",
) -> int | None:
    """The shared arrow-key (TUI) or numbered (plain-REPL) model picker over
    `entries` (see _build_model_menu). Returns the chosen index, len(entries)
    for "Other…", or None if cancelled/kept as-is. Shared by /model and
    /vision model."""
    try:
        from agent.tui import ask_choice, is_tui_active
        tui_active = is_tui_active()
    except ImportError:  # pragma: no cover — textual is a base dependency
        tui_active = False

    if tui_active:
        current_idx = next((i for i, e in enumerate(entries) if e["current"]), 0)
        idx = ask_choice(
            picker_title,
            [_model_menu_label(e) for e in entries] + [other_label],
            initial_index=current_idx,
        )
        if idx is None:
            console.print(f"[dim]Kept current model: {current_name}[/dim]")
            return None
        return idx

    _render_model_menu(entries, menu_title)
    choice = Prompt.ask(
        f"Select a model [1-{len(entries)}, 'o' for another model, Enter to keep current]",
        default="",
    ).strip()
    if not choice:
        console.print(f"[dim]Kept current model: {current_name}[/dim]")
        return None
    if choice.lower() in ("o", "other"):
        return len(entries)
    if choice.isdigit() and 1 <= int(choice) <= len(entries):
        return int(choice) - 1
    console.print("[yellow]Invalid selection — no change made.[/yellow]")
    return None


def _handle_model_unreachable(cfg) -> None:
    """`/model` (no arg) when LM Studio can't be reached — show what's
    active and how to change it."""
    console.print(Panel(
        f"[dim]Model:[/dim]    {cfg.model_name}\n"
        f"[dim]Endpoint:[/dim] {cfg.model_base_url}\n\n"
        "[yellow]⚠ Couldn't reach LM Studio to list models.[/yellow]\n"
        "[dim]Make sure its local server is running (Developer tab → Start Server), or "
        "switch directly: [/dim][bold]/model <name>[/bold]",
        title="[cyan]Current model[/cyan]",
        border_style="cyan",
    ))


def _handle_model_command(arg: str, session: "AgentSession") -> None:
    """`/model` — pick a model interactively, or switch straight to
    `/model <name>`. Mirrors the Claude Code `/model` picker: select one and
    it becomes your default for new sessions, not just this one.

    Lists every model already downloaded in LM Studio (via the `lms` CLI) —
    no pull flow, grabbing a new model stays a manual step in LM Studio
    itself (its own `lms get` proved too unreliable this session — bad
    slug/casing resolution, hangs on interactive prompts — to build an
    auto-download flow on top of). Falls back to a status panel if LM Studio
    can't be reached at all."""
    from core.config import get_config

    cfg = get_config()
    if arg:
        _switch_model(arg, session)
        return

    if not _is_lmstudio_endpoint(cfg):
        _handle_model_unreachable(cfg)
        return

    from core.model import list_lmstudio_models

    try:
        installed = list_lmstudio_models()
    except Exception:
        _handle_model_unreachable(cfg)
        return
    if not installed:
        console.print("[yellow]No models downloaded in LM Studio yet.[/yellow] "
                      "[dim]Grab one in LM Studio's own model search, then try /model "
                      "again.[/dim]")
        return

    entries = _build_model_menu(cfg.model_name, installed)
    idx = _run_model_picker("Select a model", entries, cfg.model_name)
    if idx is None:
        return
    if idx == len(entries):
        _handle_custom_model_entry(session)
        return
    _switch_model(entries[idx]["tag"], session)


# ── /vision — attach an image by file path (the two-model vision handoff) ──────
# The TUI's Ctrl+V clipboard paste (agent/tui.py) is the richer version of this
# same flow — both end up calling AgentSession.send_with_images.

def _handle_vision_model_command(arg: str = "") -> None:
    """`/vision model` — pick the vision-capable model interactively, mirroring
    `/model`'s picker but persisting to vision.model, not model.name — the
    coding model actually driving the session is untouched. `/vision model
    <name>` switches straight to <name>, same as `/model <name>` does for
    the coding model. Downloaded models filtered to vision-capable ones only
    (see list_lmstudio_models' vision flag), no pull flow — same reasoning
    as /model."""
    if arg:
        _switch_vision_model(arg)
        return

    from core.config import get_config

    cfg = get_config()
    if not _is_lmstudio_endpoint(cfg):
        console.print(
            "[yellow]⚠ Couldn't reach LM Studio to list models.[/yellow]\n"
            "[dim]Make sure its local server is running, or set vision.model directly in "
            f"config.yaml. Current: {cfg.vision_model or '(not set)'}[/dim]"
        )
        return

    from core.model import list_lmstudio_models

    try:
        installed = list_lmstudio_models(vision_only=True)
    except Exception:
        console.print(
            "[yellow]⚠ Couldn't reach LM Studio to list models.[/yellow]\n"
            "[dim]Make sure its local server is running, or set vision.model directly in "
            f"config.yaml. Current: {cfg.vision_model or '(not set)'}[/dim]"
        )
        return
    if not installed:
        console.print(
            "[yellow]No vision-capable models downloaded in LM Studio yet.[/yellow] "
            "[dim]Grab one (e.g. a Qwen-VL/InternVL build) in LM Studio's own model "
            "search, then try /vision model again.[/dim]"
        )
        return

    entries = _build_model_menu(cfg.vision_model, installed)
    idx = _run_model_picker(
        "Select a vision model", entries, cfg.vision_model or "(not set)",
        menu_title="[cyan]Available vision models (via LM Studio)[/cyan]",
    )
    if idx is None:
        return

    if idx == len(entries):
        name = Prompt.ask("Vision model id (see `lms ls`)", default="").strip()
        if not name:
            console.print("[dim]No change made.[/dim]")
            return
        _switch_vision_model(name)
        return

    _switch_vision_model(entries[idx]["tag"])


def _handle_vision_command(arg: str, session: "AgentSession") -> None:
    if not arg:
        console.print("[yellow]Usage: /vision <path to an image> [optional question/context], "
                      "or /vision model to pick the vision model[/yellow]")
        return

    parts = arg.split(maxsplit=1)
    if parts[0].lower() in ("model", "models"):
        _handle_vision_model_command(parts[1].strip() if len(parts) > 1 else "")
        return

    path_str, user_text = parts[0], (parts[1] if len(parts) > 1 else "")

    # Deliberately not the sandboxed resolve() used by read_file/write_file —
    # this only reads bytes to show a model, and screenshots typically live
    # outside the workspace (Desktop, /tmp, Downloads), not inside it.
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (session.workspace / path).resolve()

    if not path.exists() or not path.is_file():
        # Not a valid path — a natural-language follow-up about the same
        # image(s) as last time ("/vision what about the top-right corner?"),
        # rather than requiring the image to be re-attached for every
        # question. Only kicks in when there's something to follow up on.
        if session.last_image_paths:
            try:
                session.send_with_images(arg, session.last_image_paths)
            except RuntimeError as e:
                console.print(f"[red]⚠ {e}[/red]")
            return
        console.print(f"[red]⚠ Image not found: {path_str}[/red]")
        return

    try:
        session.send_with_images(user_text, [path])
    except RuntimeError as e:
        console.print(f"[red]⚠ {e}[/red]")


# ── /history — browse and view past sessions (prompts, actions, diffs) ─────────

def _list_sessions(workspace: Path) -> list[Path]:
    """Every saved session file for this workspace, most recent first —
    session_id's timestamp format sorts correctly as plain filename strings."""
    from core.config import MEMORY_DIR, project_id

    sessions_dir = MEMORY_DIR / project_id(workspace) / "sessions"
    if not sessions_dir.is_dir():
        return []
    return sorted(sessions_dir.glob("*.json"), reverse=True)


def _handle_history_command(arg: str, session: "AgentSession", workspace: Path) -> None:
    files = _list_sessions(workspace)
    if not files:
        console.print("[dim]No saved sessions yet for this workspace.[/dim]")
        return

    if not arg:
        _render_session_list(files, session)
        return

    if not arg.isdigit() or not (1 <= int(arg) <= len(files)):
        console.print(f"[yellow]Usage: /history [1-{len(files)}][/yellow]")
        return
    _render_session_detail(files[int(arg) - 1])


def _render_session_list(files: list[Path], session: "AgentSession") -> None:
    import json

    from rich.markup import escape

    lines = []
    for i, path in enumerate(files, 1):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — skip an unreadable session, don't crash the list
            continue
        turns = data.get("turns", [])
        first_prompt = turns[0]["prompt"] if turns else "(no messages yet)"
        touched = sorted({
            d["path"] for t in turns for a in t.get("actions", []) for d in a.get("diffs", [])
        })
        marker = " [dim](current)[/dim]" if data.get("session_id") == session.session_id else ""
        started = data.get("started_at", "")[:16].replace("T", " ")
        detail = f"{len(turns)} turn(s)"
        if touched:
            # A stored prompt/path is arbitrary past user text — escape before
            # embedding, or a message like "check the [foo] case" silently
            # loses "[foo]" (a syntactically valid, if meaningless, markup
            # tag doesn't raise, it just consumes the bracketed text).
            detail += f" · files: {', '.join(escape(p) for p in touched)}"
        lines.append(f"[cyan]{i:>2}.[/cyan] {started}  [bold]{escape(_short(first_prompt, 50))}[/bold]"
                     f"{marker}\n      [dim]{detail}[/dim]")

    console.print(Panel("\n\n".join(lines), title="[cyan]Past sessions[/cyan]", border_style="cyan"))
    console.print(f"[dim]Use '/history <n>' (1-{len(files)}) to view one in detail.[/dim]")


def _render_session_detail(path: Path) -> None:
    import json

    from rich.markup import escape
    from rich.syntax import Syntax

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 — surface a clean message, not a traceback
        console.print(f"[red]⚠ Couldn't read that session: {escape(str(e))}[/red]")
        return

    console.print(Panel.fit(
        f"[dim]Workspace:[/dim] {escape(str(data.get('workspace', '')))}\n"
        f"[dim]Started:[/dim]   {escape(str(data.get('started_at', '')))}",
        title=f"[cyan]Session {escape(str(data.get('session_id', '')))}[/cyan]",
        border_style="cyan",
    ))
    for i, turn in enumerate(data.get("turns", []), 1):
        console.print(f"\n[bold cyan]── Turn {i} ──[/bold cyan]")
        console.print(f"[bold green]> {escape(turn['prompt'])}[/bold green]")
        for action in turn.get("actions", []):
            preview = ", ".join(
                f"{k}={escape(_short(v, 40))}" for k, v in (action.get("args") or {}).items()
            )
            console.print(f"[cyan]→ {escape(str(action['tool']))}[/cyan]([dim]{preview}[/dim])")
            for d in action.get("diffs", []):
                console.print(Syntax(d["diff"], "diff", theme="monokai", word_wrap=True))
        if turn.get("answer"):
            console.print(f"\n{turn['answer']}", markup=False)
        else:
            console.print("[dim](no answer — interrupted or hit the step limit)[/dim]")


# ── /init — analyze the codebase and write/update AICODER.md ───────────────────

_INIT_PROMPT = (
    "Write a comprehensive AICODER.md file for this project, documenting: its "
    "purpose and tech stack, the folder/file structure and where key logic "
    "lives, the coding conventions actually used (naming, formatting, error "
    "handling, docstring style), how to run tests and linters, and any other "
    "rules a coding assistant should follow to work effectively in this repo. "
    "If an AICODER.md already exists, read it first and refine/update it rather "
    "than starting from scratch — preserve anything still accurate.\n\n"
    "Do this now, autonomously — do NOT ask me any questions or wait for "
    "confirmation; there is no one else to answer. Work STEP BY STEP, one tool "
    "at a time (do not batch tool calls, and do not describe a tool call in "
    "prose instead of actually calling it):\n"
    "1. list_files to see the top-level layout.\n"
    "2. read_file on each of a representative sample of files (pick the ones "
    "that best show the project's structure and conventions — you decide how "
    "many, based on what you see).\n"
    "3. Once you've read enough to write accurately, call write_file with the "
    "complete AICODER.md content — do not stop before this step.\n"
    "4. Give a short summary of what you documented."
)


def _reload_instructions(session: "AgentSession", workspace: Path) -> None:
    """Refresh the system prompt's loaded AICODER.md content — e.g. after
    /init writes or updates one, so this session honours it immediately
    instead of requiring a restart."""
    session.instructions = _load_instructions(workspace)
    session.messages[0] = SystemMessage(
        content=system_prompt(
            workspace, list(session.tools_by_name),
            _repo_overview(workspace), _project_memory(workspace), session.instructions,
            _active_work_note(workspace),
        )
    )


def _handle_init_command(session: "AgentSession", workspace: Path) -> None:
    console.print("[dim]🔎 Analyzing the codebase to write AICODER.md…[/dim]")
    session.send(_INIT_PROMPT)
    _reload_instructions(session, workspace)


# ── /status — session info on demand (the startup banner's content, reprinted) ──

def _handle_status_command(session: "AgentSession", workspace: Path) -> None:
    from core.config import get_config

    cfg = get_config()
    console.print(Panel(
        f"{BRAND}\n"
        f"[dim]Workspace:[/dim] {workspace}\n"
        f"[dim]Provider:[/dim]  {cfg.model_provider}\n"
        f"[dim]Model:[/dim]     {cfg.model_name}\n"
        f"[dim]Dev Mode:[/dim]  {cfg.devmode_profile()} profile\n"
        f"[dim]Tools:[/dim]     {len(session.tools_by_name)} available",
        title="[cyan]Status[/cyan]", border_style="cyan",
    ))


# ── /context — how full the conversation is vs. the compaction budget ───────────

def _handle_context_command(session: "AgentSession") -> None:
    used = sum(_msg_chars(m) for m in session.messages[1:])
    budget = session._history_budget
    pct = min(100, round(100 * used / budget)) if budget else 0
    note = " — will compact on the next turn" if used > budget else ""
    console.print(Panel(
        f"Conversation size: ~{used:,} chars\n"
        f"Compaction budget: ~{budget:,} chars\n"
        f"Usage: {pct}%{note}",
        title="[cyan]Context usage[/cyan]", border_style="cyan",
    ))


# ── /compact — manually trigger the same compaction that runs automatically ─────

def _handle_compact_command(session: "AgentSession") -> None:
    if len(session.messages) <= 2:
        console.print("[dim]Nothing to compact yet.[/dim]")
        return
    if sum(_msg_chars(m) for m in session.messages[1:]) <= session._history_budget:
        console.print("[dim]Conversation is within budget — nothing to compact.[/dim]")
        return
    session._compact_history_if_needed()  # prints its own "📝 Compacted…" note


# ── /permissions — view/change the shell & file confirmation modes ─────────────

def _handle_permissions_command(arg: str) -> None:
    from core.config import get_config

    cfg = get_config()
    if not arg:
        console.print(Panel(
            f"[dim]Shell commands:[/dim] {cfg.shell_confirmation}  [dim](always | smart | never)[/dim]\n"
            f"[dim]File writes:[/dim]    {cfg.file_confirmation}  [dim](always | auto | never)[/dim]\n\n"
            "[dim]/permissions shell <mode>   — change shell confirmation\n"
            "/permissions files <mode>   — change file-write confirmation[/dim]",
            title="[cyan]Permissions[/cyan]", border_style="cyan",
        ))
        return
    kind, _, mode = arg.partition(" ")
    kind, mode = kind.strip().lower(), mode.strip().lower()
    try:
        if kind == "shell":
            cfg.set_shell_confirmation(mode)
            console.print(f"[green]✓ Shell confirmation set to {mode}.[/green]")
        elif kind == "files":
            cfg.set_file_confirmation(mode)
            console.print(f"[green]✓ File confirmation set to {mode}.[/green]")
        else:
            console.print("[yellow]Usage: /permissions [shell|files] <mode>[/yellow]")
    except ValueError as e:
        console.print(f"[yellow]{e}[/yellow]")


# ── /mcp — connected MCP servers and their tools ────────────────────────────────

def _handle_mcp_command(session: "AgentSession") -> None:
    servers = session.mcp.status()
    if not servers:
        console.print("[dim]No MCP servers configured. Add them under `mcp.servers` in "
                      "config.yaml.[/dim]")
        return
    lines = []
    for s in servers:
        icon = "[green]●[/green]" if s["connected"] else "[red]●[/red]"
        tools = ", ".join(s["tools"]) if s["tools"] else "[dim](no tools)[/dim]"
        lines.append(f"{icon} [bold]{s['name']}[/bold] — {tools}")
    console.print(Panel("\n".join(lines), title="[cyan]MCP servers[/cyan]", border_style="cyan"))


# ── /hooks — configured lifecycle hooks ─────────────────────────────────────────

def _handle_hooks_command() -> None:
    from core.config import get_config

    hooks = get_config().get("hooks", default={}) or {}
    if not any(hooks.get(event) for event in ("PreToolUse", "PostToolUse", "Stop")):
        console.print("[dim]No hooks configured. Add them under `hooks` in config.yaml.[/dim]")
        return
    lines = []
    for event in ("PreToolUse", "PostToolUse", "Stop"):
        for h in hooks.get(event) or []:
            matcher = f" [dim](matcher: {h['matcher']})[/dim]" if h.get("matcher") else ""
            lines.append(f"[bold]{event}[/bold]{matcher} → {h.get('command', '?')}")
    console.print(Panel("\n".join(lines), title="[cyan]Hooks[/cyan]", border_style="cyan"))


# ── /review — ask the agent to review the working-tree diff ────────────────────

def _handle_review_command(session: "AgentSession") -> None:
    session.send(
        "Review the current git diff (git_diff) for correctness bugs and for "
        "reuse/simplification/efficiency issues in what changed. Report findings "
        "concisely, file:line where relevant; if it looks solid, say so."
    )


# ── /bug — where and what to report ─────────────────────────────────────────────

def _handle_bug_command() -> None:
    console.print(Panel(
        "File a bug at: [bold]https://github.com/kiranchenna/ai-coder/issues/new[/bold]\n\n"
        "Please include:\n"
        "  • [dim]aicoder --version[/dim] and [dim]aicoder --config[/dim] output\n"
        "  • Your OS, LM Studio version, and the model in use\n"
        "  • The exact command/prompt and the full error or unexpected output\n"
        "  • Whether [dim]aicoder --selftest[/dim] passes",
        title="[cyan]Report a bug[/cyan]", border_style="cyan",
    ))


# ── /doctor — the --selftest diagnostic, callable without restarting ───────────

def _handle_doctor_command() -> None:
    from core.model import selftest

    selftest()


# ── /export — dump the conversation transcript to a file ───────────────────────

def _format_transcript(messages: list) -> str:
    lines = []
    for m in messages[1:]:  # skip the system prompt
        if isinstance(m, HumanMessage):
            lines.append(f"## You\n\n{_msg_text(m)}\n")
        elif isinstance(m, AIMessage):
            text = _msg_text(m).strip()
            if text:
                lines.append(f"## AICoder\n\n{text}\n")
            for tc in (m.tool_calls or []):
                lines.append(f"_called `{tc.get('name', '?')}`_\n")
        elif isinstance(m, ToolMessage):
            lines.append(f"```\n{_msg_text(m)[:2000]}\n```\n")
        elif isinstance(m, SystemMessage):
            lines.append(f"> _{_msg_text(m)[:300]}_\n")  # a compaction note
    return "\n".join(lines)


def _handle_export_command(arg: str, session: "AgentSession", workspace: Path) -> None:
    from datetime import datetime

    import tools.file_tools as ft

    if len(session.messages) <= 1:
        console.print("[dim]Nothing to export yet.[/dim]")
        return
    filename = arg.strip() or f"aicoder-transcript-{datetime.now():%Y%m%d-%H%M%S}.md"
    try:
        target = ft.resolve(workspace, filename)
    except PermissionError as e:
        console.print(f"[yellow]{e}[/yellow]")
        return
    target.write_text(_format_transcript(session.messages), encoding="utf-8")
    console.print(f"[green]✓ Exported conversation to {filename}[/green]")


# ── /develop, /dev, /plan, /resume — moved under "/" for a consistent command
# surface (previously bare words: `develop ...`, `dev ...`, `plan ...`, `resume`) ─

def _handle_develop_command(arg: str, workspace: Path) -> None:
    idea = arg
    fast = False
    if idea.split(" ", 1)[0] in ("--fast", "-f"):
        fast = True
        idea = idea.split(" ", 1)[1].strip() if " " in idea else ""
    if not idea:
        console.print("[yellow]Usage: /develop [--fast] <your project idea>[/yellow]")
        return
    from devmode.session import DevSession

    DevSession(workspace, idea, auto=fast).run()


def _handle_dev_command(arg: str, workspace: Path) -> None:
    from devmode.session import DevSession

    sub = arg.lower()
    ds = DevSession(workspace)
    if sub.startswith("status"):
        ds.show_status()
    elif sub.startswith("build"):
        from devmode.build import Builder

        Builder(workspace, ds).build()
    elif sub.startswith("revisit "):
        ds.revisit(sub.split(maxsplit=1)[1].strip())
    elif sub.startswith("resolve"):
        ds.resolve()
    else:
        ds.run(resume=True)


def _handle_plan_command(arg: str, session: "AgentSession", workspace: Path) -> None:
    from agent.planner import Planner

    if not arg:
        console.print("[yellow]Usage: /plan <what to build>[/yellow]")
        return
    planner = Planner(workspace, session)
    plan = planner.create_plan(arg)
    if not plan:
        console.print("[yellow]Couldn't produce a task plan — try rephrasing the goal.[/yellow]")
        return
    planner.show(plan)
    if Confirm.ask("Execute this plan now?", default=True):
        planner.run()


def _handle_resume_command(session: "AgentSession", workspace: Path) -> None:
    from agent.planner import Planner

    Planner(workspace, session).run()


def _handle_command(raw: str, session: "AgentSession", workspace: Path) -> bool:
    """Handle a /slash command in the agent REPL. Returns True if the REPL
    should exit (a /exit or /quit was requested), False/None otherwise."""
    from core.config import get_config

    parts = raw[1:].split(maxsplit=1)
    name = (parts[0] if parts else "").lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if name in ("exit", "quit", "q"):
        console.print("[dim]Goodbye.[/dim]")
        return True
    if name in ("help", "h", "?"):
        profile = get_config().devmode_profile()
        console.print(
            Panel(
                r"[bold]/develop \[--fast] <idea>[/bold] Developer Mode: role-driven SDLC design → build (--fast = no back-and-forth)" "\n"
                f"[bold]/dev[/bold]          resume Developer Mode ('/dev status' / '/dev build' / '/dev revisit <phase>' / '/dev resolve') "
                f"[dim]— active profile: {profile}[/dim]\n"
                "[bold]/plan <goal>[/bold]  decompose a goal into tasks and build it\n"
                "[bold]/resume[/bold]       continue an in-progress plan\n"
                "[bold]/init[/bold]         analyze the codebase and write/update AICODER.md\n"
                r"[bold]/model \[name][/bold] pick a model interactively, or switch straight to <name>" "\n"
                r"[bold]/vision <path>[/bold] attach an image (or Ctrl+V to paste one) — a vision "
                "model describes it, your coding model acts on that. Follow-up questions with "
                "no path ('/vision what about the corner?') reuse the last image\n"
                r"[bold]/vision model[/bold] pick the vision-capable model interactively, or "
                "'/vision model <name>'\n"
                r"[bold]/history \[n][/bold]  list past sessions for this workspace, or view one "
                "in detail (prompts, actions, diffs)\n"
                "[bold]/status[/bold]       show workspace, model, provider, and dev-mode profile\n"
                "[bold]/context[/bold]      show conversation size vs. the compaction budget\n"
                "[bold]/compact[/bold]      summarize older turns now (usually automatic)\n"
                "[bold]/permissions[/bold]  view or change shell/file confirmation modes\n"
                "[bold]/review[/bold]       ask the agent to review the current git diff\n"
                "[bold]/tools[/bold]        list the agent's tools\n"
                "[bold]/mcp[/bold]          list connected MCP servers and their tools\n"
                "[bold]/hooks[/bold]        list configured lifecycle hooks\n"
                "[bold]/diff[/bold]         show the git diff of changes so far\n"
                "[bold]/memory[/bold]       show what's remembered about this project\n"
                "[bold]/knowledge[/bold]    RAG: 'learn <topic|URL>', stats, 'clear[ all]'\n"
                r"[bold]/export \[file][/bold] save this conversation to a markdown file" "\n"
                "[bold]/doctor[/bold]       diagnose the model/tool-calling setup (like --selftest)\n"
                "[bold]/bug[/bold]          where and what to report if something's wrong\n"
                "[bold]/clear[/bold]        forget this conversation (keeps saved memory)\n"
                "[bold]/help[/bold]         this help\n"
                "[bold]/exit[/bold]         quit\n\n"
                "[dim]Or just describe a task in plain English.[/dim]",
                title="[cyan]AICoder commands[/cyan]",
                border_style="cyan",
            )
        )
    elif name == "develop":
        _handle_develop_command(arg, workspace)
    elif name == "dev":
        _handle_dev_command(arg, workspace)
    elif name == "plan":
        _handle_plan_command(arg, session, workspace)
    elif name == "resume":
        _handle_resume_command(session, workspace)
    elif name == "init":
        _handle_init_command(session, workspace)
    elif name == "status":
        _handle_status_command(session, workspace)
    elif name == "context":
        _handle_context_command(session)
    elif name == "compact":
        _handle_compact_command(session)
    elif name == "permissions":
        _handle_permissions_command(arg)
    elif name == "review":
        _handle_review_command(session)
    elif name == "mcp":
        _handle_mcp_command(session)
    elif name == "hooks":
        _handle_hooks_command()
    elif name == "export":
        _handle_export_command(arg, session, workspace)
    elif name == "doctor":
        _handle_doctor_command()
    elif name == "bug":
        _handle_bug_command()
    elif name == "tools":
        console.print("[bold]Tools:[/bold] " + ", ".join(session.tools_by_name))
    elif name == "diff":
        from rich.syntax import Syntax
        from tools.shell_tools import run_command

        out, err, code = run_command("git diff HEAD", cwd=workspace, stream_output=False)
        if code != 0:
            # Outside a repo, git treats "HEAD" as a --no-index path arg and
            # dumps its full flag reference (exit 129) instead of a clean
            # "not a repo" message — confirmed live. Special-case it so the
            # user sees one line, not a wall of git usage text.
            detail = (err or out).strip()
            if "not a git repository" in detail.lower():
                console.print("[yellow]Not a git repo.[/yellow]")
            else:
                console.print(f"[yellow]git error: {detail}[/yellow]")
        elif not out.strip():
            console.print("[dim]No changes vs HEAD.[/dim]")
        else:
            console.print(Syntax(out, "diff", theme="monokai", word_wrap=True))
    elif name == "model":
        _handle_model_command(arg, session)
    elif name == "vision":
        _handle_vision_command(arg, session)
    elif name == "history":
        _handle_history_command(arg, session, workspace)
    elif name == "memory":
        from memory.project import ProjectMemory

        rendered = ProjectMemory(workspace).render()
        console.print(
            Panel(
                rendered or "[dim]Nothing remembered yet.[/dim]",
                title="[magenta]Project memory[/magenta]",
                border_style="magenta",
            )
        )
    elif name == "knowledge":
        from core.config import project_id
        from rag.store import KnowledgeBase

        kb = KnowledgeBase.get()
        action, _, rest = arg.partition(" ")
        action = action.lower()
        rest = rest.strip()
        try:
            if action == "learn":
                if not rest:
                    console.print("[yellow]Usage: /knowledge learn <topic or URL>[/yellow]")
                else:
                    _knowledge_learn(rest)
            elif action == "clear" and rest.lower() == "all":
                n = kb.clear_all()
                console.print(f"[green]Cleared the entire knowledge base ({n} chunk(s)).[/green]")
            elif action == "clear":
                n = kb.clear_project(project_id(workspace))
                console.print(
                    f"[green]Cleared {n} document chunk(s) for this project.[/green] "
                    "[dim](global web cache kept — use '/knowledge clear all' to wipe everything)[/dim]"
                )
            else:
                info = kb.info()
                here = kb.count(project_id(workspace))
                console.print(
                    Panel(
                        f"Total chunks: {info['total_chunks']}\n"
                        f"This project:  {here}\n"
                        f"Storage:       {info['storage_path']}\n\n"
                        "[dim]/knowledge learn <topic|URL> — research & cache\n"
                        "/knowledge clear              — clear this project's documents\n"
                        "/knowledge clear all          — wipe everything[/dim]",
                        title="[cyan]Knowledge base[/cyan]",
                        border_style="cyan",
                    )
                )
        except Exception as e:
            console.print(f"[yellow]Knowledge base unavailable: {e}[/yellow]")
    elif name == "clear":
        session.reset()
        console.print("[dim]Conversation cleared (saved project memory kept).[/dim]")
    else:
        console.print(f"[yellow]Unknown command: /{name}. Try /help.[/yellow]")
    return False


def _installed_version() -> str:
    """The installed local-aicoder package version (mirrors cli.py's
    `--version`, duplicated rather than imported to avoid coupling this
    module to cli.py for a three-line lookup)."""
    try:
        from importlib.metadata import version

        return version("local-aicoder")
    except Exception:  # PackageNotFoundError, or running from a bare checkout
        return "0+unknown"


def _last_updated() -> str | None:
    """Date of the last commit to the AICoder source tree itself (not the
    user's project workspace) — there's no formally tagged/dated release
    process today, so the last commit date is the closest honest proxy.
    Returns None (silently omitted from the banner) if git or a repo isn't
    available, e.g. installed from a wheel with no .git directory."""
    root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "log", "-1", "--format=%cd", "--date=short"],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _startup_banner(cfg, model_name: str, workspace: Path, session: "AgentSession") -> Panel:
    """The banner shown at session start — shared by the plain REPL and the TUI
    so the two front-ends never drift out of sync with each other. Two parts:
    identity/version/workspace up top, a rotating tip + tool highlights below."""
    host = socket.gethostname().removesuffix(".local")
    updated = _last_updated()
    version_line = f"AICoder [bold]v{_installed_version()}[/bold]" + (
        f" [dim]· updated {updated}[/dim]" if updated else ""
    )
    tools_block = "\n".join(
        f"  [bold cyan]•[/bold cyan] [bold]{name}[/bold] [dim]— {desc}[/dim]"
        for name, desc in TOOL_HIGHLIGHTS
    )
    tip = random.choice(STARTUP_TIPS)

    body = (
        f"[bold]Welcome, {host}[/bold]\n\n"
        f"{LOGO}\n\n"
        f"{version_line}\n"
        f"[dim]📁 Workspace:[/dim] {workspace}\n"
        f"[dim]🧠 Model:[/dim]     {model_name}  [dim]· Dev Mode:[/dim] {cfg.devmode_profile()} profile\n"
        f"\n[dim]{'─' * 43}[/dim]\n\n"
        f"[bold yellow1]💡 Tip:[/bold yellow1] {tip}\n\n"
        f"[bold]Top tools[/bold]\n"
        f"{tools_block}\n\n"
        f"[dim]'/help' for commands · '/exit' to quit.[/dim]"
    )
    return Panel.fit(body, border_style="cyan")


def run_agent_repl(workspace: Path, continue_session: bool = False) -> None:
    """Interactive agent loop over the given workspace — the plain-terminal
    fallback used when output isn't a real tty (piped/scripted/tests). On a
    real terminal, `aicoder` launches agent.tui.run() instead (see cli.py).
    continue_session (`aicoder --continue`): resume the most recently saved
    conversation for this workspace instead of starting fresh."""
    from core.config import get_config

    from agent.planner import Planner

    cfg = get_config()
    model_name = cfg.model_name
    try:
        session = AgentSession(workspace)
    except RuntimeError as e:
        # A misconfigured/missing provider package — a clean, actionable exit
        # rather than a raw traceback (e.g. openai_compatible without the
        # optional `langchain-openai` package installed).
        console.print(f"[red]⚠ {e}[/red]")
        return
    planner = Planner(workspace, session)

    # Full-screen "alternate screen buffer" mode — the same terminal mechanism
    # vim/less/htop/Claude Code use: swaps to a separate blank screen with no
    # scrollback, and restores the terminal to exactly what was there before on
    # exit (normal or via any exception) — no session trace left behind. A
    # no-op on a non-terminal (piped/redirected output), so scripted usage is
    # unaffected. hide_cursor=False: unlike those apps, input here is normal
    # line-buffered Prompt.ask(), which needs a visible cursor to type against.
    with console.screen(hide_cursor=False):
        console.print(_startup_banner(cfg, model_name, workspace, session))

        if continue_session:
            if session.load_transcript():
                console.print(f"[dim]↺ Resumed the previous conversation "
                              f"({len(session.messages) - 1} message(s)).[/dim]")
            else:
                console.print("[dim]No previous conversation found for this workspace — "
                              "starting fresh.[/dim]")
        if session.instructions:
            console.print("[dim]📄 Loaded project instructions (AICODER.md).[/dim]")
        if planner.has_active_plan():
            console.print("[dim]An in-progress plan exists for this project — type '/resume' to continue it.[/dim]")
        if _has_devmode_session(workspace):
            console.print("[dim]A Developer Mode design exists for this project — type "
                          "'/dev status' to see progress, or '/dev' to resume it.[/dim]")

        while True:
            console.print()
            try:
                user = Prompt.ask(f"[bold green]{workspace.name}[/bold green]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/dim]")
                break

            if not user:
                continue

            start = time.monotonic()
            try:
                if user.startswith("/"):
                    if _handle_command(user, session, workspace):
                        break
                    continue

                session.send(user)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted — back to the prompt.[/yellow]")
            except Exception as e:  # noqa: BLE001 — keep the REPL alive on any failure
                console.print(f"\n[red]⚠ Error: {e}[/red]")
                console.print("[dim]If the model is unreachable, check that LM Studio's local "
                              "server is running (Developer tab → Start Server).[/dim]")
            finally:
                # finally, not after the try block — must still run (and time)
                # a slash command that `continue`s or `break`s out above.
                console.print(f"[dim]⏱ {time.monotonic() - start:.1f}s[/dim]")

        session.mcp.shutdown()
