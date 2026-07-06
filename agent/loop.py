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

import threading
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text

from core.model import balanced_json_objects, extract_text_tool_calls, get_chat_model
from agent.prompts import system_prompt
from agent.tools import build_tools

console = Console()

# Safety cap on tool-call iterations within a single user turn.
MAX_STEPS = 12

# The brand mark echoes the app icon's palette (cyan circuit-bracket, amber
# neural mark, on dark) and its bracket / ">_" terminal-cursor motifs — a
# terminal can't render the actual bitmap logo, so this is the in-terminal
# equivalent (see assets/icon.png for the real logo, used in the README).
BRAND = ("[bold cyan]⟨[/bold cyan][bold yellow1]❯[/bold yellow1][dim yellow1]_[/dim yellow1]"
         "[bold cyan]⟩[/bold cyan]  [bold cyan]AI[/bold cyan][bold yellow1]Coder[/bold yellow1]")

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
        self.workspace = workspace
        self._interrupt = threading.Event()
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
        self._compact_history_if_needed()
        self.messages.append(HumanMessage(content=user_input))

        try:
            return self._run_steps()
        except _TurnInterrupted:
            console.print("\n[yellow]Interrupted — back to the prompt.[/yellow]")
            return ""

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

        if cfg.model_provider == "ollama":
            from core.model import is_model_pulled

            if is_model_pulled(cfg.model_base_url, vision_model_name) is False:
                pulled = _pull_via_ollama(
                    vision_model_name,
                    confirm_prompt=f"Vision model [bold]{vision_model_name}[/bold] isn't "
                    "pulled yet — pull it now? This can take a while depending on your "
                    "connection.",
                )
                if not pulled:
                    raise RuntimeError(f"{vision_model_name} isn't pulled — can't look at "
                                        "the image.")

        content: list[dict] = [{
            "type": "text",
            "text": user_text or "Describe this image in detail, especially anything that "
                    "looks like a bug, error, or visual issue.",
        }]
        for path in image_paths:
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode()
            content.append({"type": "image_url", "image_url": f"data:{mime};base64,{encoded}"})

        vision_llm = get_chat_model(model=vision_model_name)
        response = vision_llm.invoke([HumanMessage(content=content)])
        return response.content if isinstance(response.content, str) else str(response.content)

    def send_with_images(self, user_input: str, image_paths: list[Path]) -> str:
        """The two-model handoff: describe the attached images with the
        vision model, fold that description into a normal text turn, then run
        it through the ordinary agentic loop (tool calling, editing) with the
        regular coding model — completely unchanged from a plain send()."""
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
        name = call.get("name", "?")
        args = call.get("args", {}) or {}
        preview = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
        console.print(f"[cyan]→ {name}[/cyan]([dim]{preview}[/dim])")

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
            return (
                f"ERROR: unknown tool '{name}'. "
                f"Available tools: {', '.join(self.tools_by_name)}."
            )

        # PreToolUse hooks can block the call.
        block = self.hooks.pre_tool_use(name, args, self.workspace)
        if block is not None:
            console.print(f"[yellow]⛔ {name} blocked by hook: {block}[/yellow]")
            return f"BLOCKED by a PreToolUse hook: {block}"

        try:
            # StructuredTool validates args against the schema and raises on
            # bad input — surfaced back to the model so it can self-correct.
            result = str(tool.invoke(args))
        except Exception as e:  # noqa: BLE001 — report any tool failure to the model
            result = f"ERROR running {name}: {e}"

        # PostToolUse hooks (auto-format, notify, …) may add a note.
        note = self.hooks.post_tool_use(name, args, result, self.workspace)
        if note:
            result += f"\n[hook] {note}"
        return result


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

    # The "is it actually pulled?" check only applies to Ollama — an
    # openai_compatible endpoint has no equivalent local-pull concept.
    if cfg.model_provider == "ollama":
        from core.model import is_model_pulled

        if is_model_pulled(cfg.model_base_url, name) is False:
            console.print(f"[yellow]⚠ '{name}' may not be pulled yet.[/yellow] "
                          f"[dim]Run: ollama pull {name}[/dim]")


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

    if cfg.model_provider == "ollama":
        from core.model import is_model_pulled

        if is_model_pulled(cfg.model_base_url, name) is False:
            console.print(f"[yellow]⚠ '{name}' may not be pulled yet.[/yellow] "
                          f"[dim]Run: ollama pull {name}[/dim]")


def _confirm_and_pull(tag: str, size_bytes: int, session: "AgentSession") -> None:
    """Pull a not-yet-installed catalog model (with confirmation — it's a real
    download), then switch to it and persist as the new default. `tag` always
    comes from our own hardcoded RECOMMENDED_MODELS, never raw user input, so
    it's safe to interpolate into the shell command directly."""
    from rich.prompt import Confirm
    from tools.shell_tools import run_command

    if not Confirm.ask(
        f"Pull [bold]{tag}[/bold] (~{_format_size(size_bytes)})? This can take a while "
        "depending on your connection.",
        default=True,
    ):
        console.print("[dim]No change made.[/dim]")
        return

    console.print(f"[dim]⬇ Pulling {tag} — hang tight, this can take a while for larger "
                  f"models…[/dim]")
    _out, err, code = run_command(f"ollama pull {tag}", timeout=7200)
    if code != 0:
        console.print(f"[red]✗ Failed to pull {tag} (exit {code}).[/red]"
                      + (f"\n{err.strip()[-500:]}" if err else ""))
        return
    console.print(f"[green]✓ Pulled {tag}.[/green]")
    _switch_model(tag, session)


def _pull_via_ollama(tag: str, *, confirm_prompt: str | None = None) -> bool:
    """Confirm, then pull an arbitrary Ollama tag — raw user/config input, not
    our curated RECOMMENDED_MODELS catalog, so this uses subprocess.run with
    an argv list rather than run_command's shell=True (interpolating
    untrusted input into a shell string would be an injection risk). Returns
    whether the pull succeeded (False on decline, failure, or 'ollama' not
    being on PATH — a message is always printed either way)."""
    import subprocess

    from rich.prompt import Confirm

    if not Confirm.ask(
        confirm_prompt or f"Pull [bold]{tag}[/bold] from Ollama? This can take a while "
        "depending on your connection.",
        default=True,
    ):
        console.print("[dim]No change made.[/dim]")
        return False

    console.print(f"[dim]⬇ Pulling {tag} — hang tight, this can take a while for larger "
                  f"models…[/dim]")
    try:
        result = subprocess.run(
            ["ollama", "pull", tag], capture_output=True, text=True, timeout=7200,
        )
    except FileNotFoundError:
        console.print("[red]✗ 'ollama' isn't on your PATH.[/red]")
        return False
    except subprocess.TimeoutExpired:
        console.print(f"[red]✗ Pulling {tag} timed out.[/red]")
        return False
    if result.returncode != 0:
        console.print(f"[red]✗ Failed to pull {tag} (exit {result.returncode}).[/red]"
                      + (f"\n{result.stderr.strip()[-500:]}" if result.stderr else ""))
        return False
    console.print(f"[green]✓ Pulled {tag}.[/green]")
    return True


def _pull_arbitrary_model(tag: str, session: "AgentSession") -> None:
    """Pull a model tag the user typed themselves — any Ollama model, not just
    our curated recommendations (see 'Other…' in the /model picker) — then
    switch to it and persist as the new default."""
    if _pull_via_ollama(tag):
        _switch_model(tag, session)


def _handle_custom_model_entry(session: "AgentSession") -> None:
    """'Other…' in the /model picker — any Ollama model not in our curated
    recommendations (see ollama.com/library for the full catalog)."""
    from core.config import get_config

    name = Prompt.ask("Model name (e.g. llama3.2:1b — see ollama.com/library)",
                       default="").strip()
    if not name:
        console.print("[dim]No change made.[/dim]")
        return

    cfg = get_config()
    if cfg.model_provider == "ollama":
        from core.model import is_model_pulled

        if is_model_pulled(cfg.model_base_url, name) is False:
            _pull_arbitrary_model(name, session)
            return
    _switch_model(name, session)


def _build_model_menu(current_name: str, installed: list[dict], catalog: list) -> list[dict]:
    """Merge installed models with not-yet-installed catalog recommendations
    into one ordered, section-labeled list for a model picker. Shared by
    `/model` (catalog=RECOMMENDED_MODELS) and `/vision model`
    (catalog=VISION_MODELS) — the two-model-family split is a real
    architectural fact of Ollama's local ecosystem, not just a naming choice,
    so these stay two distinct catalogs rather than one filtered list."""
    from core.model_catalog import TIER_LABELS, TIER_ORDER

    installed_names = {m["name"] for m in installed}
    entries = [
        {"tag": m["name"], "size_bytes": m["size"], "installed": True,
         "current": m["name"] == current_name, "note": None, "section": "Installed"}
        for m in installed
    ]
    for tier in TIER_ORDER:
        for spec in catalog:
            if spec.tier != tier or spec.tag in installed_names:
                continue  # already installed — don't recommend it a second time
            entries.append({
                "tag": spec.tag, "size_bytes": int(spec.size_gb * 1024 ** 3),
                "installed": False, "current": False, "note": spec.note,
                "section": f"Recommended — {TIER_LABELS[tier]}",
            })
    return entries


def _model_menu_entries(cfg, installed: list[dict]) -> list[dict]:
    """`/model`'s entries: installed + RECOMMENDED_MODELS, marked against the
    active coding model."""
    from core.model_catalog import RECOMMENDED_MODELS

    return _build_model_menu(cfg.model_name, installed, RECOMMENDED_MODELS)


def _model_menu_label(e: dict) -> str:
    """One-line label for a single model-menu entry, used by the TUI's
    arrow-key OptionList (see ask_choice in agent/tui.py). The plain-text
    Rich panel below has its own column-aligned formatting and doesn't share
    this — kept separate to avoid touching its already-tested rendering."""
    check = "[green]✓[/green] " if e["current"] else "  "
    pulled = " [dim]· not pulled[/dim]" if not e["installed"] else ""
    note = f" [dim]— {e['note']}[/dim]" if e["note"] else ""
    return f"{check}{e['tag']} [dim]({_format_size(e['size_bytes'])})[/dim]{pulled}{note}"


def _render_model_menu(entries: list[dict], title: str = "[cyan]Available models (via Ollama)[/cyan]") -> None:
    lines: list[str] = []
    last_section = None
    for i, e in enumerate(entries, 1):
        if e["section"] != last_section:
            if last_section is not None:
                lines.append("")
            lines.append(f"[bold]{e['section']}[/bold]")
            last_section = e["section"]
        marker = " [dim](current)[/dim]" if e["current"] else ""
        pulled_tag = "" if e["installed"] else " [dim]· not pulled[/dim]"
        note = f"  [dim]{e['note']}[/dim]" if e["note"] else ""
        lines.append(f"  [cyan]{i:>2}[/cyan]  {e['tag']:<24} {_format_size(e['size_bytes']):>8}"
                     f"{marker}{pulled_tag}{note}")
    console.print(Panel("\n".join(lines), title=title, border_style="cyan"))


def _run_model_picker(picker_title: str, entries: list[dict], current_name: str) -> int | None:
    """The shared arrow-key (TUI) or numbered (plain-REPL) model picker over
    `entries` (see _build_model_menu). Returns the chosen index, len(entries)
    for "Other…", or None if cancelled/kept as-is. Shared by /model and
    /vision model."""
    other_label = "Other… (type any Ollama model name)"

    try:
        from agent.tui import ask_choice, is_tui_active
        tui_active = is_tui_active()
    except ImportError:  # pragma: no cover — textual is a base dependency
        tui_active = False

    if tui_active:
        # The richer, arrow-key-navigable version of the same picker: grouped
        # by section (Installed / Recommended — tier, like the plain-REPL
        # panel below), opening with the current model already highlighted.
        current_idx = next((i for i, e in enumerate(entries) if e["current"]), 0)
        idx = ask_choice(
            picker_title,
            [_model_menu_label(e) for e in entries] + [other_label],
            groups=[e["section"] for e in entries] + ["Other"],
            initial_index=current_idx,
        )
        if idx is None:
            console.print(f"[dim]Kept current model: {current_name}[/dim]")
            return None
        return idx

    _render_model_menu(entries)
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


def _handle_non_ollama_model_command(cfg) -> None:
    """`/model` (no arg) for the openai_compatible provider — there's no
    Ollama-style discovery API for an arbitrary server/hosted endpoint, so
    just show what's active and how to change it."""
    console.print(Panel(
        f"[dim]Provider:[/dim] {cfg.model_provider}\n"
        f"[dim]Model:[/dim]    {cfg.model_name}\n"
        f"[dim]Endpoint:[/dim] {cfg.model_base_url}\n\n"
        "[dim]Use [/dim][bold]/model <name>[/bold][dim] to switch to a different model id on "
        "this same endpoint, or edit [/dim][bold]~/.aicoder/config.yaml[/bold][dim] to point at "
        "a different server/API entirely (model.base_url, model.api_key).[/dim]",
        title="[cyan]Current model[/cyan]",
        border_style="cyan",
    ))


def _handle_model_command(arg: str, session: "AgentSession") -> None:
    """`/model` — for the default Ollama provider: pick a model interactively
    (installed models plus curated, not-yet-installed recommendations by
    hardware/preference tier), or switch straight to `/model <name>`. Mirrors
    the Claude Code `/model` picker: select one and it becomes your default
    for new sessions, not just this one. For the openai_compatible provider
    (a custom server or hosted API), there's no discovery API to pick from —
    see `_handle_non_ollama_model_command`."""
    from core.config import get_config

    cfg = get_config()
    if arg:
        _switch_model(arg, session)
        return

    if cfg.model_provider != "ollama":
        _handle_non_ollama_model_command(cfg)
        return

    from core.model import list_ollama_models

    try:
        installed = list_ollama_models(cfg.model_base_url)
    except Exception:
        console.print(
            "[yellow]⚠ Couldn't reach Ollama to list models.[/yellow]\n"
            f"[dim]Make sure it's running (`ollama serve`), or switch directly: "
            f"/model <name>. Current: {cfg.model_name}[/dim]"
        )
        return

    # Our curated RECOMMENDED_MODELS is a hand-picked subset — Ollama has no
    # API to browse its full library, so /vision model's picker (and this
    # one) always ends with an "Other…" escape hatch for anything not in it.
    entries = _model_menu_entries(cfg, installed)
    idx = _run_model_picker("Select a model", entries, cfg.model_name)
    if idx is None:
        return
    if idx == len(entries):
        _handle_custom_model_entry(session)
        return

    entry = entries[idx]
    if entry["installed"]:
        _switch_model(entry["tag"], session)
    else:
        _confirm_and_pull(entry["tag"], entry["size_bytes"], session)


# ── /vision — attach an image by file path (the two-model vision handoff) ──────
# The TUI's Ctrl+V clipboard paste (agent/tui.py) is the richer version of this
# same flow — both end up calling AgentSession.send_with_images.

def _handle_vision_model_command(arg: str = "") -> None:
    """`/vision model` — pick the vision-capable model interactively, mirroring
    `/model`'s picker exactly (same shared entries builder and picker
    dispatch) but sourced from VISION_MODELS and persisting to vision.model,
    not model.name — the coding model that's actually driving the session is
    untouched. `/vision model <name>` switches straight to <name>, same as
    `/model <name>` does for the coding model."""
    if arg:
        _switch_vision_model(arg)
        return

    from core.config import get_config

    cfg = get_config()
    if cfg.model_provider != "ollama":
        console.print(
            "[yellow]The vision model picker only supports discovering models via "
            "Ollama.[/yellow] [dim]Set vision.model directly in config.yaml for a custom "
            f"endpoint. Current: {cfg.vision_model or '(not set)'}[/dim]"
        )
        return

    from core.model import is_model_pulled, list_ollama_models

    try:
        installed = list_ollama_models(cfg.model_base_url)
    except Exception:
        console.print(
            "[yellow]⚠ Couldn't reach Ollama to list models.[/yellow]\n"
            "[dim]Make sure it's running (`ollama serve`), or set vision.model directly in "
            f"config.yaml. Current: {cfg.vision_model or '(not set)'}[/dim]"
        )
        return

    from core.model_catalog import VISION_MODELS

    entries = _build_model_menu(cfg.vision_model, installed, VISION_MODELS)
    idx = _run_model_picker(
        "Select a vision model", entries, cfg.vision_model or "(not set)",
    )
    if idx is None:
        return

    if idx == len(entries):
        name = Prompt.ask(
            "Vision model name (e.g. qwen2.5vl:7b — see ollama.com/library)", default="",
        ).strip()
        if not name:
            console.print("[dim]No change made.[/dim]")
            return
        if is_model_pulled(cfg.model_base_url, name) is False and not _pull_via_ollama(name):
            return
        _switch_vision_model(name)
        return

    entry = entries[idx]
    if entry["installed"]:
        _switch_vision_model(entry["tag"])
        return
    if _pull_via_ollama(
        entry["tag"],
        confirm_prompt=f"Pull [bold]{entry['tag']}[/bold] "
        f"(~{_format_size(entry['size_bytes'])})? This can take a while depending on your "
        "connection.",
    ):
        _switch_vision_model(entry["tag"])


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
        console.print(f"[red]⚠ Image not found: {path_str}[/red]")
        return

    try:
        session.send_with_images(user_text, [path])
    except RuntimeError as e:
        console.print(f"[red]⚠ {e}[/red]")


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
        "  • Your OS, Ollama version, and the model in use\n"
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
                "model describes it, your coding model acts on that\n"
                r"[bold]/vision model[/bold] pick the vision-capable model interactively, or "
                "'/vision model <name>'\n"
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
            console.print(f"[yellow]Not a git repo, or git error: {(err or out).strip()}[/yellow]")
        elif not out.strip():
            console.print("[dim]No changes vs HEAD.[/dim]")
        else:
            console.print(Syntax(out, "diff", theme="monokai", word_wrap=True))
    elif name == "model":
        _handle_model_command(arg, session)
    elif name == "vision":
        _handle_vision_command(arg, session)
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


def _startup_banner(cfg, model_name: str, workspace: Path, session: "AgentSession") -> Panel:
    """The banner shown at session start — shared by the plain REPL and the TUI
    so the two front-ends never drift out of sync with each other."""
    return Panel.fit(
        f"{BRAND} [dim]— local agentic coding assistant[/dim]\n"
        f"[dim]Workspace:[/dim] {workspace}\n"
        f"[dim]Model:[/dim]     {model_name}\n"
        f"[dim]Dev Mode:[/dim]  {cfg.devmode_profile()} profile\n"
        f"[dim]Tools:[/dim]     {', '.join(session.tools_by_name)}\n\n"
        f"[dim]Describe a task in plain English, or '/plan <goal>' for a multi-step build.\n"
        f"'/help' for commands · '/exit' to quit.[/dim]",
        border_style="cyan",
    )


def run_agent_repl(workspace: Path) -> None:
    """Interactive agent loop over the given workspace — the plain-terminal
    fallback used when output isn't a real tty (piped/scripted/tests). On a
    real terminal, `aicoder` launches agent.tui.run() instead (see cli.py)."""
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

        if session.instructions:
            console.print("[dim]📄 Loaded project instructions (AICODER.md).[/dim]")
        if planner.has_active_plan():
            console.print("[dim]An in-progress plan exists for this project — type '/resume' to continue it.[/dim]")

        while True:
            console.print()
            try:
                user = Prompt.ask(f"[bold green]{workspace.name}[/bold green]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/dim]")
                break

            if not user:
                continue

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
                console.print("[dim]If the model is unreachable, check that Ollama is running.[/dim]")

        session.mcp.shutdown()
