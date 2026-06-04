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

_EXIT_WORDS = {"exit", "quit", ":q", ":quit"}


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
    Decide whether text-emitted tool-call JSON is an actual call vs. an
    illustrative example inside an explanation. Rejects only the clear case of a
    small example JSON buried in a long prose answer.
    """
    spans = balanced_json_objects(content)
    if not spans:
        return False
    remainder = content
    for span in spans:
        remainder = remainder.replace(span, "", 1)
    remainder = remainder.replace("```json", "").replace("```", "").strip()
    json_len = sum(len(s) for s in spans)
    if json_len < 200 and len(remainder) > 500:
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


class AgentSession:
    """Holds conversation state and drives the tool-calling loop."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.tools = build_tools(workspace)
        # Optional MCP server tools (no-op unless configured).
        from agent.mcp_client import MCPManager

        self.mcp = MCPManager.from_config()
        self.tools += self.mcp.langchain_tools()
        self.tools_by_name = {t.name: t for t in self.tools}
        self.llm = get_chat_model(tools=self.tools)
        from agent.hooks import HookRunner

        self.hooks = HookRunner()
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
        """Process one user message to completion (through any tool calls)."""
        self._compact_history_if_needed()
        self.messages.append(HumanMessage(content=user_input))

        for _ in range(MAX_STEPS):
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
                self.messages.append(AIMessage(
                    content=f"(Requested tools: {', '.join(c['name'] for c in text_calls)})"
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
            return text

        console.print(
            "[yellow]⚠ Reached the step limit for this turn. "
            "Ask me to continue if there's more to do.[/yellow]"
        )
        return ""

    # ── Internals ─────────────────────────────────────────────────────────────

    def _invoke(self) -> AIMessage:
        """
        Stream the model's response token-by-token into a transient live region.

        The preview is erased when the stream ends (transient=True), so tool-call
        JSON doesn't linger; final answers are re-rendered as Markdown by send().
        Returns a clean AIMessage (coerced from the streamed chunk) with any
        tool_calls. Raises on a model/stream failure so callers (the REPL handler,
        the planner) can react rather than mistaking it for an empty answer.
        """
        accumulated = None
        shown = ""
        with Live(console=console, refresh_per_second=10, transient=True) as live:
            live.update(Text("💭 Thinking…", style="dim italic"))
            for chunk in self.llm.stream(self.messages):
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
        if not keep_from:  # None or 0 → nothing safe to drop
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


def _handle_command(raw: str, session: "AgentSession", workspace: Path) -> None:
    """Handle a /slash command in the agent REPL."""
    from core.config import get_config

    parts = raw[1:].split(maxsplit=1)
    name = (parts[0] if parts else "").lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if name in ("help", "h", "?"):
        console.print(
            Panel(
                "[bold]develop [--fast] <idea>[/bold] Developer Mode: role-driven SDLC design → build (--fast = no back-and-forth)\n"
                "[bold]dev[/bold]           resume Developer Mode ('dev status' / 'dev build' / 'dev revisit <phase>' / 'dev resolve')\n"
                "[bold]plan <goal>[/bold]   decompose a goal into tasks and build it\n"
                "[bold]resume[/bold]        continue an in-progress plan\n"
                "[bold]/model [name][/bold] show or switch the model for this session\n"
                "[bold]/tools[/bold]        list the agent's tools\n"
                "[bold]/diff[/bold]         show the git diff of changes so far\n"
                "[bold]/memory[/bold]       show what's remembered about this project\n"
                "[bold]/knowledge[/bold]    RAG: 'learn <topic|URL>', stats, 'clear[ all]'\n"
                "[bold]/clear[/bold]        forget this conversation (keeps saved memory)\n"
                "[bold]/help[/bold]         this help\n"
                "[bold]exit[/bold]          quit\n\n"
                "[dim]Or just describe a task in plain English.[/dim]",
                title="[cyan]AICoder commands[/cyan]",
                border_style="cyan",
            )
        )
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
        cfg = get_config()
        if not arg:
            console.print(f"Current model: [bold]{cfg.model_name}[/bold]")
        else:
            cfg.raw()["model"]["name"] = arg
            session.llm = get_chat_model(tools=session.tools)
            console.print(f"[green]Switched model to [bold]{arg}[/bold] for this session.[/green]")
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


def run_agent_repl(workspace: Path) -> None:
    """Interactive agent loop over the given workspace."""
    from core.config import get_config

    from agent.planner import Planner

    model_name = get_config().model_name
    session = AgentSession(workspace)
    planner = Planner(workspace, session)

    console.print(
        Panel.fit(
            f"[bold magenta]AICoder[/bold magenta] [dim]— local agentic coding assistant[/dim]\n"
            f"[dim]Workspace:[/dim] {workspace}\n"
            f"[dim]Model:[/dim]     {model_name}\n"
            f"[dim]Tools:[/dim]     {', '.join(session.tools_by_name)}\n\n"
            f"[dim]Describe a task in plain English, or 'plan <goal>' for a multi-step build.\n"
            f"'/help' for commands · 'exit' to quit.[/dim]",
            border_style="magenta",
        )
    )

    if session.instructions:
        console.print("[dim]📄 Loaded project instructions (AICODER.md).[/dim]")
    if planner.has_active_plan():
        console.print("[dim]An in-progress plan exists for this project — type 'resume' to continue it.[/dim]")

    while True:
        console.print()
        try:
            user = Prompt.ask(f"[bold green]{workspace.name}[/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user:
            continue

        low = user.lower()
        if low in _EXIT_WORDS:
            console.print("[dim]Goodbye.[/dim]")
            break

        try:
            if user.startswith("/"):
                _handle_command(user, session, workspace)
                continue
            if low == "develop" or low.startswith("develop "):
                idea = user[len("develop"):].strip()
                fast = False
                if idea.split(" ", 1)[0] in ("--fast", "-f"):
                    fast = True
                    idea = idea.split(" ", 1)[1].strip() if " " in idea else ""
                if not idea:
                    console.print("[yellow]Usage: develop [--fast] <your project idea>[/yellow]")
                    continue
                from devmode.session import DevSession
                DevSession(workspace, idea, auto=fast).run()
                continue
            if low == "dev" or low.startswith("dev "):
                from devmode.session import DevSession
                arg = user[3:].strip().lower()
                ds = DevSession(workspace)
                if arg.startswith("status"):
                    ds.show_status()
                elif arg.startswith("build"):
                    from devmode.build import Builder
                    Builder(workspace, ds).build()
                elif arg.startswith("revisit "):
                    ds.revisit(arg.split(maxsplit=1)[1].strip())
                elif arg.startswith("resolve"):
                    ds.resolve()
                else:
                    ds.run(resume=True)
                continue
            if low == "resume":
                planner.run()
                continue
            if low == "plan" or low.startswith("plan "):
                goal = user[5:].strip()
                if not goal:
                    console.print("[yellow]Usage: plan <what to build>[/yellow]")
                    continue
                plan = planner.create_plan(goal)
                if not plan:
                    console.print("[yellow]Couldn't produce a task plan — try rephrasing the goal.[/yellow]")
                    continue
                planner.show(plan)
                if Confirm.ask("Execute this plan now?", default=True):
                    planner.run()
                continue

            session.send(user)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted — back to the prompt.[/yellow]")
        except Exception as e:  # noqa: BLE001 — keep the REPL alive on any failure
            console.print(f"\n[red]⚠ Error: {e}[/red]")
            console.print("[dim]If the model is unreachable, check that Ollama is running.[/dim]")

    session.mcp.shutdown()
