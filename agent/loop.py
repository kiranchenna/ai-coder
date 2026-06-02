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
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.status import Status

from core.model import extract_text_tool_calls, get_chat_model
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


class AgentSession:
    """Holds conversation state and drives the tool-calling loop."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.tools = build_tools(workspace)
        self.tools_by_name = {t.name: t for t in self.tools}
        self.llm = get_chat_model(tools=self.tools)
        self.messages = [
            SystemMessage(
                content=system_prompt(
                    workspace,
                    list(self.tools_by_name),
                    _repo_overview(workspace),
                    _project_memory(workspace),
                )
            )
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    def send(self, user_input: str) -> str:
        """Process one user message to completion (through any tool calls)."""
        self.messages.append(HumanMessage(content=user_input))

        for _ in range(MAX_STEPS):
            ai = self._invoke()
            self.messages.append(ai)

            # 1) Native tool calls (preferred path).
            if ai.tool_calls:
                for call in ai.tool_calls:
                    self._render_call(call)
                    result = self._exec(call)
                    self.messages.append(
                        ToolMessage(content=result, tool_call_id=call.get("id", ""))
                    )
                continue

            # 2) Fallback: some local models emit tool calls as JSON text in the
            #    content instead of via native tool calling. Recover and run them.
            text_calls = [
                c for c in extract_text_tool_calls(ai.content or "")
                if c["name"] in self.tools_by_name
            ]
            if text_calls:
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
            text = (ai.content or "").strip()
            console.print()
            console.print(Markdown(text) if text else "[dim](no further response)[/dim]")
            return text

        console.print(
            "[yellow]⚠ Reached the step limit for this turn. "
            "Ask me to continue if there's more to do.[/yellow]"
        )
        return ""

    # ── Internals ─────────────────────────────────────────────────────────────

    def _invoke(self) -> AIMessage:
        status = Status("[dim italic]💭 Thinking…[/dim italic]", console=console)
        status.start()
        try:
            return self.llm.invoke(self.messages)
        finally:
            status.stop()

    def _render_call(self, call: dict) -> None:
        name = call.get("name", "?")
        args = call.get("args", {}) or {}
        preview = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
        console.print(f"[cyan]→ {name}[/cyan]([dim]{preview}[/dim])")

    def reset(self) -> None:
        """Forget the conversation, keeping the system prompt (and saved memory)."""
        self.messages = self.messages[:1]

    def _exec(self, call: dict) -> str:
        name = call.get("name", "")
        args = call.get("args", {}) or {}
        tool = self.tools_by_name.get(name)
        if tool is None:
            return (
                f"ERROR: unknown tool '{name}'. "
                f"Available tools: {', '.join(self.tools_by_name)}."
            )
        try:
            # StructuredTool validates args against the schema and raises on
            # bad input — surfaced back to the model so it can self-correct.
            return str(tool.invoke(args))
        except Exception as e:  # noqa: BLE001 — report any tool failure to the model
            return f"ERROR running {name}: {e}"


# ── REPL ───────────────────────────────────────────────────────────────────────

def _handle_command(raw: str, session: "AgentSession", workspace: Path) -> None:
    """Handle a /slash command in the agent REPL."""
    from core.config import get_config

    parts = raw[1:].split(maxsplit=1)
    name = (parts[0] if parts else "").lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if name in ("help", "h", "?"):
        console.print(
            Panel(
                "[bold]plan <goal>[/bold]   decompose a goal into tasks and build it\n"
                "[bold]resume[/bold]        continue an in-progress plan\n"
                "[bold]/model [name][/bold] show or switch the model for this session\n"
                "[bold]/tools[/bold]        list the agent's tools\n"
                "[bold]/memory[/bold]       show what's remembered about this project\n"
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
