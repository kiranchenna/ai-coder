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

import json
import re
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

from core.model import get_chat_model
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


def _balanced_json_objects(text: str) -> list[str]:
    """Extract top-level {...} substrings via brace matching (string-aware)."""
    objects: list[str] = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    objects.append(text[start : i + 1])
    return objects


def _extract_text_tool_calls(content: str) -> list[dict]:
    """
    Recover tool calls that a model emitted as JSON *text* instead of via native
    tool calling (common with local models, e.g. qwen2.5-coder over Ollama).
    Looks for {"name": ..., "arguments"/"args"/"parameters": {...}} objects.
    """
    calls: list[dict] = []
    for candidate in _balanced_json_objects(content or ""):
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        args = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
        if isinstance(args, dict):
            calls.append({"name": obj["name"], "args": args, "id": ""})
    return calls


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
                c for c in _extract_text_tool_calls(ai.content or "")
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
            f"Type 'exit' to quit.[/dim]",
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
