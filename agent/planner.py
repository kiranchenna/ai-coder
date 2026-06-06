"""
agent/planner.py — Decompose a large goal into a resumable task list
====================================================================
For big asks ("build the service from this PRD"), the planner asks the model
for an ordered task list, persists it per-project, and drives the agent through
each task one at a time — verifying as it goes and saving state after every
step so a build can resume after a quit. This is the useful part of the old
linear pipeline, now agentic and non-linear.

State: ~/.aicoder/memory/<project_id>/plan.json  (resumable)
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from core.config import MEMORY_DIR, project_id
from core.model import balanced_json_arrays, get_chat_model

console = Console()

PLANNER_SYSTEM = (
    "You are a senior engineer planning the work to accomplish a goal in a software "
    "project. Break the goal into an ordered list of concrete, independently-executable "
    "tasks. Each task should be a small, verifiable step (e.g. create a file, add an "
    "endpoint, write a test). Order them by dependency: setup/config first, then data "
    "models, then business logic, then routes/UI, then tests.\n"
    "Output ONLY a JSON array, no prose:\n"
    '[{"title": "short title", "description": "specific, actionable detail"}]'
)

_STATUS_ICON = {"done": "✅", "failed": "❌", "pending": "○", "in_progress": "🔄"}


def _parse_tasks(text: str) -> list[dict]:
    """Extract a JSON array of task objects from model output, tolerantly.

    Scans for balanced top-level [...] spans (string-aware) and returns the first
    that parses to a list of dicts — so prose or stray brackets around the array
    (common with small local models) don't break parsing.
    """
    for span in balanced_json_arrays(text or ""):
        try:
            data = json.loads(span)
        except Exception:
            continue
        if isinstance(data, list):
            dicts = [t for t in data if isinstance(t, dict)]
            if dicts:
                return dicts
    return []


class Planner:
    """Creates, persists, and executes a resumable task plan against the agent."""

    def __init__(self, workspace: Path, session):
        self.workspace = workspace.resolve()
        self.session = session
        self.state_file = MEMORY_DIR / project_id(self.workspace) / "plan.json"

    # ── State ──────────────────────────────────────────────────────────────────

    def load(self) -> dict | None:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def save(self, plan: dict) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def has_active_plan(self) -> bool:
        plan = self.load()
        if not plan:
            return False
        return any(
            t.get("status", "pending") in ("pending", "in_progress")
            for t in plan.get("tasks", [])
        )

    # ── Planning ───────────────────────────────────────────────────────────────

    def _ground(self, goal: str) -> str:
        """Pull relevant ingested-document context to ground the plan."""
        try:
            from rag.store import KnowledgeBase

            hits = KnowledgeBase.get().search(
                goal, n=4, max_distance=0.8, project=project_id(self.workspace)
            )
            if hits:
                return "\n\n".join(h["content"] for h in hits)[:3000]
        except Exception:
            pass
        return ""

    def create_plan(self, goal: str) -> dict | None:
        context = self._ground(goal)
        prompt = f"Goal: {goal}\n"
        if context:
            prompt += f"\nRelevant project/document context:\n{context}\n"
        prompt += "\nProduce the ordered JSON task list."

        console.print("[dim]📋 Planning tasks…[/dim]")
        try:
            ai = get_chat_model(precise=True).invoke(
                [SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=prompt)]
            )
        except Exception as e:
            console.print(f"[red]Planning failed: {e}[/red]")
            return None

        tasks = _parse_tasks(ai.content)
        if not tasks:
            return None
        plan = {
            "goal": goal,
            "tasks": [
                {
                    "id": i + 1,
                    "title": t.get("title", f"Task {i + 1}"),
                    "description": t.get("description", ""),
                    "status": "pending",
                }
                for i, t in enumerate(tasks)
            ],
        }
        self.save(plan)
        return plan

    # ── Display ────────────────────────────────────────────────────────────────

    def show(self, plan: dict) -> None:
        lines = [
            f"  {_STATUS_ICON.get(t.get('status', 'pending'), '○')} {t.get('id', '?')}. "
            f"{t.get('title', 'task')}"
            for t in plan.get("tasks", [])
        ]
        console.print(
            Panel(
                "\n".join(lines),
                title=f"[bold magenta]📋 Plan: {plan['goal'][:60]}[/bold magenta]",
                border_style="magenta",
            )
        )

    # ── Execution ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        plan = self.load()
        if not plan:
            console.print("[yellow]No saved plan to run. Use: plan <goal>[/yellow]")
            return

        self.show(plan)
        tasks = plan.get("tasks", [])
        done_count = sum(1 for t in tasks if t.get("status") == "done")
        if done_count:
            console.print(f"[dim]Resuming — {done_count}/{len(tasks)} tasks already done.[/dim]")

        for task in tasks:
            if task.get("status") == "done":
                continue
            task.setdefault("status", "pending")

            console.print()
            console.print(Rule(f"[bold cyan]Task {task['id']}: {task['title']}[/bold cyan]"))
            task["status"] = "in_progress"
            self.save(plan)

            try:
                self.session.send(
                    f"We are executing a planned build, one task at a time.\n"
                    f"Task {task['id']}: {task['title']}\n{task['description']}\n\n"
                    f"Complete ONLY this task now, using your tools. If it involves code, "
                    f"verify it with run_tests when appropriate. Keep changes focused."
                )
            except KeyboardInterrupt:
                task["status"] = "pending"
                self.save(plan)
                console.print("\n[yellow]Paused. Type 'resume' to continue later.[/yellow]")
                return

            # The agent hit the step cap before finishing — don't mark the task done
            # (that would silently skip unfinished work). Leave it pending to resume.
            if not getattr(self.session, "last_turn_complete", True):
                task["status"] = "pending"
                self.save(plan)
                console.print(
                    f"[yellow]⚠ Task {task['id']} hit the step limit before completing — "
                    "left as pending.[/yellow]"
                )
                console.print("[dim]Type 'resume' to give it more steps, or refine the task.[/dim]")
                return

            task["status"] = "done"
            self.save(plan)

            try:
                choice = Prompt.ask(
                    "[bold]Task complete.[/bold] Continue to next? (y/n)", default="y"
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]Paused. Type 'resume' to continue later.[/yellow]")
                return
            if choice in ("n", "no", "stop", "q"):
                console.print("[dim]Stopped. Type 'resume' to continue later.[/dim]")
                return

        console.print()
        console.print(Panel("🎉 All planned tasks complete.", border_style="green"))
