"""
core/pipeline.py — Project Planning Pipeline Orchestrator
==========================================================
Manages the full 7-phase pipeline with:
  - Phase ordering and optional skipping
  - Persistent state (resume after exit)
  - Per-project process/ and specs/ directories
  - Knowledge base integration
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule

from core.knowledge import KnowledgeBase

console = Console()

# Phase number → human name mapping (for state display)
PHASE_META = {
    1: ("Idea Refinement",      "01_idea_refinement.md"),
    2: ("Competitor Analysis",  "02_competitor_analysis.md"),
    3: ("Architecture Design",  "03_architecture_design.md"),
    4: ("Data Models",          "04_data_models.md"),
    5: ("API Design",           "05_api_design.md"),
    6: ("Frontend Planning",    "06_frontend_planning.md"),
    7: ("Code Generation",      "07_codegen.md"),
}


class Pipeline:
    """
    Orchestrates the full project planning pipeline.

    Usage:
        pipeline = Pipeline(workspace, "my-task-manager", "a task manager for developers")
        pipeline.run()
    """

    def __init__(self, workspace: Path, project_name: str, idea: str):
        self.workspace    = workspace.resolve()
        self.project_name = project_name
        self.idea         = idea

        # Directories
        self.process_dir = workspace / "process" / project_name
        self.spec_file   = workspace / "specs" / f"{project_name}.md"
        self.state_file  = self.process_dir / "state.json"

        # Create dirs
        self.process_dir.mkdir(parents=True, exist_ok=True)
        (workspace / "specs").mkdir(exist_ok=True)
        (workspace / "output").mkdir(exist_ok=True)

        # Shared knowledge base
        self.knowledge_base = KnowledgeBase.get()

    # ── State management ───────────────────────────────────────────────────────

    def _default_state(self) -> dict:
        return {
            "project_name":  self.project_name,
            "idea":          self.idea,
            "created_at":    datetime.datetime.now().isoformat(),
            "updated_at":    datetime.datetime.now().isoformat(),
            "current_phase": 1,
            "phases": {
                str(n): {"status": "pending", "name": PHASE_META[n][0]}
                for n in range(1, 8)
            },
        }

    def load_state(self) -> dict:
        """Load saved state, or return a fresh default."""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return self._default_state()

    def save_state(self, state: dict) -> None:
        state["updated_at"] = datetime.datetime.now().isoformat()
        self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _status_icon(self, status: str) -> str:
        return {"done": "✅", "skipped": "⏭", "in_progress": "🔄", "pending": "○"}.get(status, "○")

    # ── Phase loader ───────────────────────────────────────────────────────────

    def _load_phase(self, phase_num: int):
        """Lazy-import and instantiate the phase class for phase_num."""
        from phases.idea         import IdeaPhase
        from phases.competitors  import CompetitorPhase
        from phases.architecture import ArchitecturePhase
        from phases.models       import ModelsPhase
        from phases.api_design   import APIPhase
        from phases.frontend     import FrontendPhase
        from phases.codegen      import CodegenPhase

        PHASE_MAP = {
            1: IdeaPhase,
            2: CompetitorPhase,
            3: ArchitecturePhase,
            4: ModelsPhase,
            5: APIPhase,
            6: FrontendPhase,
            7: CodegenPhase,
        }
        cls = PHASE_MAP[phase_num]
        return cls(self)

    # ── Display helpers ────────────────────────────────────────────────────────

    def _show_progress(self, state: dict) -> None:
        """Show a visual pipeline progress board."""
        lines = []
        for n in range(1, 8):
            p       = state["phases"][str(n)]
            icon    = self._status_icon(p["status"])
            current = " ◀" if state["current_phase"] == n else ""
            lines.append(f"  {icon}  Phase {n}: {p['name']}{current}")

        console.print()
        console.print(
            Panel(
                "\n".join(lines),
                title=f"[bold magenta]🚀 {self.project_name}[/bold magenta]",
                subtitle=f"[dim]{self.idea}[/dim]",
                border_style="magenta",
            )
        )
        console.print()

    # ── Ask to skip ───────────────────────────────────────────────────────────

    def _ask_skip(self, phase_num: int) -> bool:
        """Ask user if they want to skip a phase. Returns True if skipping."""
        name = PHASE_META[phase_num][0]
        console.print()
        return not Confirm.ask(
            f"[bold]Run Phase {phase_num}: {name}?[/bold]",
            default=True,
        )

    # ── Main run ───────────────────────────────────────────────────────────────

    def run(self, start_phase: int = 1) -> None:
        """
        Execute the pipeline from start_phase onwards.
        Supports skipping individual phases and resuming after interruption.
        """
        state = self.load_state()
        self._show_progress(state)

        for phase_num in range(start_phase, 8):
            phase_state = state["phases"][str(phase_num)]

            # Already done — skip unless user wants to redo
            if phase_state["status"] == "done":
                console.print(
                    f"[dim]Phase {phase_num} ({phase_state['name']}) already complete.[/dim]"
                )
                if phase_num > start_phase:
                    redo = Confirm.ask(f"  Redo Phase {phase_num}?", default=False)
                    if not redo:
                        continue

            # Ask to skip (only for phases after the first)
            if phase_num > start_phase:
                skip = self._ask_skip(phase_num)
                if skip:
                    phase_state["status"] = "skipped"
                    state["current_phase"] = phase_num + 1
                    self.save_state(state)
                    console.print(f"[dim]  ⏭ Skipped Phase {phase_num}[/dim]")
                    continue

            # Mark in-progress
            phase_state["status"]   = "in_progress"
            state["current_phase"]  = phase_num
            self.save_state(state)

            # Run
            try:
                phase  = self._load_phase(phase_num)
                result = phase.run()

                phase_state["status"] = result.get("status", "done")
                if result.get("summary"):
                    phase_state["summary"] = result["summary"][:300]

            except KeyboardInterrupt:
                console.print("\n[yellow]Pipeline paused. Run /project resume to continue.[/yellow]")
                self.save_state(state)
                return

            self.save_state(state)
            self._show_progress(state)

        # Done!
        console.print()
        console.print(
            Panel(
                f"🎉 All phases complete!\n\n"
                f"📄 Spec:    specs/{self.project_name}.md\n"
                f"📁 Process: process/{self.project_name}/\n"
                f"💻 Output:  output/{self.project_name}/",
                title="[bold green]Project Planning Complete[/bold green]",
                border_style="green",
            )
        )


# ── Resume helper ──────────────────────────────────────────────────────────────

def find_resumable_projects(workspace: Path) -> list[dict]:
    """Find all in-progress pipeline projects in this workspace."""
    process_dir = workspace / "process"
    if not process_dir.exists():
        return []

    projects = []
    for state_file in process_dir.glob("*/state.json"):
        try:
            state = json.loads(state_file.read_text())
            # Check if any phase is in-progress or pending
            statuses = [p["status"] for p in state["phases"].values()]
            if "in_progress" in statuses or "pending" in statuses:
                done  = sum(1 for s in statuses if s == "done")
                total = len(statuses)
                projects.append({
                    "name":          state["project_name"],
                    "idea":          state.get("idea", ""),
                    "current_phase": state.get("current_phase", 1),
                    "progress":      f"{done}/{total} phases done",
                    "updated_at":    state.get("updated_at", "")[:16],
                })
        except Exception:
            pass
    return projects
