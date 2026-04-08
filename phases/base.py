"""
phases/base.py — Abstract base class for all pipeline phases
=============================================================
Every phase in the project planning pipeline inherits from BasePhase,
which provides the common discussion loop, research helpers, file I/O,
and spec-building logic.
"""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

if TYPE_CHECKING:
    from core.pipeline import Pipeline

console = Console()


# ── Custom exceptions for phase flow control ──────────────────────────────────

class PhaseSkipped(Exception):
    """Raised when the user explicitly skips a phase."""

class PhaseRevise(Exception):
    """Raised when the user wants to redo the phase from scratch."""


# ── Base Phase ─────────────────────────────────────────────────────────────────

class BasePhase(ABC):
    """
    Abstract base for all pipeline phases.

    Subclasses must define:
        PHASE_NUM   : int         (1–7)
        PHASE_NAME  : str         ("Idea Refinement")
        ROLE        : str         (key in core/roles.py ROLES dict)
        PHASE_FILE  : str         ("01_idea_refinement.md")

    Subclasses should implement:
        get_research_queries(idea) -> list[str]
        build_initial_prompt(research, context) -> str
        build_summary_prompt(transcript) -> str
    """

    PHASE_NUM:  int = 0
    PHASE_NAME: str = ""
    ROLE:       str = "fullstack_developer"
    PHASE_FILE: str = "00_unnamed.md"

    def __init__(self, pipeline: "Pipeline"):
        self.pipeline    = pipeline
        self.workspace   = pipeline.workspace
        self.process_dir = pipeline.process_dir
        self.spec_file   = pipeline.spec_file
        self.project     = pipeline.project_name
        self.kb          = pipeline.knowledge_base
        self.idea        = pipeline.idea

    # ─── Research ─────────────────────────────────────────────────────────────

    def get_research_queries(self) -> list[str]:
        """Override to return a list of web-search queries for this phase."""
        return []

    def do_research(self) -> str:
        """Run all phase-specific search queries and return combined text."""
        queries = self.get_research_queries()
        if not queries:
            return ""

        console.print(f"  [dim]Researching {len(queries)} topic(s)…[/dim]")
        texts: list[str] = []
        for q in queries:
            text = self.kb.fetch_and_store(
                query      = q,
                ttl_hours  = 12.0,
                project    = self.project,
                phase      = self.PHASE_NAME,
            )
            texts.append(f"### Query: {q}\n\n{text}")

        return "\n\n---\n\n".join(texts)

    # ─── Context from previous phases ─────────────────────────────────────────

    def load_previous_context(self) -> str:
        """
        Read all previously completed phase files and return their content.
        This gives later phases full awareness of earlier decisions.
        """
        lines: list[str] = []
        for f in sorted(self.process_dir.glob("0*.md")):
            if f.name == self.PHASE_FILE:
                continue   # Skip current phase's file
            try:
                content = f.read_text(encoding="utf-8")
                lines.append(f"## {f.stem.replace('_', ' ').title()}\n\n{content}")
            except Exception:
                pass
        return "\n\n---\n\n".join(lines) if lines else ""

    # ─── System message builder ───────────────────────────────────────────────

    def _build_system(self, research: str, context: str) -> str:
        """Assemble the phase system prompt from role + research + prior context."""
        from core.roles import get_role_prompt, get_role_label

        label      = get_role_label(self.ROLE)
        role_prompt = get_role_prompt(self.ROLE)

        parts = [
            f"# Role: {label}",
            "",
            role_prompt,
            "",
            f"# Current Task: {self.PHASE_NAME}",
            f"Project: {self.project}",
            f"Idea: {self.idea}",
        ]

        if context:
            parts += ["", "# Previous Phase Decisions", "", context]

        if research:
            parts += ["", "# Web Research (latest from internet)", "", research]

        parts += [
            "",
            "---",
            "Keep responses focused and actionable."
            " When the user seems satisfied, proactively suggest typing 'done' to continue.",
        ]

        return "\n".join(parts)

    # ─── Abstract prompts ─────────────────────────────────────────────────────

    @abstractmethod
    def build_initial_prompt(self, research: str, context: str) -> str:
        """Return the first message sent to the AI to kick off this phase."""
        ...

    def build_summary_prompt(self, transcript: str) -> str:
        """Return the prompt used to generate the structured phase summary."""
        return (
            f"Based on the following conversation for the '{self.PHASE_NAME}' phase, "
            f"produce a clean, structured Markdown summary of ALL decisions made.\n"
            f"Use headings, bullet points, and tables where appropriate.\n"
            f"This will be stored as the permanent record of this phase.\n\n"
            f"Conversation:\n\n{transcript}"
        )

    # ─── Discussion loop ──────────────────────────────────────────────────────

    def run_discussion(self, system: str, initial_prompt: str) -> tuple[str, list[BaseMessage]]:
        """
        Run the interactive phase discussion.

        1. AI responds to initial_prompt (streamed)
        2. User types messages back and forth
        3. User types 'done' to proceed, 'skip' to skip phase, 'revise' to restart

        Returns (full transcript as markdown, message history list).
        """
        from core.streaming import stream_response
        from core.roles import get_role_label

        messages: list[BaseMessage] = [
            SystemMessage(content=system),
            HumanMessage(content=initial_prompt),
        ]

        label = get_role_label(self.ROLE)
        console.print()
        console.print(Rule(f"[bold magenta]{label} — {self.PHASE_NAME}[/bold magenta]"))

        # Stream first AI response
        ai_response = stream_response(messages, label=label, precise=False)
        messages.append(AIMessage(content=ai_response))

        transcript_parts = [
            f"**Initial Analysis:**\n\n{ai_response}",
        ]

        # ── Conversation loop ──────────────────────────────────────────────────
        while True:
            console.print()
            console.print(
                "[dim]  Reply, or type: [bold]done[/bold] · [bold]skip[/bold] · [bold]revise[/bold][/dim]"
            )
            try:
                user_input = Prompt.ask(f"[bold yellow]{self.project}[/bold yellow]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Phase interrupted.[/dim]")
                raise PhaseSkipped()

            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in ("done", "d", "continue", "next"):
                break
            elif cmd in ("skip", "s"):
                raise PhaseSkipped()
            elif cmd in ("revise", "r", "restart"):
                raise PhaseRevise()

            # Regular user message — append and get AI response
            messages.append(HumanMessage(content=user_input))
            transcript_parts.append(f"**You:** {user_input}")

            response = stream_response(messages, label=label, precise=False)
            messages.append(AIMessage(content=response))
            transcript_parts.append(f"**{label}:** {response}")

        return "\n\n---\n\n".join(transcript_parts), messages

    # ─── Summary generation ───────────────────────────────────────────────────

    def generate_summary(self, transcript: str, messages: list[BaseMessage]) -> str:
        """
        Call the AI to produce a structured markdown summary of the phase.
        Uses precise/low-temperature mode for consistent output.
        """
        from core.streaming import stream_response
        from core.roles import get_role_label

        console.print()
        console.print(Rule("[dim]📝 Generating phase summary…[/dim]"))

        summary_messages = list(messages) + [
            HumanMessage(content=self.build_summary_prompt(transcript))
        ]
        label = get_role_label(self.ROLE)
        summary = stream_response(summary_messages, label=f"📝 {label} Summary", precise=True)
        return summary

    # ─── File I/O ─────────────────────────────────────────────────────────────

    def save_phase_output(self, transcript: str, summary: str) -> Path:
        """Save transcript + summary to process/<name>/<phase_file>."""
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        content = (
            f"# {self.PHASE_NAME}\n\n"
            f"_Generated: {ts}_\n\n"
            f"---\n\n"
            f"## Summary\n\n{summary}\n\n"
            f"---\n\n"
            f"## Full Discussion\n\n{transcript}\n"
        )
        path = self.process_dir / self.PHASE_FILE
        path.write_text(content, encoding="utf-8")
        return path

    def append_to_spec(self, summary: str) -> None:
        """Append this phase's summary to the cumulative spec file."""
        self.spec_file.parent.mkdir(parents=True, exist_ok=True)
        sep = "\n\n---\n\n"

        if not self.spec_file.exists():
            header = (
                f"# Project Spec: {self.project}\n\n"
                f"_Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n"
                f"**Idea:** {self.idea}\n"
            )
            self.spec_file.write_text(header, encoding="utf-8")

        existing = self.spec_file.read_text(encoding="utf-8")
        addition = f"\n\n---\n\n## Phase {self.PHASE_NUM}: {self.PHASE_NAME}\n\n{summary}"
        self.spec_file.write_text(existing + addition, encoding="utf-8")

    # ─── Phase confirmation ───────────────────────────────────────────────────

    def confirm_phase(self) -> str:
        """
        Ask the user what to do after showing the summary.
        Returns: 'done' | 'revise' | 'skip'
        """
        console.print()
        choice = Prompt.ask(
            f"[bold]Phase {self.PHASE_NUM} complete — what next?[/bold]",
            choices=["done", "revise", "skip"],
            default="done",
        )
        return choice

    # ─── Main entry point ─────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute this phase. Handles research, discussion, summary, and saving.
        Returns a dict with 'summary', 'transcript', 'status'.
        """
        from core.roles import get_role_label

        console.print()
        console.print(
            Panel.fit(
                f"[bold magenta]Phase {self.PHASE_NUM}:[/bold magenta] {self.PHASE_NAME}\n"
                f"[dim]Role: {get_role_label(self.ROLE)}[/dim]",
                border_style="magenta",
            )
        )

        while True:  # retry loop on 'revise'
            try:
                # Step 1: Research
                research = self.do_research()

                # Step 2: Load prior context
                context = self.load_previous_context()

                # Step 3: Build system + initial prompt
                system         = self._build_system(research, context)
                initial_prompt = self.build_initial_prompt(research, context)

                # Step 4: Run discussion
                transcript, messages = self.run_discussion(system, initial_prompt)

                # Step 5: Generate summary
                summary = self.generate_summary(transcript, messages)

                # Step 6: Show summary and confirm
                console.print()
                console.print(Panel(summary, title=f"[bold green]✔ {self.PHASE_NAME} Summary[/bold green]", border_style="green"))

                choice = self.confirm_phase()
                if choice == "revise":
                    console.print("[dim]Restarting phase…[/dim]")
                    continue
                elif choice == "skip":
                    return {"status": "skipped"}

                # Step 7: Save
                path = self.save_phase_output(transcript, summary)
                self.append_to_spec(summary)
                console.print(f"  [green]✔[/green] Saved → [bold]{path.name}[/bold]")

                return {"status": "done", "summary": summary, "transcript": transcript}

            except PhaseSkipped:
                console.print(f"[dim]Skipped Phase {self.PHASE_NUM}: {self.PHASE_NAME}[/dim]")
                return {"status": "skipped"}

            except PhaseRevise:
                console.print("[dim]Restarting phase…[/dim]")
                continue
