"""
devmode/resync.py — Propagate a changed design decision into the code
=====================================================================
When a phase decision is revisited and changed, run an agentic task that finds
the affected code and updates it to match, then verifies. How completely it
resyncs depends on the model; the pipeline (diff → apply → verify) is the same.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

console = Console()

_MAX = 4000


def resync(workspace: Path, phase_title: str, old_decision: str, new_decision: str) -> None:
    diff = "\n".join(difflib.unified_diff(
        old_decision.splitlines(), new_decision.splitlines(),
        fromfile="old decision", tofile="new decision", lineterm="",
    ))
    console.print()
    console.print(Rule(f"[bold cyan]Auto-resync — applying the {phase_title} change[/bold cyan]"))

    task = (
        f"A design decision for '{phase_title}' has CHANGED, and the existing code must be "
        f"updated to match it.\n\n"
        f"# What changed (unified diff of the decision)\n{diff[:_MAX]}\n\n"
        f"# The new {phase_title} decision (authoritative)\n{new_decision[:_MAX]}\n\n"
        f"Work through this STEP BY STEP, one tool at a time (do not batch tool calls):\n"
        f"1. Find the affected code with find_symbol or search_code.\n"
        f"2. read_file the affected file(s) FULLY first.\n"
        f"3. Then make minimal, focused edit_file changes to match the new decision — copy the "
        f"exact text you saw when reading.\n"
        f"4. Update any tests/usages that depend on the changed behaviour.\n"
        f"5. Finally verify with run_tests and run_checks, and summarize what you changed."
    )

    from agent.loop import AgentSession

    AgentSession(workspace).send(task)
