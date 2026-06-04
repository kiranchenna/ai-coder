"""
devmode/build.py — Developer Mode build hand-off
=================================================
Turns the captured design (docs/dev/*.md + AICODER.md conventions) into code:
propose a folder/file plan → you approve/edit (it's a JSON file you control) →
generate file-by-file, grounded in the spec + conventions, resumable per file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.rule import Rule

from core.model import balanced_json_arrays, get_chat_model
from devmode.phases import PHASES
from devmode.session import DevSession, _stream

console = Console()


def _parse_files(text: str) -> list[dict]:
    for span in balanced_json_arrays(text or ""):
        try:
            data = json.loads(span)
        except Exception:
            continue
        if isinstance(data, list):
            files = [d for d in data if isinstance(d, dict) and d.get("path")]
            if files:
                return files
    return []


class Builder:
    def __init__(self, workspace: Path, session: DevSession | None = None):
        self.workspace = workspace.resolve()
        self.dir = self.workspace / "docs" / "dev"
        self.plan_file = self.dir / "build_plan.json"
        self.session = session or DevSession(self.workspace)

    # ── Context from the design ────────────────────────────────────────────────

    def _spec(self) -> str:
        parts = []
        for p in PHASES:
            if p.target != "doc" or p.kind == "review":
                continue
            f = self.dir / p.filename
            if f.exists():
                parts.append(f"## {p.title}\n" + f.read_text(encoding="utf-8", errors="replace"))
        return "\n\n".join(parts)

    def _conventions(self) -> str:
        f = self.workspace / "AICODER.md"
        return f.read_text(encoding="utf-8", errors="replace") if f.exists() else ""

    # ── Plan ───────────────────────────────────────────────────────────────────

    def _load_plan(self) -> dict | None:
        if self.plan_file.exists():
            try:
                return json.loads(self.plan_file.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _save_plan(self, plan: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.plan_file.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    def generate_plan(self) -> dict | None:
        spec = self._spec()
        if not spec.strip():
            console.print("[yellow]No design found in docs/dev/. Run 'develop <idea>' first.[/yellow]")
            return None
        conv = self._conventions()
        console.print("[dim]📋 Planning the file structure from the design…[/dim]")
        system = (
            "You are a senior engineer. From the design spec and the project's coding "
            "conventions, list ALL files needed for a complete, runnable project, in "
            "dependency order (config/setup, then data models, then business logic, then "
            "API/routes, then UI, then tests). Follow the conventions' folder structure and "
            "naming EXACTLY. Output ONLY a JSON array, no prose:\n"
            '[{"path": "relative/path/file.ext", "purpose": "what this file contains"}]'
        )
        prompt = f"Design spec:\n{spec[:24000]}\n\nCoding conventions:\n{conv[:3000]}\n\nProduce the JSON file list."
        try:
            ai = get_chat_model(precise=True).invoke(
                [SystemMessage(content=system), HumanMessage(content=prompt)]
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Planning failed: {e}[/red]")
            return None
        files = _parse_files(ai.content if isinstance(ai.content, str) else str(ai.content))
        if not files:
            console.print("[yellow]Couldn't produce a file plan — check the design docs and retry.[/yellow]")
            return None
        plan = {
            "idea": self.session.state.get("idea", ""),
            "files": [{"path": f["path"], "purpose": f.get("purpose", ""), "status": "pending"}
                      for f in files],
        }
        self._save_plan(plan)
        return plan

    def _show_plan(self, plan: dict) -> None:
        lines = [f"  {'✔' if f['status'] == 'done' else '○'} {f['path']}" for f in plan["files"]]
        console.print(Panel("\n".join(lines),
                            title=f"[bold magenta]📁 File plan ({len(plan['files'])} files)[/bold magenta]",
                            border_style="magenta"))

    # ── Generation ─────────────────────────────────────────────────────────────

    def build(self) -> None:
        plan = self._load_plan() or self.generate_plan()
        if not plan:
            return
        self._show_plan(plan)
        console.print("[dim]You control the structure — edit docs/dev/build_plan.json (paths, "
                      "order, naming) and run 'dev build' again to use your version.[/dim]")
        done = sum(1 for f in plan["files"] if f["status"] == "done")
        if done:
            console.print(f"[dim]Resuming — {done}/{len(plan['files'])} files already generated.[/dim]")
        if not Confirm.ask("Generate the pending files now?", default=True):
            console.print("[dim]Stopped. Plan saved — edit it and re-run 'dev build'.[/dim]")
            return

        spec = self._spec()
        conv = self._conventions()
        completed = [f["path"] for f in plan["files"] if f["status"] == "done"]

        for entry in plan["files"]:
            if entry["status"] == "done":
                continue
            console.print()
            console.print(Rule(f"[bold cyan]{entry['path']}[/bold cyan]"))
            content = self._generate_file(entry, spec, conv, completed)
            if not content:
                console.print(f"[red]✗ empty generation for {entry['path']} — skipping[/red]")
                continue
            from agent.tools import _apply_write
            msg = _apply_write(self.workspace, entry["path"], content)
            console.print(f"  [green]✔[/green] {msg}")
            entry["status"] = "done"
            completed.append(entry["path"])
            self._save_plan(plan)

        console.print()
        console.print(Panel("✅ Build complete.\n[dim]Verifying… then ask the agent to fix anything, "
                            "or 'dev revisit <phase>' to change a decision.[/dim]",
                            title="[bold green]Developer Mode — build[/bold green]", border_style="green"))
        self._verify()

    def _generate_file(self, entry: dict, spec: str, conv: str, completed: list[str]) -> str:
        recent = ""
        for path in completed[-4:]:
            f = self.workspace / path
            if f.exists():
                recent += f"\n=== {path} ===\n" + f.read_text(encoding="utf-8", errors="replace")[:800]
        system = (
            (conv + "\n\n" if conv else "")
            + "You generate ONE file at a time for a production codebase, following the design "
            "spec and the conventions above EXACTLY (folder structure, naming, style). Write "
            "COMPLETE, production-ready code — no placeholders or TODOs. Output ONLY the raw "
            "file content: no markdown fences, no explanation."
        )
        prompt = (
            f"Project: {self.session.state.get('idea', '')}\n"
            f"Generate this file: {entry['path']}\nPurpose: {entry['purpose']}\n\n"
            f"Design spec (relevant parts):\n{spec[:14000]}\n\n"
            f"Recently generated files (match their imports/style):\n{recent}\n\n"
            f"Output the complete content of {entry['path']} now."
        )
        out = _stream([SystemMessage(content=system), HumanMessage(content=prompt)], precise=True)
        out = self._strip_fences(out)
        from core.config import get_config
        if out and get_config().get("devmode", "build_review", default=True):
            out = self._review_file(entry, out, system, prompt)
        return out

    @staticmethod
    def _strip_fences(text: str) -> str:
        out = re.sub(r"^```[a-zA-Z0-9]*\n?", "", (text or "").strip())
        out = re.sub(r"\n?```$", "", out)
        return out.strip()

    def _review_file(self, entry: dict, draft: str, system: str, prompt: str) -> str:
        """Draft → self-review → fix: catch bugs/placeholders before writing."""
        console.print("  [dim]↻ self-reviewing…[/dim]")
        review_prompt = (
            f"Here is your draft of {entry['path']}:\n\n{draft}\n\n"
            "Review it critically against the design spec and conventions above. Check for: "
            "bugs and logic errors, anything left as a placeholder/TODO/stub, missing imports "
            "or undefined references, mismatches with the file's stated purpose, and convention "
            "violations. If it is already correct and complete, output it UNCHANGED. Otherwise "
            "output the corrected file. Output ONLY the raw file content — no fences, no prose."
        )
        try:
            fixed = _stream(
                [SystemMessage(content=system), HumanMessage(content=prompt),
                 HumanMessage(content=review_prompt)],
                precise=True,
            )
        except Exception:  # noqa: BLE001
            return draft
        fixed = self._strip_fences(fixed)
        # Guard against a degenerate/truncated review nuking a good draft.
        if len(fixed) < len(draft) * 0.5:
            return draft
        return fixed or draft

    def _verify(self) -> None:
        try:
            from core.project import detect_test_command
            from tools.shell_tools import run_command

            detected = detect_test_command(self.workspace)
            if not detected:
                return
            console.print(f"[dim]Running {detected[1]}…[/dim]")
            _o, _e, code = run_command(detected[0], cwd=self.workspace, stream_output=False, timeout=180)
            console.print(f"[{'green' if code == 0 else 'yellow'}]Tests exit code {code}[/]")
        except Exception:  # noqa: BLE001
            pass
