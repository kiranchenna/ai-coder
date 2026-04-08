"""
phases/codegen.py — Phase 7: Code Generation
=============================================
Role: Appropriate developer (chosen from architecture decision)
Goal: Generate all project files one-by-one based on the complete spec,
      using latest official documentation fetched from the web.
      Each file is written immediately so if generation stops it can resume.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.syntax import Syntax

from phases.base import BasePhase, PhaseSkipped

if TYPE_CHECKING:
    from core.pipeline import Pipeline

console = Console()


# ── Role selector ─────────────────────────────────────────────────────────────

def _pick_role(spec_summary: str) -> str:
    """Infer the correct developer role from the architecture summary."""
    s = spec_summary.lower()
    if "fastapi" in s or "django" in s or "flask" in s:
        return "python_developer"
    if "next.js" in s or "nextjs" in s or "react" in s:
        return "react_developer"
    if "vue" in s or "nuxt" in s:
        return "vue_developer"
    if "rust" in s or "axum" in s or "actix" in s:
        return "rust_developer"
    if "spring" in s or "java" in s or "kotlin" in s:
        return "java_developer"
    if "android" in s or "jetpack" in s:
        return "android_developer"
    if "swift" in s or "swiftui" in s or "ios" in s:
        return "ios_developer"
    return "fullstack_developer"


class CodegenPhase(BasePhase):
    PHASE_NUM  = 7
    PHASE_NAME = "Code Generation"
    ROLE       = "fullstack_developer"   # overridden dynamically
    PHASE_FILE = "07_codegen.md"

    def __init__(self, pipeline: "Pipeline"):
        super().__init__(pipeline)
        self.output_dir = pipeline.workspace / "output" / pipeline.project_name
        self.state_file = self.process_dir / "codegen_state.json"

    # ── Research: fetch official quick-start docs ──────────────────────────────

    def get_research_queries(self) -> list[str]:
        context = self.load_previous_context()
        queries = [
            f"{self.idea} project setup guide official documentation 2025",
            f"{self.idea} full stack tutorial latest version 2025",
        ]
        # Detect tech from previous phases and add specific doc queries
        for tech in ["fastapi", "next.js", "react", "vue", "django", "spring boot",
                     "postgresql", "prisma", "sqlalchemy", "tailwind", "shadcn"]:
            if tech in context.lower():
                queries.append(f"{tech} official documentation quickstart latest")
        return queries[:5]  # Cap to avoid too many requests

    # ── File plan generation ───────────────────────────────────────────────────

    def _generate_file_plan(self, spec: str, research: str) -> list[dict]:
        """
        Ask AI to produce a JSON list of files to generate, in dependency order.
        Returns list of {path, description, dependencies} dicts.
        """
        from core.streaming import stream_response

        system = (
            "You are a senior developer planning a codebase. "
            "Based on the project spec provided, list ALL files to generate in dependency order "
            "(config and setup files first, then models/types, then business logic, then routes, "
            "then frontend pages).\n"
            "Output ONLY a JSON array, no other text:\n"
            '[{"path": "relative/path/file.ext", "description": "what this file does"}]\n'
            "Include every file needed for the project to run. Be comprehensive."
        )

        prompt = (
            f"Project spec:\n{spec}\n\n"
            f"Latest docs reference:\n{research[:2000]}\n\n"
            f"List ALL files to generate as JSON array."
        )

        from rich.console import Console as C
        C().print("[dim]📋 Planning file structure…[/dim]")
        resp = stream_response(
            [SystemMessage(content=system), HumanMessage(content=prompt)],
            label="📋 File Plan",
            precise=True,
            show_label=False,
        )

        # Parse JSON from response
        match = re.search(r"\[.*\]", resp, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        console.print("[yellow]Could not parse file plan JSON — using fallback.[/yellow]")
        return [{"path": "README.md", "description": "Project readme"}]

    # ── Single file generation ─────────────────────────────────────────────────

    def _generate_file(
        self,
        file_info:    dict,
        spec:         str,
        completed:    list[str],
        research:     str,
    ) -> str | None:
        """Generate a single file and return its content."""
        from core.streaming import stream_response
        from core.roles import get_role_prompt, get_role_label

        path = file_info["path"]
        desc = file_info.get("description", "")

        # Build context: what has been generated so far
        context_files = ""
        for done_path in completed[-5:]:   # last 5 files for context
            done_file = self.output_dir / done_path
            if done_file.exists():
                try:
                    content = done_file.read_text(encoding="utf-8")[:800]
                    context_files += f"\n===FILE: {done_path}===\n{content}\n===END===\n"
                except Exception:
                    pass

        system = (
            get_role_prompt(self.ROLE)
            + "\n\nYou are generating ONE file at a time for a production codebase.\n"
            + "Write COMPLETE, production-ready code — no placeholders, no TODOs.\n"
            + "Use the LATEST versions of all libraries as specified in the project spec.\n"
            + "Follow official documentation patterns exactly.\n"
            + "Output ONLY the file content — no explanation, no markdown fences."
        )

        prompt = (
            f"Project: {self.project} — {self.idea}\n\n"
            f"Generate this file: **{path}**\n"
            f"Purpose: {desc}\n\n"
            f"Full project spec:\n{spec[:4000]}\n\n"
            f"Recently generated files (for imports/consistency):\n{context_files}\n\n"
            f"Latest documentation:\n{research[:1500]}\n\n"
            f"Output the complete content of `{path}` now. No markdown, just the raw file."
        )

        label = get_role_label(self.ROLE)
        content = stream_response(
            [SystemMessage(content=system), HumanMessage(content=prompt)],
            label=f"✍ {Path(path).name}",
            precise=True,
        )

        # Strip any accidental markdown fences
        content = re.sub(r"^```[a-z]*\n?", "", content, flags=re.MULTILINE)
        content = re.sub(r"\n?```$", "", content, flags=re.MULTILINE)

        return content.strip() if content.strip() else None

    # ── State management ───────────────────────────────────────────────────────

    def _load_gen_state(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except Exception:
                pass
        return {"completed": [], "failed": [], "file_plan": []}

    def _save_gen_state(self, state: dict) -> None:
        self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # ── Main run ───────────────────────────────────────────────────────────────

    def run(self) -> dict:
        from core.roles import get_role_label

        # Dynamically choose role
        prev_context = self.load_previous_context()
        self.ROLE = _pick_role(prev_context)

        console.print()
        console.print(
            Panel.fit(
                f"[bold magenta]Phase 7:[/bold magenta] Code Generation\n"
                f"[dim]Developer role: {get_role_label(self.ROLE)}[/dim]\n"
                f"[dim]Output: output/{self.project}/[/dim]",
                border_style="magenta",
            )
        )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Research: fetch official docs
        console.print("[dim]Fetching latest documentation…[/dim]")
        research = self.do_research()

        # Load spec (all previous phases)
        spec = prev_context[:6000]

        # Load or build file plan
        gen_state = self._load_gen_state()
        file_plan = gen_state.get("file_plan", [])

        if not file_plan:
            file_plan = self._generate_file_plan(spec, research)
            gen_state["file_plan"] = file_plan
            self._save_gen_state(gen_state)

        completed = gen_state.get("completed", [])
        total     = len(file_plan)

        console.print()
        console.print(
            Panel(
                "\n".join(f"  {'✔' if f['path'] in completed else '○'} {f['path']}"
                          for f in file_plan),
                title=f"[bold]📁 Files to generate ({total} total)[/bold]",
                border_style="dim",
            )
        )

        if completed:
            console.print(f"[dim]  Resuming — {len(completed)}/{total} already done.[/dim]")

        # Generate file-by-file
        for i, file_info in enumerate(file_plan):
            path = file_info["path"]

            if path in completed:
                console.print(f"  [dim]↷ Skipping (already generated): {path}[/dim]")
                continue

            console.print()
            console.print(Rule(f"[bold cyan]({i+1}/{total}) {path}[/bold cyan]"))

            content = self._generate_file(file_info, spec, completed, research)
            if not content:
                gen_state["failed"].append(path)
                self._save_gen_state(gen_state)
                console.print(f"  [red]✗ Failed to generate {path}[/red]")
                continue

            # Write file
            out_path = self.output_dir / path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")

            completed.append(path)
            gen_state["completed"] = completed
            self._save_gen_state(gen_state)
            console.print(f"  [green]✔[/green] Written: {path}")

        # Final summary
        failed = gen_state.get("failed", [])
        console.print()
        console.print(
            Panel(
                f"✅ Generated: {len(completed)}/{total} files\n"
                + (f"❌ Failed: {', '.join(failed)}" if failed else "")
                + f"\n\n📁 Output: {self.output_dir}",
                title="[bold green]Code Generation Complete[/bold green]",
                border_style="green",
            )
        )

        # Save codegen log
        log = (
            f"# Code Generation Log: {self.project}\n\n"
            f"## Generated Files\n"
            + "\n".join(f"- {p}" for p in completed)
            + (f"\n\n## Failed Files\n" + "\n".join(f"- {p}" for p in failed) if failed else "")
            + f"\n\n## Output Directory\n`{self.output_dir}`\n"
        )
        (self.process_dir / self.PHASE_FILE).write_text(log, encoding="utf-8")
        self.append_to_spec(log)

        return {"status": "done", "files": completed, "failed": failed}
