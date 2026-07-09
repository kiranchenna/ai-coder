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
from core.console import SafeConsole
from rich.panel import Panel
from rich.prompt import Confirm
from rich.rule import Rule

from core.model import balanced_json_arrays, get_chat_model
from devmode.phases import PHASES
from devmode.session import DevSession, _stream

console = SafeConsole()


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
        phase_ids = [p.id for p in PHASES if p.target == "doc" and p.kind != "review"]
        system = (
            "You are a senior engineer. From the design spec and the project's coding "
            "conventions, list ALL files needed for a complete, runnable project, in "
            "dependency order (config/setup, then data models, then business logic, then "
            "API/routes, then UI, then tests). Follow the conventions' folder structure and "
            "naming EXACTLY. For each file, list which design phases it implements (from: "
            f"{phase_ids}). Output ONLY a JSON array, no prose:\n"
            '[{"path": "relative/path/file.ext", "purpose": "what this file contains", '
            '"implements": ["data_model", "api"]}]'
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
        valid_ids = set(phase_ids)
        plan = {
            "idea": self.session.state.get("idea", ""),
            "files": [{"path": f["path"], "purpose": f.get("purpose", ""),
                       "implements": [p for p in f.get("implements", []) if p in valid_ids],
                       "status": "pending"}
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

        self._write_manifest(plan)
        console.print()
        console.print(Panel("✅ Build complete.\n[dim]Verifying (compile → tests → fix loop)…[/dim]",
                            title="[bold green]Developer Mode — build[/bold green]", border_style="green"))
        self._verify_and_fix(spec)

    def _write_manifest(self, plan: dict) -> None:
        """Provenance: map each built file to the design phases it implements."""
        manifest = {f["path"]: f.get("implements", [])
                    for f in plan["files"] if f["status"] == "done"}
        (self.dir / "build_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def _api_surface(self, completed: list[str]) -> str:
        """Compact per-file list of symbols already defined across the project, so
        a later file imports real names instead of guessing (the #1 cause of the
        compile/test fix-loop churn when only the last few files are shown)."""
        if not completed:
            return ""
        try:
            from core.code_index import build_symbol_index
            from core.config import get_config
            index = build_symbol_index(self.workspace, ignore_dirs=set(get_config().ignore_dirs))
        except Exception:  # noqa: BLE001
            return ""
        by_file: dict[str, list[str]] = {}
        for name, defs in index.items():
            for d in defs:
                by_file.setdefault(d["file"], []).append(f"{d['kind']} {name}")
        done = set(completed)
        lines = []
        for path in completed:
            syms = by_file.get(path)
            if path in done and syms:
                lines.append(f"{path}: " + ", ".join(sorted(set(syms))[:30]))
        return "\n".join(lines)

    def _generate_file(self, entry: dict, spec: str, conv: str, completed: list[str]) -> str:
        recent = ""
        for path in completed[-4:]:
            f = self.workspace / path
            if f.exists():
                recent += f"\n=== {path} ===\n" + f.read_text(encoding="utf-8", errors="replace")[:800]
        surface = self._api_surface(completed)
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
            + (f"Symbols already defined elsewhere in the project — import these by their real "
               f"names; do NOT redefine them:\n{surface}\n\n" if surface else "")
            + f"Recently generated files (match their imports/style):\n{recent}\n\n"
            f"Output the complete content of {entry['path']} now."
        )
        out = _stream([SystemMessage(content=system), HumanMessage(content=prompt)], precise=True)
        out = self._strip_fences(out)
        from core.config import get_config
        if out and get_config().devmode_lever("build_review"):
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

    # ── Verify → fix loop ──────────────────────────────────────────────────────

    _IGNORE_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "docs",
                    ".pytest_cache", ".mypy_cache", "dist", "build"}
    _MAX_FIX_ROUNDS = 3

    def _project_dir(self):
        """Where the runnable project actually lives — handles a nested subdir."""
        from core.project import detect_test_command
        if detect_test_command(self.workspace):
            return self.workspace
        try:
            subdirs = [d for d in self.workspace.iterdir()
                       if d.is_dir() and d.name not in self._IGNORE_DIRS]
        except OSError:
            return self.workspace
        for d in subdirs:                      # one level down: nested project root
            if detect_test_command(d):
                return d
        return self.workspace

    def _python_files(self, root):
        out = []
        for p in root.rglob("*.py"):
            if not any(part in self._IGNORE_DIRS for part in p.parts):
                out.append(p)
        return out

    def _compile_problems(self, proj) -> str:
        """Cheap cross-file coherence: syntax-check every Python file."""
        import py_compile
        errors = []
        for f in self._python_files(proj):
            try:
                py_compile.compile(str(f), doraise=True)
            except py_compile.PyCompileError as e:
                errors.append(str(e).strip())
            except Exception:  # noqa: BLE001
                pass
        return ("Python syntax errors (fix these first):\n\n" + "\n\n".join(errors[:10])) if errors else ""

    def _run_tests(self, proj):
        """Run the detected test command in the real project dir. → (label, code, output) | None."""
        from core.project import detect_test_command
        from tools.shell_tools import run_command
        detected = detect_test_command(proj)
        if not detected:
            return None
        console.print(f"[dim]Running {detected[1]} in {proj.name}/…[/dim]")
        out, err, code = run_command(detected[0], cwd=proj, stream_output=False, timeout=240)
        return detected[1], code, (out or "") + (("\n" + err) if err else "")

    def _verify_and_fix(self, spec: str) -> None:
        proj = self._project_dir()
        for rnd in range(1, self._MAX_FIX_ROUNDS + 1):
            try:
                problem = self._compile_problems(proj)
                label = "compile"
                if not problem:
                    tr = self._run_tests(proj)
                    if tr is None:
                        console.print("[dim]No test toolchain detected — skipping verification.[/dim]")
                        return
                    label, code, output = tr
                    if code == 0:
                        console.print(f"[bold green]✔ {label} passed.[/bold green]")
                        return
                    # pytest exit code 5 = "no tests collected" — not a failure to
                    # fix; the project just has no tests yet. Don't burn fix rounds.
                    if label == "pytest" and code == 5:
                        console.print("[dim]No tests collected yet — nothing to verify.[/dim]")
                        return
                    problem = f"`{label}` failed (exit {code}):\n{output[-3000:]}"
            except Exception as e:  # noqa: BLE001
                console.print(f"[yellow]Verification could not run: {e}[/yellow]")
                return

            console.print(f"[yellow]⚠ Verification found problems (round {rnd}/{self._MAX_FIX_ROUNDS}).[/yellow]")
            if rnd == self._MAX_FIX_ROUNDS:
                console.print(Panel(problem[:1500], title="[yellow]Unresolved after fix attempts[/yellow]",
                                    border_style="yellow"))
                console.print("[dim]Fix manually, or re-run 'dev build', or 'dev revisit <phase>'.[/dim]")
                return
            if not Confirm.ask("Let the agent fix it?", default=True):
                console.print("[dim]Left as-is. The problem above is saved nowhere — copy it if needed.[/dim]")
                return
            self._agentic_fix(proj, problem)

    def _agentic_fix(self, proj, problem: str) -> None:
        from agent.loop import AgentSession
        rel = "." if proj == self.workspace else str(proj.relative_to(self.workspace))
        task = (
            "The generated project fails its build verification. Fix the code so it compiles "
            "and the tests pass.\n\n"
            f"# The failure\n{problem}\n\n"
            f"The project root is '{rel}'. Work STEP BY STEP, one tool at a time (do not batch "
            "tool calls):\n"
            "1. read_file the file(s) named in the error FULLY before editing.\n"
            "2. Make minimal, focused edit_file changes — copy the exact text you saw.\n"
            "3. Re-run the tests to confirm, and stop once they pass.\n"
            "Do not rewrite files wholesale; fix only what's broken."
        )
        AgentSession(self.workspace).send(task)
