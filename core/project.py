"""
core/project.py — Project-type detection
=========================================
Detects how to test (and build) the project in the workspace, so the agent can
verify its changes without being told the toolchain. Inspects well-known marker
files and returns a command + a short label.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _exists(root: Path, name: str) -> bool:
    return (root / name).exists()


def _python_executable(root: Path) -> str:
    """
    Pick the interpreter to run pytest with. Prefer the project's own virtualenv
    (so its dependencies are available); otherwise fall back to the interpreter
    running aicoder. Avoids relying on a bare `python` on PATH (absent on macOS).
    """
    for candidate in (
        ".venv/bin/python", "venv/bin/python",
        ".venv/Scripts/python.exe", "venv/Scripts/python.exe",
    ):
        p = root / candidate
        if p.exists():
            return f'"{p}"'
    return f'"{sys.executable}"'


def detect_test_command(workspace: Path) -> tuple[str, str] | None:
    """
    Return (command, label) for running this project's tests, or None if no
    known toolchain is detected. Detection is ordered by specificity.
    """
    root = workspace

    # ── Node / JS / TS ─────────────────────────────────────────────────────────
    if _exists(root, "package.json"):
        try:
            pkg = json.loads((root / "package.json").read_text(encoding="utf-8", errors="replace"))
            if "test" in pkg.get("scripts", {}):
                pm = "npm"
                if _exists(root, "pnpm-lock.yaml"):
                    pm = "pnpm"
                elif _exists(root, "yarn.lock"):
                    pm = "yarn"
                elif _exists(root, "bun.lockb"):
                    pm = "bun"
                cmd = f"{pm} test" if pm != "npm" else "npm test"
                return (cmd, f"{pm} test")
        except Exception:
            pass

    # ── Rust ───────────────────────────────────────────────────────────────────
    if _exists(root, "Cargo.toml"):
        return ("cargo test", "cargo test")

    # ── Go ─────────────────────────────────────────────────────────────────────
    if _exists(root, "go.mod"):
        return ("go test ./...", "go test")

    # ── Python ─────────────────────────────────────────────────────────────────
    if (
        _exists(root, "pytest.ini")
        or _exists(root, "tox.ini")
        or _exists(root, "setup.cfg")
        or _exists(root, "pyproject.toml")
        or (root / "tests").is_dir()
        or (root / "test").is_dir()
    ):
        return (f"{_python_executable(root)} -m pytest -q", "pytest")

    # ── Make ───────────────────────────────────────────────────────────────────
    if _exists(root, "Makefile"):
        try:
            mk = (root / "Makefile").read_text(encoding="utf-8", errors="replace")
            if re.search(r"^test:", mk, re.MULTILINE):
                return ("make test", "make test")
        except Exception:
            pass

    # ── JVM ────────────────────────────────────────────────────────────────────
    if _exists(root, "pom.xml"):
        return ("mvn -q test", "maven")
    if _exists(root, "build.gradle") or _exists(root, "build.gradle.kts"):
        return ("./gradlew test" if _exists(root, "gradlew") else "gradle test", "gradle")

    return None


def _node_pm(root: Path) -> str:
    if _exists(root, "pnpm-lock.yaml"):
        return "pnpm"
    if _exists(root, "yarn.lock"):
        return "yarn"
    if _exists(root, "bun.lockb"):
        return "bun"
    return "npm"


def detect_lint_commands(workspace: Path) -> list[tuple[str, str]]:
    """
    Return [(command, label)] for the project's linters / type checkers, or an
    empty list. Python tools are config-gated (only run if configured) to avoid
    noisy default-config output; others key off marker files.
    """
    root = workspace
    cmds: list[tuple[str, str]] = []

    # ── Node / TypeScript ──────────────────────────────────────────────────────
    if _exists(root, "package.json"):
        pm = _node_pm(root)
        try:
            pkg = json.loads((root / "package.json").read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pkg = {}
        if "lint" in pkg.get("scripts", {}):
            cmds.append((f"{pm} run lint", f"{pm} run lint"))
        if _exists(root, "tsconfig.json"):
            cmds.append(("npx --no-install tsc --noEmit", "tsc"))

    # ── Python (only if a tool is configured) ──────────────────────────────────
    pyproject = ""
    if _exists(root, "pyproject.toml"):
        try:
            pyproject = (root / "pyproject.toml").read_text(encoding="utf-8", errors="replace")
        except Exception:
            pyproject = ""
    py = _python_executable(root)
    if "[tool.ruff" in pyproject or _exists(root, "ruff.toml") or _exists(root, ".ruff.toml"):
        cmds.append((f"{py} -m ruff check .", "ruff"))
    if "[tool.mypy" in pyproject or _exists(root, "mypy.ini") or _exists(root, ".mypy.ini"):
        cmds.append((f"{py} -m mypy .", "mypy"))
    if _exists(root, ".flake8"):
        cmds.append((f"{py} -m flake8", "flake8"))

    # ── Rust / Go ──────────────────────────────────────────────────────────────
    if _exists(root, "Cargo.toml"):
        cmds.append(("cargo clippy -q", "clippy"))
    if _exists(root, "go.mod"):
        cmds.append(("go vet ./...", "go vet"))

    return cmds
