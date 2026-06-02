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
