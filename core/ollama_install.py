"""
core/ollama_install.py — Detect and (optionally) install Ollama itself
========================================================================
Distinct from "Ollama is installed but not running" (handled by
core.model.is_model_pulled / cli._check_ollama): this checks whether the
`ollama` binary is on PATH at all, and offers Ollama's own official install
command if not — shown in full before running, never silent.

The commands below are Ollama's own documented one-liners (github.com/ollama/
ollama README, verified directly rather than guessed): macOS and Linux share
the same script (it detects the OS itself); Windows uses the official
PowerShell equivalent. No Homebrew/winget package is officially documented, so
we don't invent one.
"""

from __future__ import annotations

import shutil
import sys

DOWNLOAD_PAGE = "https://ollama.com/download"

_POSIX_INSTALL_CMD = "curl -fsSL https://ollama.com/install.sh | sh"
_WINDOWS_INSTALL_CMD = 'powershell -Command "irm https://ollama.com/install.ps1 | iex"'


def is_ollama_installed() -> bool:
    """Whether the `ollama` binary is present on PATH at all (independent of
    whether the server is currently running — see core.model.is_model_pulled
    for that check)."""
    return shutil.which("ollama") is not None


def install_command() -> str:
    """Ollama's official install one-liner for the current platform."""
    return _WINDOWS_INSTALL_CMD if sys.platform == "win32" else _POSIX_INSTALL_CMD
