"""Tests for core/ollama_install.py — detecting/offering to install Ollama itself."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ollama_install import install_command, is_ollama_installed


def test_is_ollama_installed_true_when_on_path(monkeypatch):
    import core.ollama_install as mod

    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/local/bin/ollama")
    assert is_ollama_installed() is True


def test_is_ollama_installed_false_when_missing(monkeypatch):
    import core.ollama_install as mod

    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    assert is_ollama_installed() is False


def test_install_command_windows_uses_powershell(monkeypatch):
    import core.ollama_install as mod

    monkeypatch.setattr(mod.sys, "platform", "win32")
    cmd = install_command()
    assert "powershell" in cmd.lower()
    assert "install.ps1" in cmd


def test_install_command_posix_uses_curl(monkeypatch):
    import core.ollama_install as mod

    monkeypatch.setattr(mod.sys, "platform", "darwin")
    cmd = install_command()
    assert cmd == "curl -fsSL https://ollama.com/install.sh | sh"

    monkeypatch.setattr(mod.sys, "platform", "linux")
    assert install_command() == "curl -fsSL https://ollama.com/install.sh | sh"
