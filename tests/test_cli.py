"""Tests for cli.py's _offer_install_ollama — the confirm-then-install flow."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from cli import _offer_install_ollama, _run_ollama_preflight


def test_skips_entirely_when_already_installed(monkeypatch, capsys):
    import core.ollama_install as install_mod

    monkeypatch.setattr(install_mod, "is_ollama_installed", lambda: True)
    _offer_install_ollama()
    assert capsys.readouterr().out == ""    # no prompt, no output at all


def test_declining_falls_back_to_manual_link(monkeypatch, capsys):
    import core.ollama_install as install_mod
    from rich.prompt import Confirm

    monkeypatch.setattr(install_mod, "is_ollama_installed", lambda: False)
    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: False))
    called = []
    monkeypatch.setattr("tools.shell_tools.run_command",
                        lambda *a, **k: called.append(1) or ("", "", 0))
    _offer_install_ollama()
    assert not called                                    # never attempted install
    assert "ollama.com/download" in capsys.readouterr().out


def test_eof_in_non_interactive_context_falls_back_gracefully(monkeypatch, capsys):
    # A non-interactive/no-stdin context (e.g. piped input) must not crash.
    import core.ollama_install as install_mod
    from rich.prompt import Confirm

    monkeypatch.setattr(install_mod, "is_ollama_installed", lambda: False)

    def raise_eof(*a, **k):
        raise EOFError

    monkeypatch.setattr(Confirm, "ask", staticmethod(raise_eof))
    _offer_install_ollama()   # must not raise
    assert "Skipped" in capsys.readouterr().out


def test_confirmed_and_successful_install(monkeypatch, capsys):
    import core.ollama_install as install_mod
    from rich.prompt import Confirm

    # Simulate: not installed -> confirm yes -> install succeeds -> now installed.
    state = {"installed": False}
    monkeypatch.setattr(install_mod, "is_ollama_installed", lambda: state["installed"])
    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: True))

    def fake_run(cmd, timeout=None):
        state["installed"] = True
        return ("done", "", 0)

    monkeypatch.setattr("tools.shell_tools.run_command", fake_run)
    _offer_install_ollama()
    assert "Ollama installed" in capsys.readouterr().out


def test_confirmed_but_install_fails(monkeypatch, capsys):
    import core.ollama_install as install_mod
    from rich.prompt import Confirm

    monkeypatch.setattr(install_mod, "is_ollama_installed", lambda: False)
    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: True))
    monkeypatch.setattr("tools.shell_tools.run_command",
                        lambda cmd, timeout=None: ("", "network unreachable", 1))
    _offer_install_ollama()
    out = capsys.readouterr().out
    assert "did not complete successfully" in out
    assert "ollama.com/download" in out


# ── _run_ollama_preflight — provider-gated ──────────────────────────────────────

def test_preflight_runs_for_ollama_provider(monkeypatch):
    import cli

    calls = []
    monkeypatch.setattr(cli, "_offer_install_ollama", lambda: calls.append("offer"))
    monkeypatch.setattr(cli, "_check_ollama", lambda *a: calls.append("check"))
    _run_ollama_preflight(SimpleNamespace(model_provider="ollama",
                                          model_base_url="http://x", model_name="m"))
    assert calls == ["offer", "check"]


def test_preflight_skipped_for_openai_compatible_provider(monkeypatch):
    import cli

    calls = []
    monkeypatch.setattr(cli, "_offer_install_ollama", lambda: calls.append("offer"))
    monkeypatch.setattr(cli, "_check_ollama", lambda *a: calls.append("check"))
    _run_ollama_preflight(SimpleNamespace(model_provider="openai_compatible",
                                          model_base_url="http://x", model_name="m"))
    assert calls == []
