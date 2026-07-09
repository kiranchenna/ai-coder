"""Tests for cli.py's preflight check and --continue flag handling."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from cli import _run_preflight


# ── _run_preflight ───────────────────────────────────────────────────────────

def test_preflight_checks_the_configured_model_server(monkeypatch):
    import cli

    calls = []
    monkeypatch.setattr(cli, "_check_openai_compatible", lambda *a: calls.append("check"))
    _run_preflight(SimpleNamespace(model_base_url="http://x", model_name="m"))
    assert calls == ["check"]


# ── _check_openai_compatible — LM Studio / any OpenAI-compatible server ────────

def test_check_openai_compatible_warns_when_unreachable(monkeypatch, capsys):
    import cli
    import core.model as model_mod

    monkeypatch.setattr(model_mod, "is_lmstudio_reachable", lambda base_url: None)
    cli._check_openai_compatible("http://localhost:1234/v1", "some-model")
    out = capsys.readouterr().out
    assert "Cannot reach the configured model server" in out


def test_check_openai_compatible_warns_when_model_not_available(monkeypatch, capsys):
    import cli
    import core.model as model_mod

    monkeypatch.setattr(model_mod, "is_lmstudio_reachable", lambda base_url: {"other-model"})
    cli._check_openai_compatible("http://localhost:1234/v1", "missing-model")
    out = capsys.readouterr().out
    assert "isn't available on that server" in out


def test_check_openai_compatible_silent_when_model_available(monkeypatch, capsys):
    import cli
    import core.model as model_mod

    monkeypatch.setattr(model_mod, "is_lmstudio_reachable", lambda base_url: {"present-model"})
    cli._check_openai_compatible("http://localhost:1234/v1", "present-model")
    assert capsys.readouterr().out == ""


# ── --continue / -c ──────────────────────────────────────────────────────────

def test_continue_flag_is_passed_through_to_run_agent_repl(monkeypatch, tmp_path):
    import cli

    monkeypatch.setattr(sys, "argv", ["aicoder", "--workspace", str(tmp_path), "--continue"])
    monkeypatch.setattr(cli, "_run_preflight", lambda cfg: None)
    captured = {}
    monkeypatch.setattr(
        "agent.loop.run_agent_repl",
        lambda workspace, continue_session=False: captured.update(
            workspace=workspace, continue_session=continue_session,
        ),
    )
    # Not a real tty in a test process, so this exercises the plain-REPL path.
    cli.main()
    assert captured["continue_session"] is True
    assert captured["workspace"] == tmp_path.resolve()


def test_without_continue_flag_defaults_to_a_fresh_session(monkeypatch, tmp_path):
    import cli

    monkeypatch.setattr(sys, "argv", ["aicoder", "--workspace", str(tmp_path)])
    monkeypatch.setattr(cli, "_run_preflight", lambda cfg: None)
    captured = {}
    monkeypatch.setattr(
        "agent.loop.run_agent_repl",
        lambda workspace, continue_session=False: captured.update(
            continue_session=continue_session,
        ),
    )
    cli.main()
    assert captured["continue_session"] is False
