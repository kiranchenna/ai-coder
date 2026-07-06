"""Tests for the new Claude-Code-style slash commands in agent/loop.py:
/init, /status, /context, /compact, /permissions, /mcp, /hooks, /review,
/bug, /doctor, /export.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def _isolate_config(monkeypatch, tmp_path):
    import core.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "AICODER_HOME", tmp_path)
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.yaml")
    # get_config() caches a module-level singleton — without resetting it too,
    # a test running earlier in the same session (isolated or not) can leave
    # a *shared* Config object cached, and this redirect would be silently
    # ignored (get_config() would keep returning that stale instance instead
    # of loading fresh from the path just set above).
    monkeypatch.setattr(cfg_mod, "_config", None)
    return cfg_mod.get_config()


def _session(**overrides):
    base = dict(
        tools_by_name={"read_file": 1, "write_file": 2},
        messages=[SystemMessage(content="sys")],
        _history_budget=10_000,
        mcp=SimpleNamespace(status=lambda: []),
        instructions="",
        workspace=Path("."),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ── /init ────────────────────────────────────────────────────────────────────

def test_init_sends_the_analysis_prompt_and_reloads_instructions(tmp_path, monkeypatch):
    from agent.loop import _handle_init_command, _INIT_PROMPT

    sent = []
    session = _session()
    session.send = lambda prompt: sent.append(prompt)

    reloaded = []
    monkeypatch.setattr("agent.loop._reload_instructions",
                        lambda s, ws: reloaded.append((s, ws)))
    _handle_init_command(session, tmp_path)
    assert sent == [_INIT_PROMPT]
    assert reloaded == [(session, tmp_path)]


def test_reload_instructions_updates_system_message(tmp_path, monkeypatch):
    from agent.loop import _reload_instructions

    (tmp_path / "AICODER.md").write_text("- Use snake_case.", encoding="utf-8")
    session = _session()
    _reload_instructions(session, tmp_path)
    assert "snake_case" in session.instructions
    assert "snake_case" in session.messages[0].content


# ── /status ──────────────────────────────────────────────────────────────────

def test_status_shows_workspace_model_and_profile(tmp_path, monkeypatch, capsys):
    from agent.loop import _handle_status_command

    _isolate_config(monkeypatch, tmp_path)
    _handle_status_command(_session(), tmp_path)
    out = capsys.readouterr().out
    assert "qwen2.5-coder:7b" in out
    assert "balanced" in out
    assert "ollama" in out


# ── /context ─────────────────────────────────────────────────────────────────

def test_context_reports_usage_percentage(capsys):
    from agent.loop import _handle_context_command

    session = _session(messages=[SystemMessage(content="sys"), HumanMessage(content="x" * 1000)],
                       _history_budget=10_000)
    _handle_context_command(session)
    out = capsys.readouterr().out
    assert "10%" in out
    assert "1,000 chars" in out


def test_context_flags_when_over_budget(capsys):
    from agent.loop import _handle_context_command

    session = _session(messages=[SystemMessage(content="sys"), HumanMessage(content="x" * 5000)],
                       _history_budget=1000)
    _handle_context_command(session)
    assert "will compact" in capsys.readouterr().out


# ── /compact ─────────────────────────────────────────────────────────────────

def test_compact_empty_conversation(capsys):
    from agent.loop import _handle_compact_command

    _handle_compact_command(_session(messages=[SystemMessage(content="sys")]))
    assert "Nothing to compact" in capsys.readouterr().out


def test_compact_within_budget_is_a_noop(capsys):
    from agent.loop import _handle_compact_command

    session = _session(
        messages=[SystemMessage(content="sys"), HumanMessage(content="hi"), AIMessage(content="hey")],
        _history_budget=10_000,
    )
    called = []
    session._compact_history_if_needed = lambda: called.append(1)
    _handle_compact_command(session)
    assert not called
    assert "within budget" in capsys.readouterr().out


def test_compact_over_budget_triggers_real_compaction():
    from agent.loop import _handle_compact_command

    session = _session(
        messages=[SystemMessage(content="sys"), HumanMessage(content="x" * 5000),
                 AIMessage(content="y" * 5000)],
        _history_budget=100,
    )
    called = []
    session._compact_history_if_needed = lambda: called.append(1)
    _handle_compact_command(session)
    assert called == [1]


# ── /permissions ─────────────────────────────────────────────────────────────

def test_permissions_no_arg_shows_current_modes(tmp_path, monkeypatch, capsys):
    from agent.loop import _handle_permissions_command

    _isolate_config(monkeypatch, tmp_path)
    _handle_permissions_command("")
    out = capsys.readouterr().out
    assert "always" in out and "auto" in out


def test_permissions_sets_shell_mode(tmp_path, monkeypatch, capsys):
    from agent.loop import _handle_permissions_command
    from core.config import get_config

    _isolate_config(monkeypatch, tmp_path)
    _handle_permissions_command("shell smart")
    assert get_config().shell_confirmation == "smart"
    assert "set to smart" in capsys.readouterr().out


def test_permissions_sets_files_mode(tmp_path, monkeypatch, capsys):
    from agent.loop import _handle_permissions_command
    from core.config import get_config

    _isolate_config(monkeypatch, tmp_path)
    _handle_permissions_command("files never")
    assert get_config().file_confirmation == "never"


def test_permissions_invalid_mode_shows_error(tmp_path, monkeypatch, capsys):
    from agent.loop import _handle_permissions_command

    _isolate_config(monkeypatch, tmp_path)
    _handle_permissions_command("shell bogus")
    assert "Invalid mode" in capsys.readouterr().out or "Use:" in capsys.readouterr().out


def test_permissions_unknown_kind_shows_usage(tmp_path, monkeypatch, capsys):
    from agent.loop import _handle_permissions_command

    _isolate_config(monkeypatch, tmp_path)
    _handle_permissions_command("network never")
    assert "Usage" in capsys.readouterr().out


# ── /mcp ─────────────────────────────────────────────────────────────────────

def test_mcp_no_servers_configured(capsys):
    from agent.loop import _handle_mcp_command

    _handle_mcp_command(_session(mcp=SimpleNamespace(status=lambda: [])))
    assert "No MCP servers configured" in capsys.readouterr().out


def test_mcp_lists_connected_and_disconnected_servers(capsys):
    from agent.loop import _handle_mcp_command

    status = [
        {"name": "filesystem", "connected": True, "tools": ["read_file", "write_file"]},
        {"name": "broken", "connected": False, "tools": []},
    ]
    _handle_mcp_command(_session(mcp=SimpleNamespace(status=lambda: status)))
    out = capsys.readouterr().out
    assert "filesystem" in out and "read_file" in out
    assert "broken" in out


# ── /hooks ───────────────────────────────────────────────────────────────────

def test_hooks_none_configured(tmp_path, monkeypatch, capsys):
    from agent.loop import _handle_hooks_command

    _isolate_config(monkeypatch, tmp_path)
    _handle_hooks_command()
    assert "No hooks configured" in capsys.readouterr().out


def test_hooks_lists_configured_events(tmp_path, monkeypatch, capsys):
    from agent.loop import _handle_hooks_command

    cfg = _isolate_config(monkeypatch, tmp_path)
    cfg.raw()["hooks"] = {
        "PreToolUse": [{"matcher": "run_shell", "command": "guard.sh"}],
        "Stop": [{"command": "notify.sh"}],
    }
    _handle_hooks_command()
    out = capsys.readouterr().out
    assert "PreToolUse" in out and "guard.sh" in out
    assert "Stop" in out and "notify.sh" in out


# ── /review ──────────────────────────────────────────────────────────────────

def test_review_sends_a_diff_review_prompt():
    from agent.loop import _handle_review_command

    sent = []
    session = _session()
    session.send = lambda prompt: sent.append(prompt)
    _handle_review_command(session)
    assert len(sent) == 1
    assert "git_diff" in sent[0] or "diff" in sent[0].lower()


# ── /bug ─────────────────────────────────────────────────────────────────────

def test_bug_shows_issues_url(capsys):
    from agent.loop import _handle_bug_command

    _handle_bug_command()
    assert "github.com/kiranchenna/ai-coder/issues" in capsys.readouterr().out


# ── /doctor ──────────────────────────────────────────────────────────────────

def test_doctor_calls_selftest(monkeypatch):
    import core.model as model_mod
    from agent.loop import _handle_doctor_command

    called = []
    monkeypatch.setattr(model_mod, "selftest", lambda: called.append(1) or True)
    _handle_doctor_command()
    assert called == [1]


# ── /export ──────────────────────────────────────────────────────────────────

def test_export_nothing_to_export(capsys):
    from agent.loop import _handle_export_command

    session = _session(messages=[SystemMessage(content="sys")])
    _handle_export_command("", session, Path("."))
    assert "Nothing to export" in capsys.readouterr().out


def test_export_writes_transcript_to_named_file(tmp_path, capsys):
    from agent.loop import _handle_export_command

    session = _session(messages=[
        SystemMessage(content="sys"),
        HumanMessage(content="explain the repo"),
        AIMessage(content="It's a CLI tool."),
    ])
    _handle_export_command("out.md", session, tmp_path)
    out_file = tmp_path / "out.md"
    assert out_file.exists()
    text = out_file.read_text()
    assert "explain the repo" in text
    assert "It's a CLI tool." in text
    assert "Exported conversation to out.md" in capsys.readouterr().out


def test_export_default_filename_has_timestamp_pattern(tmp_path):
    from agent.loop import _handle_export_command

    session = _session(messages=[SystemMessage(content="sys"), HumanMessage(content="hi")])
    _handle_export_command("", session, tmp_path)
    matches = list(tmp_path.glob("aicoder-transcript-*.md"))
    assert len(matches) == 1


def test_export_rejects_path_escaping_workspace(tmp_path, capsys):
    from agent.loop import _handle_export_command

    session = _session(messages=[SystemMessage(content="sys"), HumanMessage(content="hi")])
    _handle_export_command("../../etc/evil.md", session, tmp_path)
    assert not (tmp_path.parent.parent / "etc" / "evil.md").exists()
    assert "escapes workspace" in capsys.readouterr().out.lower() or "permission" in capsys.readouterr().out.lower()


# ── _format_transcript ───────────────────────────────────────────────────────

def test_format_transcript_includes_all_message_kinds():
    from agent.loop import _format_transcript

    tc = AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "1"}])
    messages = [
        SystemMessage(content="sys prompt, skipped"),
        HumanMessage(content="do the thing"),
        tc,
        ToolMessage(content="file contents here", tool_call_id="1"),
        AIMessage(content="Done."),
    ]
    text = _format_transcript(messages)
    assert "sys prompt" not in text          # system prompt itself is skipped
    assert "do the thing" in text
    assert "read_file" in text
    assert "file contents here" in text
    assert "Done." in text


# ── /help renders every new command without crashing (and no swallowed brackets) ─

def test_help_renders_bracket_placeholders_literally(capsys):
    from agent.loop import _handle_command

    _handle_command("/help", _session(), Path("."))
    out = capsys.readouterr().out
    assert "/model [name]" in out
    assert "/export [file]" in out


# ── /exit, /quit, /q — the migrated exit commands ───────────────────────────────

def test_exit_returns_true_to_signal_the_repl_to_stop(capsys):
    from agent.loop import _handle_command

    assert _handle_command("/exit", _session(), Path(".")) is True
    assert "Goodbye" in capsys.readouterr().out


def test_quit_and_q_are_aliases_for_exit():
    from agent.loop import _handle_command

    assert _handle_command("/quit", _session(), Path(".")) is True
    assert _handle_command("/q", _session(), Path(".")) is True


def test_ordinary_commands_do_not_signal_exit():
    from agent.loop import _handle_command

    assert _handle_command("/help", _session(), Path(".")) is False
    assert _handle_command("/tools", _session(), Path(".")) is False


# ── /develop, /dev, /plan, /resume — migrated from bare words to "/" ────────────

def test_develop_no_idea_shows_usage(tmp_path, capsys, monkeypatch):
    from agent.loop import _handle_develop_command

    constructed = []
    monkeypatch.setattr("devmode.session.DevSession",
                        lambda *a, **k: constructed.append((a, k)))
    _handle_develop_command("", tmp_path)
    assert not constructed
    assert "Usage: /develop" in capsys.readouterr().out


def test_develop_runs_with_idea(tmp_path, monkeypatch):
    ran = []

    class FakeDevSession:
        def __init__(self, workspace, idea, auto=False):
            ran.append((workspace, idea, auto))

        def run(self):
            ran.append("ran")

    monkeypatch.setattr("devmode.session.DevSession", FakeDevSession)
    from agent.loop import _handle_develop_command

    _handle_develop_command("a todo app", tmp_path)
    assert ran[0] == (tmp_path, "a todo app", False)
    assert ran[1] == "ran"


def test_develop_fast_flag_sets_auto_true(tmp_path, monkeypatch):
    captured = {}

    class FakeDevSession:
        def __init__(self, workspace, idea, auto=False):
            captured["idea"], captured["auto"] = idea, auto

        def run(self):
            pass

    monkeypatch.setattr("devmode.session.DevSession", FakeDevSession)
    from agent.loop import _handle_develop_command

    _handle_develop_command("--fast a todo app", tmp_path)
    assert captured == {"idea": "a todo app", "auto": True}


def test_dev_no_arg_resumes(tmp_path, monkeypatch):
    calls = []

    class FakeDevSession:
        def __init__(self, workspace):
            pass

        def run(self, resume=False):
            calls.append(("run", resume))

    monkeypatch.setattr("devmode.session.DevSession", FakeDevSession)
    from agent.loop import _handle_dev_command

    _handle_dev_command("", tmp_path)
    assert calls == [("run", True)]


def test_dev_status_routes_to_show_status(tmp_path, monkeypatch):
    calls = []

    class FakeDevSession:
        def __init__(self, workspace):
            pass

        def show_status(self):
            calls.append("status")

    monkeypatch.setattr("devmode.session.DevSession", FakeDevSession)
    from agent.loop import _handle_dev_command

    _handle_dev_command("status", tmp_path)
    assert calls == ["status"]


def test_dev_build_routes_to_builder(tmp_path, monkeypatch):
    class FakeDevSession:
        def __init__(self, workspace):
            pass

    built = []

    class FakeBuilder:
        def __init__(self, workspace, ds):
            built.append((workspace, ds))

        def build(self):
            built.append("built")

    monkeypatch.setattr("devmode.session.DevSession", FakeDevSession)
    monkeypatch.setattr("devmode.build.Builder", FakeBuilder)
    from agent.loop import _handle_dev_command

    _handle_dev_command("build", tmp_path)
    assert built[-1] == "built"


def test_dev_revisit_routes_with_phase_argument(tmp_path, monkeypatch):
    calls = []

    class FakeDevSession:
        def __init__(self, workspace):
            pass

        def revisit(self, phase):
            calls.append(phase)

    monkeypatch.setattr("devmode.session.DevSession", FakeDevSession)
    from agent.loop import _handle_dev_command

    _handle_dev_command("revisit security", tmp_path)
    assert calls == ["security"]


def test_dev_resolve_routes_to_resolve(tmp_path, monkeypatch):
    calls = []

    class FakeDevSession:
        def __init__(self, workspace):
            pass

        def resolve(self):
            calls.append("resolved")

    monkeypatch.setattr("devmode.session.DevSession", FakeDevSession)
    from agent.loop import _handle_dev_command

    _handle_dev_command("resolve", tmp_path)
    assert calls == ["resolved"]


def test_plan_no_goal_shows_usage(tmp_path, capsys, monkeypatch):
    from agent.loop import _handle_plan_command

    constructed = []
    monkeypatch.setattr("agent.planner.Planner", lambda *a, **k: constructed.append(1))
    _handle_plan_command("", _session(), tmp_path)
    assert not constructed
    assert "Usage: /plan" in capsys.readouterr().out


def test_plan_creates_shows_and_runs_on_confirm(tmp_path, monkeypatch):
    from rich.prompt import Confirm

    calls = []

    class FakePlanner:
        def __init__(self, workspace, session):
            pass

        def create_plan(self, goal):
            calls.append(("create", goal))
            return {"goal": goal, "tasks": []}

        def show(self, plan):
            calls.append(("show", plan))

        def run(self):
            calls.append("ran")

    monkeypatch.setattr("agent.planner.Planner", FakePlanner)
    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: True))
    from agent.loop import _handle_plan_command

    _handle_plan_command("build a todo API", _session(), tmp_path)
    assert calls[0] == ("create", "build a todo API")
    assert calls[-1] == "ran"


def test_plan_not_run_when_confirmation_declined(tmp_path, monkeypatch):
    from rich.prompt import Confirm

    calls = []

    class FakePlanner:
        def __init__(self, workspace, session):
            pass

        def create_plan(self, goal):
            return {"goal": goal, "tasks": []}

        def show(self, plan):
            pass

        def run(self):
            calls.append("ran")

    monkeypatch.setattr("agent.planner.Planner", FakePlanner)
    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: False))
    from agent.loop import _handle_plan_command

    _handle_plan_command("build a todo API", _session(), tmp_path)
    assert not calls


def test_plan_no_plan_produced_shows_message(tmp_path, capsys, monkeypatch):
    class FakePlanner:
        def __init__(self, workspace, session):
            pass

        def create_plan(self, goal):
            return None

    monkeypatch.setattr("agent.planner.Planner", FakePlanner)
    from agent.loop import _handle_plan_command

    _handle_plan_command("build a todo API", _session(), tmp_path)
    assert "Couldn't produce a task plan" in capsys.readouterr().out


def test_resume_runs_the_planner(tmp_path, monkeypatch):
    calls = []

    class FakePlanner:
        def __init__(self, workspace, session):
            pass

        def run(self):
            calls.append("ran")

    monkeypatch.setattr("agent.planner.Planner", FakePlanner)
    from agent.loop import _handle_resume_command

    _handle_resume_command(_session(), tmp_path)
    assert calls == ["ran"]


def test_bare_words_no_longer_recognized_as_commands():
    # /develop, /dev, /plan, /resume, /exit are the only entry points now —
    # confirm the dispatcher only recognizes the "/"-prefixed name, matching
    # every other command (no special-cased bare-word fallback). The real
    # guarantee is enforced at the call site in run_agent_repl, which only
    # invokes _handle_command when user.startswith("/") — verified below by
    # inspecting that no bare-word branches remain in its source.
    import inspect

    src = inspect.getsource(__import__("agent.loop", fromlist=["run_agent_repl"]).run_agent_repl)
    assert 'user.startswith("/")' in src
    assert "low ==" not in src          # the old bare-word branches are gone
    assert "_EXIT_WORDS" not in src
