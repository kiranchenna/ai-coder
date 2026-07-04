"""Integration tests for the agent loop (AgentSession.send) using a scripted LLM."""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import yaml
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from agent.loop import MAX_STEPS, AgentSession


class ScriptedLLM:
    """Yields pre-scripted streamed responses; repeats the last when exhausted."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def stream(self, messages):
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return iter(self.responses[idx])


def _session(responses):
    """An AgentSession on an empty temp workspace, driven by a scripted LLM."""
    ws = Path(tempfile.mkdtemp())
    s = AgentSession(ws)               # offline: ChatOllama is constructed lazily
    s.llm = ScriptedLLM(responses)     # swap in the script
    s._history_budget = 10_000_000     # don't compact during these tests
    return s


def _native_tool_call(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "1", "type": "tool_call"}])


# ─── Final answer (streamed in chunks) ────────────────────────────────────────

def test_final_answer_streamed_in_chunks():
    s = _session([[AIMessageChunk(content="Hel"), AIMessageChunk(content="lo!")]])
    out = s.send("hi")
    assert out == "Hello!"
    assert s.llm.calls == 1
    # history ends with a clean AIMessage carrying the full answer
    assert isinstance(s.messages[-1], AIMessage) and s.messages[-1].content == "Hello!"


# ─── Native tool call → result fed back → final answer ────────────────────────

def test_native_tool_call_then_answer():
    s = _session([
        [_native_tool_call("list_files", {"path": "."})],
        [AIMessageChunk(content="done")],
    ])
    out = s.send("list the files")
    assert out == "done"
    assert s.llm.calls == 2
    # the tool actually ran and its result was appended as a ToolMessage
    assert any(isinstance(m, ToolMessage) for m in s.messages)


# ─── Text-emitted tool call (local-model fallback) is recovered and run ────────

def test_text_tool_call_fallback():
    s = _session([
        [AIMessageChunk(content='{"name": "list_files", "arguments": {"path": "."}}')],
        [AIMessageChunk(content="listed them")],
    ])
    out = s.send("list the files")
    assert out == "listed them"
    assert s.llm.calls == 2
    # fallback feeds results back as a HumanMessage, not a ToolMessage
    assert any(getattr(m, "content", "").startswith("Tool results:")
               for m in s.messages if hasattr(m, "content"))


# ─── Runaway tool-calling is bounded by the step cap ──────────────────────────

def test_step_cap_bounds_runaway_tool_calls():
    # always returns a tool call → never reaches a final answer
    s = _session([[_native_tool_call("list_files", {"path": "."})]])
    out = s.send("loop forever")
    assert out == ""
    assert s.llm.calls == MAX_STEPS


# ─── Empty stream surfaces as an error (not a silent empty answer) ────────────

def test_empty_stream_raises():
    s = _session([[]])  # zero chunks
    with pytest.raises(RuntimeError):
        s.send("hi")


# ─── last_turn_complete distinguishes a real answer from step-cap exhaustion ───

def test_last_turn_complete_true_on_final_answer():
    s = _session([[AIMessageChunk(content="done")]])
    s.send("hi")
    assert s.last_turn_complete is True


def test_last_turn_complete_false_when_step_cap_hit():
    # never stops calling a tool → exhausts the step cap without a final answer
    s = _session([[_native_tool_call("list_files", {"path": "."})]])
    s.send("loop forever")
    assert s.last_turn_complete is False


# ─── The text-recovery gate rejects illustrative JSON, accepts real calls ──────

def test_actionable_gate_rejects_large_example_in_prose():
    from agent.loop import _is_actionable_tool_message

    # A big JSON *example* embedded in a longer explanation must NOT be executed.
    explanation = (
        "To write a file you would call the write_file tool. For example, you could "
        "pass a structure like the following to create a config file, but only do this "
        "when the user actually asks you to — here is roughly what that call looks like "
        "in practice when you decide to use it during a real task:"
    )
    example = '{"name": "write_file", "arguments": {"path": "x.py", "content": "print(1)"}}'
    assert _is_actionable_tool_message(explanation + " " + example) is False


def test_actionable_gate_accepts_real_call_with_short_leadin():
    from agent.loop import _is_actionable_tool_message

    msg = 'Reading the file now: {"name": "read_file", "arguments": {"path": "x.py"}}'
    assert _is_actionable_tool_message(msg) is True


# ── /model — interactive picker (mirrors Claude Code's /model) ──────────────────

def _isolate_config(monkeypatch, tmp_path):
    """Redirect config.yaml to a temp dir so /model persistence tests never
    write to the developer's real ~/.aicoder/config.yaml."""
    import core.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "AICODER_HOME", tmp_path)
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.yaml")
    return cfg_mod.get_config()


def test_switch_model_persists_and_rebinds(tmp_path, monkeypatch):
    import core.model as model_mod
    from agent.loop import _switch_model

    monkeypatch.setattr(model_mod, "is_model_pulled", lambda base_url, name: True)
    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["model"]["name"]
    session = SimpleNamespace(llm=None, tools=[])
    try:
        _switch_model("qwen2.5-coder:14b", session)
        assert cfg.raw()["model"]["name"] == "qwen2.5-coder:14b"
        assert session.llm is not None                     # rebuilt for the new model
        saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert saved["model"]["name"] == "qwen2.5-coder:14b"   # persisted, not session-only
    finally:
        cfg.raw()["model"]["name"] = original


def test_switch_model_warns_when_not_pulled(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _switch_model

    monkeypatch.setattr(model_mod, "is_model_pulled", lambda base_url, name: False)
    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["model"]["name"]
    try:
        _switch_model("not-pulled:99b", SimpleNamespace(llm=None, tools=[]))
        assert "may not be pulled yet" in capsys.readouterr().out
    finally:
        cfg.raw()["model"]["name"] = original


def test_model_command_direct_name_switches(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_model_command

    monkeypatch.setattr(model_mod, "is_model_pulled", lambda base_url, name: True)
    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["model"]["name"]
    try:
        _handle_model_command("qwen2.5-coder:14b", SimpleNamespace(llm=None, tools=[]))
        assert cfg.raw()["model"]["name"] == "qwen2.5-coder:14b"
        assert "saved as your default" in capsys.readouterr().out
    finally:
        cfg.raw()["model"]["name"] = original


def test_model_command_lists_and_selects(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_model_command
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["model"]["name"]
    fake_models = [
        {"name": original, "size": 4_000_000_000},
        {"name": "qwen2.5-coder:14b", "size": 9_000_000_000},
    ]
    monkeypatch.setattr(model_mod, "list_ollama_models", lambda base_url: fake_models)
    monkeypatch.setattr(model_mod, "is_model_pulled", lambda base_url, name: True)
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: "2"))  # pick the 2nd entry
    try:
        _handle_model_command("", SimpleNamespace(llm=None, tools=[]))
        out = capsys.readouterr().out
        assert "(current)" in out                          # the listing marks the active model
        assert cfg.raw()["model"]["name"] == "qwen2.5-coder:14b"
    finally:
        cfg.raw()["model"]["name"] = original


def test_model_command_blank_keeps_current(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_model_command
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["model"]["name"]
    monkeypatch.setattr(model_mod, "list_ollama_models",
                        lambda base_url: [{"name": original, "size": 1}])
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: ""))
    _handle_model_command("", SimpleNamespace(llm=None, tools=[]))
    assert cfg.raw()["model"]["name"] == original           # blank = cancel, no change
    assert "Kept current model" in capsys.readouterr().out


def test_model_command_rejects_out_of_range_choice(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_model_command
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["model"]["name"]
    monkeypatch.setattr(model_mod, "list_ollama_models",
                        lambda base_url: [{"name": original, "size": 1}])
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: "99"))
    _handle_model_command("", SimpleNamespace(llm=None, tools=[]))
    assert cfg.raw()["model"]["name"] == original           # out of range → no change
    assert "Invalid selection" in capsys.readouterr().out


def test_model_command_unreachable_ollama_suggests_direct_switch(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_model_command

    _isolate_config(monkeypatch, tmp_path)

    def boom(base_url):
        raise ConnectionError("no ollama")

    monkeypatch.setattr(model_mod, "list_ollama_models", boom)
    _handle_model_command("", SimpleNamespace(llm=None, tools=[]))
    assert "Couldn't reach Ollama" in capsys.readouterr().out


# ── /model — curated "not yet installed" recommendations ────────────────────────

def test_model_menu_entries_dedupes_installed_from_recommended():
    from agent.loop import _model_menu_entries

    cfg = SimpleNamespace(model_name="qwen2.5-coder:7b")
    installed = [{"name": "qwen2.5-coder:7b", "size": 4_700_000_000}]
    entries = _model_menu_entries(cfg, installed)

    tags = [e["tag"] for e in entries]
    assert tags.count("qwen2.5-coder:7b") == 1        # installed → not also recommended
    assert entries[0]["installed"] and entries[0]["current"]
    assert any(not e["installed"] for e in entries)    # catalog picks still offered


def test_model_menu_entries_no_installed_still_offers_full_catalog():
    # A brand-new user with zero pulled models should still get every
    # recommendation, not a dead end.
    from agent.loop import _model_menu_entries
    from core.model_catalog import RECOMMENDED_MODELS

    cfg = SimpleNamespace(model_name="qwen2.5-coder:7b")
    entries = _model_menu_entries(cfg, [])
    assert len(entries) == len(RECOMMENDED_MODELS)
    assert all(not e["installed"] for e in entries)


def test_model_menu_entries_orders_installed_then_tiers_in_order():
    from agent.loop import _model_menu_entries

    cfg = SimpleNamespace(model_name="x")
    entries = _model_menu_entries(cfg, [{"name": "x", "size": 1}])
    sections = list(dict.fromkeys(e["section"] for e in entries))  # first-seen order
    assert sections[0] == "Installed"
    assert sections[1:] == [
        "Recommended — Fast & light (~8GB RAM/VRAM)",
        "Recommended — Balanced (~16GB) — the sweet spot",
        "Recommended — Powerful (24GB+)",
    ]


def test_render_model_menu_shows_tier_and_not_pulled_marker(capsys):
    from agent.loop import _model_menu_entries, _render_model_menu

    cfg = SimpleNamespace(model_name="qwen2.5-coder:7b")
    entries = _model_menu_entries(cfg, [{"name": "qwen2.5-coder:7b", "size": 4_700_000_000}])
    _render_model_menu(entries)
    out = capsys.readouterr().out
    assert "Recommended — Fast & light" in out
    assert "not pulled" in out
    assert "(current)" in out


def test_confirm_and_pull_declined_makes_no_change(tmp_path, monkeypatch, capsys):
    from agent.loop import _confirm_and_pull
    from rich.prompt import Confirm

    _isolate_config(monkeypatch, tmp_path)
    called = []
    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: False))
    monkeypatch.setattr("tools.shell_tools.run_command",
                        lambda *a, **k: called.append(1) or ("", "", 0))
    _confirm_and_pull("qwen3-coder:30b", 19_000_000_000, SimpleNamespace(llm=None, tools=[]))
    assert not called                                   # never attempted the pull
    assert "No change made" in capsys.readouterr().out


def test_confirm_and_pull_success_switches_and_persists(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _confirm_and_pull
    from rich.prompt import Confirm

    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["model"]["name"]
    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: True))
    monkeypatch.setattr("tools.shell_tools.run_command", lambda *a, **k: ("done", "", 0))
    monkeypatch.setattr(model_mod, "is_model_pulled", lambda base_url, name: True)
    try:
        _confirm_and_pull("qwen3-coder:30b", 19_000_000_000, SimpleNamespace(llm=None, tools=[]))
        assert cfg.raw()["model"]["name"] == "qwen3-coder:30b"
        out = capsys.readouterr().out
        assert "Pulled qwen3-coder:30b" in out
        assert "saved as your default" in out
    finally:
        cfg.raw()["model"]["name"] = original


def test_confirm_and_pull_failure_does_not_switch(tmp_path, monkeypatch, capsys):
    from agent.loop import _confirm_and_pull
    from rich.prompt import Confirm

    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["model"]["name"]
    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: True))
    monkeypatch.setattr("tools.shell_tools.run_command",
                        lambda *a, **k: ("", "no space left on device", 1))
    _confirm_and_pull("qwen3-coder:30b", 19_000_000_000, SimpleNamespace(llm=None, tools=[]))
    assert cfg.raw()["model"]["name"] == original           # failed pull → no switch
    assert "Failed to pull" in capsys.readouterr().out


def test_model_command_selecting_recommended_triggers_pull_flow(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_model_command
    from rich.prompt import Confirm, Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["model"]["name"]
    monkeypatch.setattr(model_mod, "list_ollama_models",
                        lambda base_url: [{"name": original, "size": 1}])
    monkeypatch.setattr(model_mod, "is_model_pulled", lambda base_url, name: True)
    # Entry 1 is the installed/current model; the first catalog (fast-tier) pick is #2.
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: "2"))
    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: True))
    monkeypatch.setattr("tools.shell_tools.run_command", lambda *a, **k: ("done", "", 0))
    try:
        _handle_model_command("", SimpleNamespace(llm=None, tools=[]))
        assert cfg.raw()["model"]["name"] == "qwen2.5-coder:3b"   # first fast-tier entry
    finally:
        cfg.raw()["model"]["name"] = original


def test_help_shows_active_devmode_profile(capsys):
    # A user editing config.yaml has no other in-session way to see which
    # Developer Mode profile/levers are actually active — /help must surface it.
    from core.config import get_config
    from agent.loop import _handle_command

    dm = get_config().raw()["devmode"]
    saved = dm.get("profile")
    dm["profile"] = "fast"
    try:
        _handle_command("/help", session=None, workspace=Path("."))
        assert "active profile: fast" in capsys.readouterr().out
    finally:
        if saved is None:
            dm.pop("profile", None)
        else:
            dm["profile"] = saved
