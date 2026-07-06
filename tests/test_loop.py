"""Integration tests for the agent loop (AgentSession.send) using a scripted LLM."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import yaml
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

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


@pytest.fixture(autouse=True)
def _isolate_memory_dir(monkeypatch, tmp_path):
    """Every AgentSession.send() now persists a transcript to
    ~/.aicoder/memory/<project_id>/conversation.json (for `aicoder
    --continue`) — autouse so no test in this file (nearly all of which
    exercise send() constantly, via _session() below) writes into the
    developer's real ~/.aicoder/memory/."""
    monkeypatch.setattr("core.config.MEMORY_DIR", tmp_path / "memory")


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


def test_text_tool_call_placeholder_is_not_imitable_as_a_request_format():
    # The injected placeholder that replaces the recovered raw JSON must read as
    # a past-tense fact ("I already called..."), not a terse "(Requested tools:
    # X)"-style directive — a small model tends to imitate the surface FORM of
    # its own prior turn, and a request-shaped placeholder gets copied verbatim
    # as a fake tool call on the next turn instead of a real one (observed with
    # qwen2.5-coder:7b during manual testing).
    s = _session([
        [AIMessageChunk(content='{"name": "list_files", "arguments": {"path": "."}}')],
        [AIMessageChunk(content="listed them")],
    ])
    s.send("list the files")
    placeholder = next(m for m in s.messages
                       if isinstance(m, AIMessage) and "list_files" in (m.content or ""))
    assert "Requested tools" not in placeholder.content
    assert "already called" in placeholder.content


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


# ─── Transcript persistence — `aicoder --continue` ─────────────────────────────
# The system prompt (index 0) is never persisted/restored — only what comes
# after it — since it's rebuilt fresh from live repo state on every launch.

def test_send_persists_a_transcript_after_the_turn():
    s = _session([[AIMessageChunk(content="Hello!")]])
    s.send("hi")
    path = s._transcript_path()
    assert path.exists()
    saved = json.loads(path.read_text())
    assert len(saved) == 2  # the human message + the AI's final answer


def test_load_transcript_restores_messages_after_a_fresh_system_prompt():
    s1 = _session([[AIMessageChunk(content="Hello!")]])
    s1.send("hi")

    s2 = AgentSession(s1.workspace)  # a fresh session — new system prompt
    original_system = s2.messages[0]
    loaded = s2.load_transcript()

    assert loaded is True
    assert s2.messages[0] is original_system  # system prompt untouched
    assert len(s2.messages) == 3  # system + human + AI
    assert isinstance(s2.messages[1], HumanMessage) and s2.messages[1].content == "hi"
    assert isinstance(s2.messages[2], AIMessage) and s2.messages[2].content == "Hello!"


def test_load_transcript_returns_false_when_nothing_saved():
    s = _session([[AIMessageChunk(content="ignored")]])
    assert s.load_transcript() is False
    assert len(s.messages) == 1  # untouched — still just the system prompt


def test_load_transcript_returns_false_on_corrupted_file():
    s = _session([[AIMessageChunk(content="ignored")]])
    path = s._transcript_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json{{{")
    assert s.load_transcript() is False
    assert len(s.messages) == 1


def test_transcript_persists_even_when_a_turn_is_interrupted():
    s = _session([[AIMessageChunk(content="ignored")]])
    chunks = [AIMessageChunk(content="Hel"), AIMessageChunk(content="lo!")]
    s.llm = _InterruptingLLM(s, chunks, interrupt_after=1)  # fires mid-stream
    out = s.send("hi")
    assert out == ""  # interrupted
    assert s._transcript_path().exists()  # still saved (best-effort, in a finally)


# ─── request_interrupt() — best-effort mid-turn cancellation (the TUI's Esc) ───

class _InterruptingLLM:
    """Fires request_interrupt() on the session partway through the stream —
    lets us test the interrupt check deterministically, without depending on
    real Ollama's chunk timing."""

    def __init__(self, session, chunks, interrupt_after: int):
        self.session = session
        self.chunks = chunks
        self.interrupt_after = interrupt_after
        self.calls = 0

    def stream(self, messages):
        self.calls += 1
        for i, chunk in enumerate(self.chunks):
            if i == self.interrupt_after:
                self.session.request_interrupt()
            yield chunk


def test_interrupt_mid_stream_stops_the_turn_cleanly(capsys):
    s = _session([[AIMessageChunk(content="ignored")]])
    chunks = [AIMessageChunk(content="Hel"), AIMessageChunk(content="lo!"),
              AIMessageChunk(content=" world")]
    s.llm = _InterruptingLLM(s, chunks, interrupt_after=1)

    out = s.send("hi")

    assert out == ""
    assert "Interrupted" in capsys.readouterr().out
    # no half-formed AIMessage was appended to history
    assert not any(isinstance(m, AIMessage) and m.content == "Hello!" for m in s.messages)


def test_interrupt_between_tool_steps_stops_before_the_next_invoke():
    s = _session([[_native_tool_call("list_files", {"path": "."})]])

    real_exec = s._exec
    def _exec_then_interrupt(call):
        s.request_interrupt()
        return real_exec(call)
    s._exec = _exec_then_interrupt

    out = s.send("loop forever")

    assert out == ""
    assert s.llm.calls == 1  # stopped before a second _invoke()


def test_interrupt_flag_is_cleared_at_the_start_of_each_send():
    s = _session([[AIMessageChunk(content="Hello!")]])
    s.request_interrupt()
    assert s._interrupt.is_set()

    out = s.send("hi")  # send() clears the flag before running

    assert out == "Hello!"


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
    # get_config() caches a module-level singleton — without resetting it too,
    # a test running earlier in the same session (isolated or not) can leave
    # a *shared* Config object cached, and this redirect would be silently
    # ignored (get_config() would keep returning that stale instance instead
    # of loading fresh from the path just set above).
    monkeypatch.setattr(cfg_mod, "_config", None)
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


# ── /model — openai_compatible provider (no Ollama discovery API) ──────────────

def test_model_command_non_ollama_shows_info_not_picker(tmp_path, monkeypatch, capsys):
    from agent.loop import _handle_model_command

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved = dict(cfg.raw()["model"])
    try:
        cfg.raw()["model"]["provider"] = "openai_compatible"
        cfg.raw()["model"]["base_url"] = "http://localhost:8080/v1"
        _handle_model_command("", SimpleNamespace(llm=None, tools=[]))
        out = capsys.readouterr().out
        assert "openai_compatible" in out
        assert "http://localhost:8080/v1" in out
        assert "config.yaml" in out
        assert "Available models (via Ollama)" not in out   # not the Ollama picker
    finally:
        cfg.raw()["model"] = saved


def test_model_command_direct_name_still_works_for_non_ollama_provider(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_model_command

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved = dict(cfg.raw()["model"])
    monkeypatch.setattr(model_mod, "is_model_pulled",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError(
                            "is_model_pulled must not be called for a non-ollama provider")))
    try:
        cfg.raw()["model"]["provider"] = "openai_compatible"
        _handle_model_command("gpt-4o-mini", SimpleNamespace(llm=None, tools=[]))
        assert cfg.raw()["model"]["name"] == "gpt-4o-mini"
        assert "saved as your default" in capsys.readouterr().out
    finally:
        cfg.raw()["model"] = saved


# ── Model-picker filtering — keep obviously-wrong-category models out ──────────
# Ollama's /api/tags lists every locally-pulled model with no way to tell an
# embedding-only model or a vision model from a coding model — so both
# pickers filter their own "Installed" section rather than showing anything
# pulled indiscriminately (a real mistake made live while testing /vision
# model: an embedding model got picked by accident).

def test_exclude_embedding_model_drops_the_configured_one():
    from agent.loop import _exclude_embedding_model

    cfg = SimpleNamespace(get=lambda *k, default=None: "nomic-embed-text-v2-moe"
                          if k == ("knowledge", "embedding_model") else default)
    installed = [
        {"name": "qwen2.5-coder:7b", "size": 1},
        {"name": "nomic-embed-text-v2-moe", "size": 1},
    ]
    result = _exclude_embedding_model(installed, cfg)
    assert [m["name"] for m in result] == ["qwen2.5-coder:7b"]


def test_exclude_embedding_model_noop_when_not_configured():
    from agent.loop import _exclude_embedding_model

    cfg = SimpleNamespace(get=lambda *k, default=None: default)
    installed = [{"name": "qwen2.5-coder:7b", "size": 1}]
    assert _exclude_embedding_model(installed, cfg) == installed


def test_filter_vision_capable_keeps_known_families_drops_others():
    from agent.loop import _filter_vision_capable

    cfg = SimpleNamespace(vision_model="qwen2.5vl:7b")
    installed = [
        {"name": "qwen2.5vl:7b", "size": 1},       # known vision family
        {"name": "llava:13b", "size": 1},          # known vision family
        {"name": "qwen2.5-coder:7b", "size": 1},   # coding model — dropped
        {"name": "nomic-embed-text-v2-moe", "size": 1},  # embedding — dropped
    ]
    result = {m["name"] for m in _filter_vision_capable(installed, cfg)}
    assert result == {"qwen2.5vl:7b", "llava:13b"}


def test_filter_vision_capable_keeps_a_custom_configured_model_regardless():
    from agent.loop import _filter_vision_capable

    # A model set directly in config.yaml that isn't in our curated
    # VISION_MODELS families must still show as "installed", not get dropped.
    cfg = SimpleNamespace(vision_model="my-custom-vlm:latest")
    installed = [{"name": "my-custom-vlm:latest", "size": 1}]
    assert _filter_vision_capable(installed, cfg) == installed


def test_model_command_picker_excludes_the_embedding_model(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_model_command
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["model"]["name"]
    fake_models = [
        {"name": original, "size": 1},
        {"name": cfg.raw()["knowledge"]["embedding_model"], "size": 1},
    ]
    monkeypatch.setattr(model_mod, "list_ollama_models", lambda base_url: fake_models)
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: ""))
    _handle_model_command("", SimpleNamespace(llm=None, tools=[]))
    out = capsys.readouterr().out
    assert cfg.raw()["knowledge"]["embedding_model"] not in out


def test_vision_model_command_picker_excludes_non_vision_models(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_vision_model_command
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["vision"]["model"]
    fake_models = [
        {"name": original, "size": 1},
        {"name": "qwen2.5-coder:7b", "size": 1},  # a coding model, happens to be pulled
    ]
    monkeypatch.setattr(model_mod, "list_ollama_models", lambda base_url: fake_models)
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: ""))
    _handle_vision_model_command()
    out = capsys.readouterr().out
    assert "qwen2.5-coder:7b" not in out


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


# ─── "Other…" — any Ollama model beyond the curated catalog ────────────────────
# _pull_arbitrary_model takes raw user input (unlike _confirm_and_pull's
# hardcoded catalog tags), so it must never go through run_command's
# shell=True — these tests specifically pin down subprocess.run with an argv
# list, since a shell-string path here would be a real injection risk.

def test_pull_arbitrary_model_declined_makes_no_change(tmp_path, monkeypatch, capsys):
    from agent.loop import _pull_arbitrary_model
    from rich.prompt import Confirm

    _isolate_config(monkeypatch, tmp_path)
    called = []
    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: False))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: called.append(1))
    _pull_arbitrary_model("llama3.2:1b", SimpleNamespace(llm=None, tools=[]))
    assert not called
    assert "No change made" in capsys.readouterr().out


def test_pull_arbitrary_model_uses_argv_list_not_a_shell_string(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _pull_arbitrary_model
    from rich.prompt import Confirm

    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["model"]["name"]
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: True))
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(model_mod, "is_model_pulled", lambda base_url, name: True)
    try:
        # A name containing shell metacharacters — if this were ever
        # interpolated into a shell string, it would inject a second command.
        tag = "llama3.2:1b; rm -rf /tmp/should-not-run"
        _pull_arbitrary_model(tag, SimpleNamespace(llm=None, tools=[]))
        # argv list form: the whole tag is ONE element, never shell-parsed.
        assert captured["args"] == ["ollama", "pull", tag]
        assert cfg.raw()["model"]["name"] == tag
    finally:
        cfg.raw()["model"]["name"] = original


def test_pull_arbitrary_model_failure_does_not_switch(tmp_path, monkeypatch, capsys):
    from agent.loop import _pull_arbitrary_model
    from rich.prompt import Confirm

    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["model"]["name"]
    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: True))
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="not found"),
    )
    _pull_arbitrary_model("bogus:model", SimpleNamespace(llm=None, tools=[]))
    assert cfg.raw()["model"]["name"] == original
    assert "Failed to pull" in capsys.readouterr().out


def test_pull_arbitrary_model_ollama_not_on_path(tmp_path, monkeypatch, capsys):
    from agent.loop import _pull_arbitrary_model
    from rich.prompt import Confirm

    _isolate_config(monkeypatch, tmp_path)
    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: True))

    def raise_not_found(*a, **k):
        raise FileNotFoundError()

    monkeypatch.setattr("subprocess.run", raise_not_found)
    _pull_arbitrary_model("llama3.2:1b", SimpleNamespace(llm=None, tools=[]))
    assert "isn't on your PATH" in capsys.readouterr().out


def test_handle_custom_model_entry_blank_makes_no_change(tmp_path, monkeypatch, capsys):
    from agent.loop import _handle_custom_model_entry
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["model"]["name"]
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: ""))
    _handle_custom_model_entry(SimpleNamespace(llm=None, tools=[]))
    assert cfg.raw()["model"]["name"] == original
    assert "No change made" in capsys.readouterr().out


def test_handle_custom_model_entry_already_pulled_switches_directly(tmp_path, monkeypatch):
    import core.model as model_mod
    from agent.loop import _handle_custom_model_entry
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["model"]["name"]
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: "llama3.2:1b"))
    monkeypatch.setattr(model_mod, "is_model_pulled", lambda base_url, name: True)
    pull_calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pull_calls.append(1))
    try:
        _handle_custom_model_entry(SimpleNamespace(llm=None, tools=[]))
        assert cfg.raw()["model"]["name"] == "llama3.2:1b"
        assert not pull_calls  # already pulled — no pull attempted
    finally:
        cfg.raw()["model"]["name"] = original


def test_handle_custom_model_entry_not_pulled_triggers_pull_flow(tmp_path, monkeypatch):
    import core.model as model_mod
    from agent.loop import _handle_custom_model_entry
    from rich.prompt import Confirm, Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["model"]["name"]
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: "llama3.2:1b"))
    monkeypatch.setattr(model_mod, "is_model_pulled", lambda base_url, name: False)
    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: True))
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    try:
        _handle_custom_model_entry(SimpleNamespace(llm=None, tools=[]))
        assert captured["args"] == ["ollama", "pull", "llama3.2:1b"]
        assert cfg.raw()["model"]["name"] == "llama3.2:1b"
    finally:
        cfg.raw()["model"]["name"] = original


def test_model_command_other_choice_routes_to_custom_entry(tmp_path, monkeypatch):
    import core.model as model_mod
    from agent.loop import _handle_model_command
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["model"]["name"]
    monkeypatch.setattr(model_mod, "list_ollama_models",
                        lambda base_url: [{"name": original, "size": 1}])
    monkeypatch.setattr(model_mod, "is_model_pulled", lambda base_url, name: True)
    # First Prompt.ask call is the picker's "1-N/o" choice, second is the
    # custom-entry's model-name prompt.
    answers = iter(["o", "llama3.2:1b"])
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: next(answers)))
    try:
        _handle_model_command("", SimpleNamespace(llm=None, tools=[]))
        assert cfg.raw()["model"]["name"] == "llama3.2:1b"
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


# ── Full-screen "alternate screen buffer" mode ──────────────────────────────────
# The same terminal mechanism vim/less/htop/Claude Code use: swap to a separate
# blank screen with no scrollback, restore the prior screen on exit. Rich's
# Console.screen() only emits the control codes when writing to a real
# terminal (is_terminal), so a Console forced into terminal mode (rather than a
# real pty) is enough to verify the codes without any visual/manual check.

def test_repl_enters_and_exits_alt_screen_on_a_real_terminal(monkeypatch, tmp_path):
    import io

    from rich.console import Console
    from rich.prompt import Prompt

    import agent.loop as loop_mod

    fake_console = Console(file=io.StringIO(), force_terminal=True, width=80)
    monkeypatch.setattr(loop_mod, "console", fake_console)
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: "/exit"))

    loop_mod.run_agent_repl(tmp_path)

    out = fake_console.file.getvalue()
    enable_at = out.find("\x1b[?1049h")
    disable_at = out.find("\x1b[?1049l")
    assert enable_at != -1, "alt-screen enable code missing"
    assert disable_at != -1, "alt-screen disable code missing"
    assert enable_at < disable_at, "screen was disabled before it was enabled"
    assert "\x1b[?25l" not in out, "cursor should stay visible (hide_cursor=False)"


def test_repl_skips_alt_screen_on_a_non_terminal(monkeypatch, tmp_path):
    # Piped/redirected output (e.g. scripted usage, tests, CI) must not get
    # raw escape codes written into it — Console.screen() already guards this
    # via is_terminal, this just locks in that AICoder relies on that guard.
    import io

    from rich.console import Console
    from rich.prompt import Prompt

    import agent.loop as loop_mod

    fake_console = Console(file=io.StringIO(), force_terminal=False, width=80)
    monkeypatch.setattr(loop_mod, "console", fake_console)
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: "/exit"))

    loop_mod.run_agent_repl(tmp_path)

    out = fake_console.file.getvalue()
    assert "\x1b[?1049h" not in out
    assert "\x1b[?1049l" not in out


# ─── describe_images / send_with_images — the two-model vision handoff ────────
# The vision model is built fresh via get_chat_model(model=...), never bound
# to self.llm/self.tools — these tests monkeypatch agent.loop.get_chat_model
# to hand back a fake vision model, independent of the session's own scripted
# text LLM used for the subsequent send().

class _FakeVisionLLM:
    """Captures the messages it was invoked with; returns a fixed description."""

    def __init__(self, description="a misaligned submit button overlapping the email field"):
        self.description = description
        self.invoked_with = None

    def invoke(self, messages):
        self.invoked_with = messages
        return AIMessage(content=self.description)


def _fake_image(tmp_path, name="screenshot.png") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\x89PNG\r\n\x1a\nnot a real png but that's fine for this test")
    return path


def test_describe_images_builds_multimodal_message(tmp_path, monkeypatch):
    import agent.loop as loop_mod

    _isolate_config(monkeypatch, tmp_path / "cfg")
    monkeypatch.setattr("core.model.is_model_pulled", lambda base_url, name: True)
    fake_vision = _FakeVisionLLM()
    monkeypatch.setattr(loop_mod, "get_chat_model", lambda **k: fake_vision)

    s = _session([[AIMessageChunk(content="ignored")]])
    image = _fake_image(tmp_path)
    result = s.describe_images([image], "what's wrong here?")

    assert result == fake_vision.description
    [message] = fake_vision.invoked_with
    assert message.content[0] == {"type": "text", "text": "what's wrong here?"}
    assert message.content[1]["type"] == "image_url"
    assert message.content[1]["image_url"].startswith("data:image/png;base64,")


def test_describe_images_no_vision_model_configured_raises(tmp_path, monkeypatch):
    cfg = _isolate_config(monkeypatch, tmp_path / "cfg")
    cfg.raw()["vision"]["model"] = ""

    s = _session([[AIMessageChunk(content="ignored")]])
    with pytest.raises(RuntimeError, match="No vision model configured"):
        s.describe_images([_fake_image(tmp_path)])


def test_describe_images_pulls_vision_model_if_not_installed(tmp_path, monkeypatch):
    import agent.loop as loop_mod
    from rich.prompt import Confirm

    _isolate_config(monkeypatch, tmp_path / "cfg")
    monkeypatch.setattr("core.model.is_model_pulled", lambda base_url, name: False)
    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: True))
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    fake_vision = _FakeVisionLLM()
    monkeypatch.setattr(loop_mod, "get_chat_model", lambda **k: fake_vision)

    s = _session([[AIMessageChunk(content="ignored")]])
    s.describe_images([_fake_image(tmp_path)])
    assert captured["args"] == ["ollama", "pull", "qwen2.5vl:7b"]


def test_describe_images_declines_pull_raises(tmp_path, monkeypatch):
    from rich.prompt import Confirm

    _isolate_config(monkeypatch, tmp_path / "cfg")
    monkeypatch.setattr("core.model.is_model_pulled", lambda base_url, name: False)
    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: False))

    s = _session([[AIMessageChunk(content="ignored")]])
    with pytest.raises(RuntimeError, match="isn't pulled"):
        s.describe_images([_fake_image(tmp_path)])


def test_send_with_images_folds_description_into_a_normal_turn(tmp_path, monkeypatch):
    import agent.loop as loop_mod

    _isolate_config(monkeypatch, tmp_path / "cfg")
    monkeypatch.setattr("core.model.is_model_pulled", lambda base_url, name: True)
    fake_vision = _FakeVisionLLM(description="the submit button overlaps the email field")
    monkeypatch.setattr(loop_mod, "get_chat_model", lambda **k: fake_vision)

    s = _session([[AIMessageChunk(content="I'll fix the overlapping button.")]])
    out = s.send_with_images("please fix this bug", [_fake_image(tmp_path)])

    assert out == "I'll fix the overlapping button."
    # the regular (scripted text) LLM's history has the augmented prompt,
    # including the vision model's description — not just the raw user text
    human_messages = [m for m in s.messages if isinstance(m, HumanMessage)]
    assert "please fix this bug" in human_messages[-1].content
    assert "the submit button overlaps the email field" in human_messages[-1].content


# ─── /vision <path> — attach an image by file path ────────────────────────────

def test_vision_command_no_arg_shows_usage(capsys):
    from agent.loop import _handle_vision_command

    _handle_vision_command("", SimpleNamespace(llm=None, tools=[]))
    assert "Usage: /vision" in capsys.readouterr().out


def test_vision_command_missing_file_shows_error(tmp_path, capsys):
    from agent.loop import _handle_vision_command

    session = SimpleNamespace(llm=None, tools=[], workspace=tmp_path, last_image_paths=[])
    _handle_vision_command(str(tmp_path / "nope.png"), session)
    assert "Image not found" in capsys.readouterr().out


def test_vision_command_success_triggers_two_model_handoff(tmp_path, monkeypatch, capsys):
    import agent.loop as loop_mod
    from agent.loop import _handle_vision_command

    _isolate_config(monkeypatch, tmp_path / "cfg")
    monkeypatch.setattr("core.model.is_model_pulled", lambda base_url, name: True)
    fake_vision = _FakeVisionLLM(description="a red error banner reading 'connection refused'")
    monkeypatch.setattr(loop_mod, "get_chat_model", lambda **k: fake_vision)

    s = _session([[AIMessageChunk(content="Looks like the API server isn't running.")]])
    image = _fake_image(tmp_path)
    _handle_vision_command(f"{image} why is this failing?", s)

    out = capsys.readouterr().out
    assert "Looks like the API server isn't running." in out
    human_messages = [m for m in s.messages if isinstance(m, HumanMessage)]
    assert "why is this failing?" in human_messages[-1].content
    assert "connection refused" in human_messages[-1].content


def test_vision_command_relative_path_resolves_against_workspace(tmp_path, monkeypatch):
    import agent.loop as loop_mod

    _isolate_config(monkeypatch, tmp_path / "cfg")
    monkeypatch.setattr("core.model.is_model_pulled", lambda base_url, name: True)
    fake_vision = _FakeVisionLLM()
    monkeypatch.setattr(loop_mod, "get_chat_model", lambda **k: fake_vision)

    ws = tmp_path / "workspace"
    ws.mkdir()
    _fake_image(ws, "shot.png")  # lives inside the workspace
    s = _session([[AIMessageChunk(content="done")]])
    s.workspace = ws

    loop_mod._handle_vision_command("shot.png", s)
    assert fake_vision.invoked_with is not None  # resolved and reached the vision model


# ─── /vision follow-up — reuse the last attached image(s), no re-attach needed ──

def test_vision_command_follow_up_reuses_last_images(tmp_path, monkeypatch):
    import agent.loop as loop_mod

    _isolate_config(monkeypatch, tmp_path / "cfg")
    monkeypatch.setattr("core.model.is_model_pulled", lambda base_url, name: True)
    fake_vision = _FakeVisionLLM()
    monkeypatch.setattr(loop_mod, "get_chat_model", lambda **k: fake_vision)

    s = _session([[AIMessageChunk(content="first answer")],
                  [AIMessageChunk(content="second answer")]])
    image = _fake_image(tmp_path)

    loop_mod._handle_vision_command(f"{image} what's wrong?", s)
    assert s.last_image_paths == [image]

    # Second call: "what about the corner?" isn't a valid path — should reuse
    # the same image rather than erroring "Image not found".
    loop_mod._handle_vision_command("what about the corner?", s)
    assert fake_vision.invoked_with is not None
    [message] = fake_vision.invoked_with
    assert message.content[0]["text"] == "what about the corner?"
    assert message.content[1]["type"] == "image_url"  # the same image, reused
    assert s.last_image_paths == [image]  # unchanged — still the same image


def test_vision_command_invalid_path_with_no_prior_image_still_errors(tmp_path, capsys):
    from agent.loop import _handle_vision_command

    session = SimpleNamespace(llm=None, tools=[], workspace=tmp_path, last_image_paths=[])
    _handle_vision_command("not a real path", session)
    assert "Image not found" in capsys.readouterr().out


def test_send_with_images_records_last_image_paths(tmp_path, monkeypatch):
    import agent.loop as loop_mod

    _isolate_config(monkeypatch, tmp_path / "cfg")
    monkeypatch.setattr("core.model.is_model_pulled", lambda base_url, name: True)
    monkeypatch.setattr(loop_mod, "get_chat_model", lambda **k: _FakeVisionLLM())

    s = _session([[AIMessageChunk(content="ok")]])
    image = _fake_image(tmp_path)
    assert s.last_image_paths == []
    s.send_with_images("hi", [image])
    assert s.last_image_paths == [image]


def test_vision_command_not_configured_shows_error_not_traceback(tmp_path, monkeypatch, capsys):
    from agent.loop import _handle_vision_command

    cfg = _isolate_config(monkeypatch, tmp_path / "cfg")
    cfg.raw()["vision"]["model"] = ""
    s = _session([[AIMessageChunk(content="ignored")]])
    _handle_vision_command(str(_fake_image(tmp_path)), s)
    assert "No vision model configured" in capsys.readouterr().out


# ─── /vision model — mirrors /model's picker, but for vision.model ─────────────

def test_vision_command_routes_model_subcommand_to_the_picker(tmp_path, monkeypatch):
    import agent.loop as loop_mod

    _isolate_config(monkeypatch, tmp_path)
    called = []
    monkeypatch.setattr(loop_mod, "_handle_vision_model_command", lambda arg="": called.append(arg))
    loop_mod._handle_vision_command("model", SimpleNamespace(llm=None, tools=[]))
    assert called == [""]


def test_vision_model_command_direct_name_switches(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_vision_model_command

    monkeypatch.setattr(model_mod, "is_model_pulled", lambda base_url, name: True)
    cfg = _isolate_config(monkeypatch, tmp_path)
    try:
        _handle_vision_model_command("qwen2.5vl:3b")
        assert cfg.raw()["vision"]["model"] == "qwen2.5vl:3b"
        out = capsys.readouterr().out
        assert "Set vision model" in out
        assert "saved as your default" in out
    finally:
        cfg.raw()["vision"]["model"] = "qwen2.5vl:7b"


def test_vision_model_command_picker_lists_and_selects(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_vision_model_command
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    fake_models = [
        {"name": "qwen2.5vl:7b", "size": 6_000_000_000},
        {"name": "llava:7b", "size": 4_700_000_000},
    ]
    monkeypatch.setattr(model_mod, "list_ollama_models", lambda base_url: fake_models)
    monkeypatch.setattr(model_mod, "is_model_pulled", lambda base_url, name: True)
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: "2"))  # 2nd entry
    try:
        _handle_vision_model_command()
        out = capsys.readouterr().out
        assert "(current)" in out
        assert cfg.raw()["vision"]["model"] == "llava:7b"
    finally:
        cfg.raw()["vision"]["model"] = "qwen2.5vl:7b"


def test_vision_model_command_blank_keeps_current(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_vision_model_command
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["vision"]["model"]
    monkeypatch.setattr(model_mod, "list_ollama_models",
                        lambda base_url: [{"name": original, "size": 1}])
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: ""))
    _handle_vision_model_command()
    assert cfg.raw()["vision"]["model"] == original
    assert "Kept current model" in capsys.readouterr().out


def test_vision_model_command_unreachable_ollama_shows_message(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_vision_model_command

    _isolate_config(monkeypatch, tmp_path)

    def boom(base_url):
        raise ConnectionError("no ollama")

    monkeypatch.setattr(model_mod, "list_ollama_models", boom)
    _handle_vision_model_command()
    assert "Couldn't reach Ollama" in capsys.readouterr().out


def test_vision_model_command_non_ollama_provider_shows_info(tmp_path, monkeypatch, capsys):
    from agent.loop import _handle_vision_model_command

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved_provider = cfg.raw()["model"]["provider"]
    try:
        cfg.raw()["model"]["provider"] = "openai_compatible"
        _handle_vision_model_command()
        out = capsys.readouterr().out
        assert "only supports discovering models via Ollama" in out
    finally:
        cfg.raw()["model"]["provider"] = saved_provider


def test_vision_model_command_other_entry_prompts_and_switches(tmp_path, monkeypatch):
    import core.model as model_mod
    from agent.loop import _handle_vision_model_command
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["vision"]["model"]
    monkeypatch.setattr(model_mod, "list_ollama_models",
                        lambda base_url: [{"name": original, "size": 1}])
    monkeypatch.setattr(model_mod, "is_model_pulled", lambda base_url, name: True)
    # Entry 1 is the installed/current vision model; "o" picks "Other…".
    answers = iter(["o", "minicpm-v:8b"])
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: next(answers)))
    try:
        _handle_vision_model_command()
        assert cfg.raw()["vision"]["model"] == "minicpm-v:8b"
    finally:
        cfg.raw()["vision"]["model"] = original


# ── run_agent_repl(continue_session=True) — the plain-REPL side of --continue ──

def test_run_agent_repl_continue_resumes_a_saved_conversation(monkeypatch, tmp_path, capsys):
    import agent.loop as loop_mod
    from rich.prompt import Prompt

    # Populate a saved transcript for this workspace directly via a session,
    # the same way an earlier `aicoder` run would have (through send()).
    prior = AgentSession(tmp_path)
    prior.llm = ScriptedLLM([[AIMessageChunk(content="Sure, on it.")]])
    prior.send("fix the login bug")

    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: "/exit"))
    loop_mod.run_agent_repl(tmp_path, continue_session=True)
    assert "Resumed the previous conversation" in capsys.readouterr().out


def test_run_agent_repl_continue_with_nothing_saved_starts_fresh(monkeypatch, tmp_path, capsys):
    import agent.loop as loop_mod
    from rich.prompt import Prompt

    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: "/exit"))
    loop_mod.run_agent_repl(tmp_path, continue_session=True)
    assert "No previous conversation found" in capsys.readouterr().out
