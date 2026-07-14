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
    s = AgentSession(ws)               # offline: the chat model is constructed lazily
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
    path = s._session_log_path()
    assert path.exists()
    saved = json.loads(path.read_text())
    assert len(saved["raw_messages"]) == 2  # the human message + the AI's final answer
    assert len(saved["turns"]) == 1
    assert saved["turns"][0] == {
        "prompt": "hi", "actions": [], "answer": "Hello!", "completed": True,
    }


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
    # A *different* session's file — load_transcript() always excludes this
    # session's own (freshly created, still-empty) file when picking "the
    # latest other session" to resume.
    other = s._sessions_dir() / "2020-01-01T00-00-00-000000.json"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("not valid json{{{")
    assert s.load_transcript() is False
    assert len(s.messages) == 1


def test_transcript_persists_even_when_a_turn_is_interrupted():
    s = _session([[AIMessageChunk(content="ignored")]])
    chunks = [AIMessageChunk(content="Hel"), AIMessageChunk(content="lo!")]
    s.llm = _InterruptingLLM(s, chunks, interrupt_after=1)  # fires mid-stream
    out = s.send("hi")
    assert out == ""  # interrupted
    assert s._session_log_path().exists()  # still saved (best-effort, in a finally)


def test_send_records_tool_calls_as_actions():
    s = _session([
        [_native_tool_call("list_files", {"path": "."})],
        [AIMessageChunk(content="Done.")],
    ])
    s.send("what's in this repo?")
    assert len(s.session_turns) == 1
    actions = s.session_turns[0]["actions"]
    assert len(actions) == 1
    assert actions[0]["tool"] == "list_files"
    assert actions[0]["args"] == {"path": "."}
    assert actions[0]["diffs"] == []  # a read, not a write — no diff produced


def test_send_records_a_real_diff_when_a_file_is_written():
    s = _session([
        [_native_tool_call("write_file", {"path": "hello.py", "content": "print(1)\n"})],
        [AIMessageChunk(content="Created it.")],
    ])
    s.send("create hello.py")

    actions = s.session_turns[0]["actions"]
    assert len(actions) == 1
    assert actions[0]["tool"] == "write_file"
    [diff_entry] = actions[0]["diffs"]
    assert diff_entry["path"] == "hello.py"
    assert "+print(1)" in diff_entry["diff"]
    assert (s.workspace / "hello.py").read_text() == "print(1)\n"  # really written


def test_log_safe_truncates_long_strings_but_keeps_short_ones():
    from agent.loop import _log_safe

    short = "hello"
    long = "x" * 1000
    assert _log_safe(short) == short
    truncated = _log_safe(long, limit=500)
    assert truncated.startswith("x" * 500)
    assert "truncated" in truncated
    assert _log_safe({"a": long}, limit=500)["a"] == truncated
    assert _log_safe([long], limit=500)[0] == truncated


# ─── request_interrupt() — best-effort mid-turn cancellation (the TUI's Esc) ───

class _InterruptingLLM:
    """Fires request_interrupt() on the session partway through the stream —
    lets us test the interrupt check deterministically, without depending on
    a real model server's chunk timing."""

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
    from agent.loop import _switch_model

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved = dict(cfg.raw()["model"])
    cfg.raw()["model"]["base_url"] = "http://localhost:8080/v1"  # not LM Studio — no lms shellout
    session = SimpleNamespace(llm=None, tools=[])
    try:
        _switch_model("qwen2.5-coder-14b-instruct", session)
        assert cfg.raw()["model"]["name"] == "qwen2.5-coder-14b-instruct"
        assert session.llm is not None                     # rebuilt for the new model
        saved_on_disk = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert saved_on_disk["model"]["name"] == "qwen2.5-coder-14b-instruct"  # persisted
    finally:
        cfg.raw()["model"] = saved


def test_switch_model_lmstudio_loads_it(tmp_path, monkeypatch):
    import core.model as model_mod
    from agent.loop import _switch_model

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved = dict(cfg.raw()["model"])
    _set_lmstudio_provider(cfg)
    calls = []
    monkeypatch.setattr(model_mod, "switch_lmstudio_model", lambda name: calls.append(name))
    try:
        _switch_model("other-model", SimpleNamespace(llm=None, tools=[]))
        assert calls == ["other-model"]
    finally:
        cfg.raw()["model"] = saved


def test_switch_model_lmstudio_load_failure_warns_not_raises(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _switch_model

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved = dict(cfg.raw()["model"])
    _set_lmstudio_provider(cfg)

    def raise_not_downloaded(name):
        raise RuntimeError("not downloaded")

    monkeypatch.setattr(model_mod, "switch_lmstudio_model", raise_not_downloaded)
    try:
        _switch_model("missing-model", SimpleNamespace(llm=None, tools=[]))  # must not raise
        assert "Couldn't load 'missing-model' in LM Studio" in capsys.readouterr().out
    finally:
        cfg.raw()["model"] = saved


def test_model_command_direct_name_switches(tmp_path, monkeypatch, capsys):
    from agent.loop import _handle_model_command

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved = dict(cfg.raw()["model"])
    cfg.raw()["model"]["base_url"] = "http://localhost:8080/v1"
    try:
        _handle_model_command("qwen2.5-coder-14b-instruct", SimpleNamespace(llm=None, tools=[]))
        assert cfg.raw()["model"]["name"] == "qwen2.5-coder-14b-instruct"
        assert "saved as your default" in capsys.readouterr().out
    finally:
        cfg.raw()["model"] = saved


def test_model_command_custom_openai_endpoint_never_tries_lmstudio(tmp_path, monkeypatch, capsys):
    # Regression test: base_url must actually match LM Studio's default
    # (localhost:1234) before treating an openai_compatible config as LM
    # Studio — live-caught bug where "lms happens to be on this machine's
    # PATH" alone caused a *different* configured server's model picker to
    # silently show LM Studio's local models instead.
    import core.model as model_mod
    from agent.loop import _handle_model_command

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved = dict(cfg.raw()["model"])
    cfg.raw()["model"]["base_url"] = "http://localhost:8080/v1"  # NOT LM Studio's default
    monkeypatch.setattr(
        model_mod, "list_lmstudio_models",
        lambda **k: (_ for _ in ()).throw(AssertionError(
            "list_lmstudio_models must not be called for a non-LM-Studio endpoint")),
    )
    try:
        _handle_model_command("", SimpleNamespace(llm=None, tools=[]))
        out = capsys.readouterr().out
        assert "http://localhost:8080/v1" in out   # the generic panel
    finally:
        cfg.raw()["model"] = saved


def test_model_command_direct_name_still_works_for_custom_endpoint(tmp_path, monkeypatch, capsys):
    from agent.loop import _handle_model_command

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved = dict(cfg.raw()["model"])
    cfg.raw()["model"]["base_url"] = "http://localhost:8080/v1"
    try:
        _handle_model_command("gpt-4o-mini", SimpleNamespace(llm=None, tools=[]))
        assert cfg.raw()["model"]["name"] == "gpt-4o-mini"
        assert "saved as your default" in capsys.readouterr().out
    finally:
        cfg.raw()["model"] = saved


# ── /model — the LM Studio path (openai_compatible pointed at its default url) ──

def _set_lmstudio_provider(cfg, model_name="qwen2.5-coder-7b-instruct-mlx"):
    cfg.raw()["model"]["provider"] = "openai_compatible"
    cfg.raw()["model"]["base_url"] = "http://localhost:1234/v1"
    cfg.raw()["model"]["name"] = model_name


def test_model_command_lmstudio_lists_and_selects(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_model_command
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved = dict(cfg.raw()["model"])
    _set_lmstudio_provider(cfg, "current-model")
    fake_models = [
        {"name": "current-model", "size": 4_000_000_000, "vision": False},
        {"name": "other-model", "size": 9_000_000_000, "vision": False},
    ]
    monkeypatch.setattr(model_mod, "list_lmstudio_models", lambda **k: fake_models)
    monkeypatch.setattr(model_mod, "switch_lmstudio_model", lambda name: None)
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: "2"))
    try:
        _handle_model_command("", SimpleNamespace(llm=None, tools=[]))
        out = capsys.readouterr().out
        assert "Available models (via LM Studio)" in out
        assert cfg.raw()["model"]["name"] == "other-model"
    finally:
        cfg.raw()["model"] = saved


def test_model_command_lmstudio_no_models_shows_message(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_model_command

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved = dict(cfg.raw()["model"])
    _set_lmstudio_provider(cfg)
    monkeypatch.setattr(model_mod, "list_lmstudio_models", lambda **k: [])
    try:
        _handle_model_command("", SimpleNamespace(llm=None, tools=[]))
        assert "No models downloaded in LM Studio yet" in capsys.readouterr().out
    finally:
        cfg.raw()["model"] = saved


def test_model_command_lmstudio_unreachable_falls_back_to_generic_panel(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_model_command

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved = dict(cfg.raw()["model"])
    _set_lmstudio_provider(cfg)

    def raise_unreachable(**k):
        raise RuntimeError("'lms' isn't on your PATH")

    monkeypatch.setattr(model_mod, "list_lmstudio_models", raise_unreachable)
    try:
        _handle_model_command("", SimpleNamespace(llm=None, tools=[]))
        out = capsys.readouterr().out
        assert "Couldn't reach LM Studio" in out    # the generic status panel, not a crash
        assert "Available models (via LM Studio)" not in out
    finally:
        cfg.raw()["model"] = saved


def test_model_command_lmstudio_blank_keeps_current(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_model_command
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved = dict(cfg.raw()["model"])
    _set_lmstudio_provider(cfg, "current-model")
    monkeypatch.setattr(model_mod, "list_lmstudio_models",
                        lambda **k: [{"name": "current-model", "size": 1, "vision": False}])
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: ""))
    try:
        _handle_model_command("", SimpleNamespace(llm=None, tools=[]))
        assert cfg.raw()["model"]["name"] == "current-model"    # blank = cancel, no change
        assert "Kept current model" in capsys.readouterr().out
    finally:
        cfg.raw()["model"] = saved


def test_model_command_lmstudio_rejects_out_of_range_choice(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_model_command
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved = dict(cfg.raw()["model"])
    _set_lmstudio_provider(cfg, "current-model")
    monkeypatch.setattr(model_mod, "list_lmstudio_models",
                        lambda **k: [{"name": "current-model", "size": 1, "vision": False}])
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: "99"))
    try:
        _handle_model_command("", SimpleNamespace(llm=None, tools=[]))
        assert cfg.raw()["model"]["name"] == "current-model"    # out of range → no change
        assert "Invalid selection" in capsys.readouterr().out
    finally:
        cfg.raw()["model"] = saved


# ── /model's "Other…" entry — an exact LM Studio model id, no pull flow ─────────

def test_handle_custom_model_entry_blank_makes_no_change(tmp_path, monkeypatch, capsys):
    from agent.loop import _handle_custom_model_entry
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    original = cfg.raw()["model"]["name"]
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: ""))
    _handle_custom_model_entry(SimpleNamespace(llm=None, tools=[]))
    assert cfg.raw()["model"]["name"] == original
    assert "No change made" in capsys.readouterr().out


def test_handle_custom_model_entry_switches_directly(tmp_path, monkeypatch):
    from agent.loop import _handle_custom_model_entry
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved = dict(cfg.raw()["model"])
    cfg.raw()["model"]["base_url"] = "http://localhost:8080/v1"  # not LM Studio — no lms shellout
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: "some-model-id"))
    try:
        _handle_custom_model_entry(SimpleNamespace(llm=None, tools=[]))
        assert cfg.raw()["model"]["name"] == "some-model-id"
    finally:
        cfg.raw()["model"] = saved


def test_model_command_other_choice_routes_to_custom_entry(tmp_path, monkeypatch):
    import core.model as model_mod
    from agent.loop import _handle_model_command
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved = dict(cfg.raw()["model"])
    _set_lmstudio_provider(cfg, "current-model")
    monkeypatch.setattr(model_mod, "list_lmstudio_models",
                        lambda **k: [{"name": "current-model", "size": 1, "vision": False}])
    # First Prompt.ask call is the picker's "1-N/o" choice, second is the
    # custom-entry's model-id prompt.
    answers = iter(["o", "typed-model-id"])
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: next(answers)))
    try:
        _handle_model_command("", SimpleNamespace(llm=None, tools=[]))
        assert cfg.raw()["model"]["name"] == "typed-model-id"
    finally:
        cfg.raw()["model"] = saved


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


def test_repl_shows_devmode_banner_when_a_design_exists(monkeypatch, tmp_path):
    import io

    from rich.console import Console
    from rich.prompt import Prompt

    import agent.loop as loop_mod

    dev_dir = tmp_path / "docs" / "dev"
    dev_dir.mkdir(parents=True)
    (dev_dir / "state.json").write_text("{}")

    fake_console = Console(file=io.StringIO(), force_terminal=True, width=80)
    monkeypatch.setattr(loop_mod, "console", fake_console)
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: "/exit"))

    loop_mod.run_agent_repl(tmp_path)

    out = fake_console.file.getvalue()
    assert "Developer Mode design exists" in out
    assert "/dev status" in out


def test_repl_skips_devmode_banner_when_no_design_exists(monkeypatch, tmp_path):
    import io

    from rich.console import Console
    from rich.prompt import Prompt

    import agent.loop as loop_mod

    fake_console = Console(file=io.StringIO(), force_terminal=True, width=80)
    monkeypatch.setattr(loop_mod, "console", fake_console)
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: "/exit"))

    loop_mod.run_agent_repl(tmp_path)

    out = fake_console.file.getvalue()
    assert "Developer Mode design exists" not in out


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

    cfg = _isolate_config(monkeypatch, tmp_path / "cfg")
    cfg.raw()["model"]["base_url"] = "http://localhost:8080/v1"  # not LM Studio — no lms shellout
    cfg.raw()["vision"]["model"] = "some-vlm"
    fake_vision = _FakeVisionLLM()
    monkeypatch.setattr(loop_mod, "get_chat_model", lambda **k: fake_vision)

    s = _session([[AIMessageChunk(content="ignored")]])
    image = _fake_image(tmp_path)
    result = s.describe_images([image], "what's wrong here?")

    assert result == fake_vision.description
    [message] = fake_vision.invoked_with
    assert message.content[0] == {"type": "text", "text": "what's wrong here?"}
    assert message.content[1]["type"] == "image_url"
    # image_url must be an object ({"url": ...}), not a bare string — a bare
    # string is what ChatOllama tolerated but LM Studio's stricter
    # OpenAI-compatible validation rejected outright with a 400, live.
    assert message.content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_describe_images_no_vision_model_configured_raises(tmp_path, monkeypatch):
    cfg = _isolate_config(monkeypatch, tmp_path / "cfg")
    cfg.raw()["vision"]["model"] = ""

    s = _session([[AIMessageChunk(content="ignored")]])
    with pytest.raises(RuntimeError, match="No vision model configured"):
        s.describe_images([_fake_image(tmp_path)])


def test_describe_images_lmstudio_not_downloaded_raises(tmp_path, monkeypatch):
    cfg = _isolate_config(monkeypatch, tmp_path / "cfg")
    _set_lmstudio_provider(cfg)
    cfg.raw()["vision"]["model"] = "some-vlm"
    monkeypatch.setattr("core.model.is_lmstudio_model_downloaded", lambda name: False)

    s = _session([[AIMessageChunk(content="ignored")]])
    with pytest.raises(RuntimeError, match="isn't downloaded in LM Studio"):
        s.describe_images([_fake_image(tmp_path)])


def test_send_with_images_folds_description_into_a_normal_turn(tmp_path, monkeypatch):
    import agent.loop as loop_mod

    cfg = _isolate_config(monkeypatch, tmp_path / "cfg")
    cfg.raw()["model"]["base_url"] = "http://localhost:8080/v1"
    cfg.raw()["vision"]["model"] = "some-vlm"
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

    cfg = _isolate_config(monkeypatch, tmp_path / "cfg")
    cfg.raw()["model"]["base_url"] = "http://localhost:8080/v1"  # not LM Studio — no lms shellout
    cfg.raw()["vision"]["model"] = "some-vlm"
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

    cfg = _isolate_config(monkeypatch, tmp_path / "cfg")
    cfg.raw()["model"]["base_url"] = "http://localhost:8080/v1"  # not LM Studio — no lms shellout
    cfg.raw()["vision"]["model"] = "some-vlm"
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

    cfg = _isolate_config(monkeypatch, tmp_path / "cfg")
    cfg.raw()["model"]["base_url"] = "http://localhost:8080/v1"  # not LM Studio — no lms shellout
    cfg.raw()["vision"]["model"] = "some-vlm"
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

    cfg = _isolate_config(monkeypatch, tmp_path / "cfg")
    cfg.raw()["model"]["base_url"] = "http://localhost:8080/v1"  # not LM Studio — no lms shellout
    cfg.raw()["vision"]["model"] = "some-vlm"
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
    from agent.loop import _handle_vision_model_command

    cfg = _isolate_config(monkeypatch, tmp_path)
    cfg.raw()["model"]["base_url"] = "http://localhost:8080/v1"  # not LM Studio — no lms shellout
    try:
        _handle_vision_model_command("some-vlm")
        assert cfg.raw()["vision"]["model"] == "some-vlm"
        out = capsys.readouterr().out
        assert "Set vision model" in out
        assert "saved as your default" in out
    finally:
        cfg.raw()["vision"]["model"] = ""


def test_vision_model_command_lmstudio_blank_keeps_current(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_vision_model_command
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved_provider = dict(cfg.raw()["model"])
    saved_vision = cfg.raw()["vision"]["model"]
    _set_lmstudio_provider(cfg)
    cfg.raw()["vision"]["model"] = "current-vlm"
    monkeypatch.setattr(model_mod, "list_lmstudio_models",
                        lambda vision_only=False: [{"name": "current-vlm", "size": 1, "vision": True}])
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: ""))
    try:
        _handle_vision_model_command()
        assert cfg.raw()["vision"]["model"] == "current-vlm"
        assert "Kept current model" in capsys.readouterr().out
    finally:
        cfg.raw()["model"] = saved_provider
        cfg.raw()["vision"]["model"] = saved_vision


def test_vision_model_command_lmstudio_unreachable_shows_message(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_vision_model_command

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved_provider = dict(cfg.raw()["model"])
    _set_lmstudio_provider(cfg)

    def boom(vision_only=False):
        raise RuntimeError("'lms' isn't on your PATH")

    monkeypatch.setattr(model_mod, "list_lmstudio_models", boom)
    try:
        _handle_vision_model_command()
        assert "Couldn't reach LM Studio" in capsys.readouterr().out
    finally:
        cfg.raw()["model"] = saved_provider


def test_vision_model_command_non_lmstudio_endpoint_shows_info(tmp_path, monkeypatch, capsys):
    from agent.loop import _handle_vision_model_command

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved_provider = cfg.raw()["model"]["provider"]
    try:
        cfg.raw()["model"]["provider"] = "openai_compatible"
        cfg.raw()["model"]["base_url"] = "https://api.openai.com/v1"  # not LM Studio's default
        _handle_vision_model_command()
        out = capsys.readouterr().out
        assert "Couldn't reach LM Studio" in out
    finally:
        cfg.raw()["model"]["provider"] = saved_provider


def test_vision_model_command_lmstudio_lists_vision_only_and_selects(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_vision_model_command

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved_provider = dict(cfg.raw()["model"])
    saved_vision = cfg.raw()["vision"]["model"]
    _set_lmstudio_provider(cfg)
    cfg.raw()["vision"]["model"] = "current-vlm"

    def fake_list(vision_only=False):
        assert vision_only is True   # the picker must filter to vision-capable models only
        return [{"name": "current-vlm", "size": 1, "vision": True},
                {"name": "other-vlm", "size": 2, "vision": True}]

    monkeypatch.setattr(model_mod, "list_lmstudio_models", fake_list)
    monkeypatch.setattr(model_mod, "is_lmstudio_model_downloaded", lambda name: True)
    from rich.prompt import Prompt
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: "2"))
    try:
        _handle_vision_model_command()
        out = capsys.readouterr().out
        assert "Available vision models (via LM Studio)" in out
        assert cfg.raw()["vision"]["model"] == "other-vlm"
    finally:
        cfg.raw()["model"] = saved_provider
        cfg.raw()["vision"]["model"] = saved_vision


def test_vision_model_command_lmstudio_no_vision_models_shows_message(tmp_path, monkeypatch, capsys):
    import core.model as model_mod
    from agent.loop import _handle_vision_model_command

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved_provider = dict(cfg.raw()["model"])
    _set_lmstudio_provider(cfg)
    monkeypatch.setattr(model_mod, "list_lmstudio_models", lambda vision_only=False: [])
    try:
        _handle_vision_model_command()
        assert "No vision-capable models downloaded in LM Studio yet" in capsys.readouterr().out
    finally:
        cfg.raw()["model"] = saved_provider


def test_vision_model_command_other_entry_prompts_and_switches(tmp_path, monkeypatch):
    import core.model as model_mod
    from agent.loop import _handle_vision_model_command
    from rich.prompt import Prompt

    cfg = _isolate_config(monkeypatch, tmp_path)
    saved_provider = dict(cfg.raw()["model"])
    saved_vision = cfg.raw()["vision"]["model"]
    _set_lmstudio_provider(cfg)
    cfg.raw()["vision"]["model"] = "current-vlm"
    monkeypatch.setattr(model_mod, "list_lmstudio_models",
                        lambda vision_only=False: [{"name": "current-vlm", "size": 1, "vision": True}])
    # Entry 1 is the installed/current vision model; "o" picks "Other…".
    answers = iter(["o", "some-other-vlm"])
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: next(answers)))
    try:
        _handle_vision_model_command()
        assert cfg.raw()["vision"]["model"] == "some-other-vlm"
    finally:
        cfg.raw()["model"] = saved_provider
        cfg.raw()["vision"]["model"] = saved_vision


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


# ── /history — browse and view past sessions ────────────────────────────────────

def test_history_command_no_sessions_shows_message(tmp_path, capsys):
    from agent.loop import _handle_history_command

    session = SimpleNamespace(session_id="whatever")
    _handle_history_command("", session, tmp_path)
    assert "No saved sessions yet" in capsys.readouterr().out


def test_history_command_lists_sessions_with_prompt_and_files_touched(capsys):
    s = _session([
        [_native_tool_call("write_file", {"path": "app.py", "content": "x = 1\n"})],
        [AIMessageChunk(content="Done.")],
    ])
    s.send("fix the bug in app.py")

    from agent.loop import _handle_history_command
    _handle_history_command("", s, s.workspace)
    out = capsys.readouterr().out
    assert "fix the bug in app.py" in out
    assert "1 turn" in out
    assert "app.py" in out
    assert "(current)" in out  # the only session, and it's this one


def test_history_command_invalid_index_shows_usage(capsys):
    s = _session([[AIMessageChunk(content="hi")]])
    s.send("hello")

    from agent.loop import _handle_history_command
    _handle_history_command("99", s, s.workspace)
    assert "Usage: /history" in capsys.readouterr().out


def test_history_command_detail_view_shows_prompt_actions_diffs_answer(capsys):
    s = _session([
        [_native_tool_call("write_file", {"path": "app.py", "content": "x = 1\n"})],
        [AIMessageChunk(content="Fixed the bug.")],
    ])
    s.send("fix the bug in app.py")

    from agent.loop import _handle_history_command
    _handle_history_command("1", s, s.workspace)
    out = capsys.readouterr().out
    assert "fix the bug in app.py" in out
    assert "write_file" in out
    assert "+x = 1" in out  # the real diff, rendered
    assert "Fixed the bug." in out


def test_history_command_detail_view_handles_unreadable_file(tmp_path, capsys):
    from agent.loop import _render_session_detail

    bad = tmp_path / "bad.json"
    bad.write_text("not valid json{{{")
    _render_session_detail(bad)
    assert "Couldn't read that session" in capsys.readouterr().out
