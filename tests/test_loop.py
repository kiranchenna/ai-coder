"""Integration tests for the agent loop (AgentSession.send) using a scripted LLM."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
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
