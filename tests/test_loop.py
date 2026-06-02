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
