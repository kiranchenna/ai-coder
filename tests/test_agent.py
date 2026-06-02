"""Unit tests for the AICoder v3 agent core (pure logic, no model calls)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model import balanced_json_arrays
from core.model import balanced_json_objects as _balanced_json_objects
from core.model import extract_text_tool_calls as _extract_text_tool_calls
from agent.loop import _is_actionable_tool_message
from agent.planner import Planner, _parse_tasks
from core.project import detect_test_command
from memory.project import ProjectMemory
from rag.store import chunk_text


# ─── Text tool-call recovery (the fix for models that narrate calls as text) ──

def test_balanced_json_objects_nested():
    objs = _balanced_json_objects('pre {"a": {"b": 1}} mid {"c": 2} end')
    assert objs == ['{"a": {"b": 1}}', '{"c": 2}']


def test_balanced_json_objects_braces_inside_strings():
    s = '{"content": "has a } and a { inside"}'
    assert _balanced_json_objects("noise " + s + " noise") == [s]


def test_extract_text_tool_calls_fenced():
    content = (
        '```json\n{"name": "write_file", "arguments": '
        '{"path": "a.py", "content": "x"}}\n```'
    )
    assert _extract_text_tool_calls(content) == [
        {"name": "write_file", "args": {"path": "a.py", "content": "x"}, "id": ""}
    ]


def test_extract_text_tool_calls_param_aliases():
    assert _extract_text_tool_calls('{"name": "run_tests", "parameters": {}}')[0]["name"] == "run_tests"
    assert _extract_text_tool_calls('{"name": "x", "args": {"a": 1}}')[0]["args"] == {"a": 1}


def test_extract_text_tool_calls_ignores_plain_json():
    assert _extract_text_tool_calls('{"foo": 1, "bar": 2}') == []
    assert _extract_text_tool_calls("no json here at all") == []


def test_extract_text_tool_calls_null_args():
    # explicit null arguments (a zero-arg tool) must not be dropped
    calls = _extract_text_tool_calls('{"name": "run_tests", "arguments": null}')
    assert calls == [{"name": "run_tests", "args": {}, "id": ""}]


# ─── Actionable-vs-illustrative tool-call heuristic ───────────────────────────

def test_is_actionable_real_call_with_preamble():
    # short prose preamble + a real call → actionable
    content = "Let me create the file.\n```json\n" + (
        '{"name": "write_file", "arguments": {"path": "a.py", "content": "'
        + "x" * 300 + '"}}'
    ) + "\n```"
    assert _is_actionable_tool_message(content) is True


def test_is_not_actionable_example_in_long_prose():
    # tiny example JSON buried in a long explanation → not actionable
    content = (
        "Here is how the tool works. " + "blah " * 120
        + '{"name": "write_file", "args": {"p": 1}} '
        + "and that is roughly the idea."
    )
    assert _is_actionable_tool_message(content) is False


# ─── balanced_json_arrays ─────────────────────────────────────────────────────

def test_balanced_json_arrays_string_aware():
    arrays = balanced_json_arrays('x ["a", "b]c"] y [1, 2]')
    assert arrays == ['["a", "b]c"]', "[1, 2]"]


# ─── RAG chunking ─────────────────────────────────────────────────────────────

def test_chunk_text_overlapping():
    chunks = chunk_text("x" * 3000, size=1000, overlap=100)
    assert len(chunks) >= 3
    assert all(len(c) <= 1000 for c in chunks)


def test_chunk_text_edge_cases():
    assert chunk_text("") == []
    assert chunk_text("short") == ["short"]


# ─── Planner JSON parsing ─────────────────────────────────────────────────────

def test_parse_tasks_extracts_array():
    tasks = _parse_tasks('Sure! [{"title": "A", "description": "x"}, {"title": "B"}] done')
    assert len(tasks) == 2
    assert tasks[0]["title"] == "A"


def test_parse_tasks_no_array():
    assert _parse_tasks("there is no json array here") == []


def test_parse_tasks_with_stray_brackets_in_prose():
    # A stray bracket before the real array used to break the greedy regex.
    text = 'First do step [1], then: [{"title": "A", "description": "x"}]'
    tasks = _parse_tasks(text)
    assert len(tasks) == 1 and tasks[0]["title"] == "A"


def test_has_active_plan_in_progress_and_missing_status(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.MEMORY_DIR", tmp_path)
    # planner.MEMORY_DIR was imported by name, patch there too
    monkeypatch.setattr("agent.planner.MEMORY_DIR", tmp_path)

    class _S:
        def send(self, m): ...

    p = Planner(tmp_path, _S())
    # in_progress should count as active (was only checking 'pending')
    p.save({"goal": "g", "tasks": [{"id": 1, "title": "t", "status": "in_progress"}]})
    assert p.has_active_plan() is True
    # a task missing 'status' must not raise
    p.save({"goal": "g", "tasks": [{"id": 1, "title": "t"}]})
    assert p.has_active_plan() is True


# ─── Project-type detection ───────────────────────────────────────────────────

def test_detect_pytest(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    cmd, label = detect_test_command(tmp_path)
    assert label == "pytest"
    assert "pytest" in cmd


def test_detect_node(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}')
    assert detect_test_command(tmp_path) == ("npm test", "npm test")


def test_detect_cargo(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]\n")
    assert detect_test_command(tmp_path) == ("cargo test", "cargo test")


def test_detect_none(tmp_path):
    (tmp_path / "readme.txt").write_text("hi")
    assert detect_test_command(tmp_path) is None


# ─── Persistent project memory ────────────────────────────────────────────────

def test_project_memory_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("memory.project.MEMORY_DIR", tmp_path / "mem")
    pm = ProjectMemory(tmp_path)

    entry = pm.add("Use argon2 for hashing", "decision")
    assert entry["category"] == "decision"
    assert "argon2" in pm.render()
    assert pm.search("argon2")


def test_project_memory_dedups(tmp_path, monkeypatch):
    monkeypatch.setattr("memory.project.MEMORY_DIR", tmp_path / "mem")
    pm = ProjectMemory(tmp_path)
    pm.add("API base path is /api/v1", "convention")
    pm.add("API base path is /api/v1", "convention")
    assert len(pm.all()) == 1
