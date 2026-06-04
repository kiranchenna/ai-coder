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


# ─── fetch-error predicate ────────────────────────────────────────────────────

def test_is_fetch_error():
    from tools.web_tools import is_fetch_error
    assert is_fetch_error("[Error fetching https://x: timeout]") is True
    assert is_fetch_error("[Non-text content (image/png) at ...]") is True
    assert is_fetch_error("# Real Page\n\nActual content here.") is False


# ─── edit_file fuzzy matching ─────────────────────────────────────────────────

def test_locate_edit_exact():
    from agent.tools import locate_edit
    content = "line one\nline two\nline three\n"
    start, end, how = locate_edit(content, "line two")
    assert how == "exact"
    assert content[start:end] == "line two"


def test_locate_edit_ambiguous():
    from agent.tools import locate_edit
    res = locate_edit("x = 1\nx = 1\n", "x = 1")
    assert res[0] is None and res[1] == "ambiguous"


def test_locate_edit_trailing_whitespace_tolerant():
    from agent.tools import locate_edit
    # model added trailing spaces the file doesn't have → exact fails, fuzzy matches
    content = "def f():\n    return 1\n"
    start, end, how = locate_edit(content, "def f():\n    return 1   ")
    assert how == "fuzzy"
    assert content[start:end] == "def f():\n    return 1"


def test_locate_edit_indentation_tolerant():
    from agent.tools import locate_edit
    content = "class A:\n        x = 1\n"   # 8-space indent in file
    start, end, how = locate_edit(content, "class A:\n    x = 1")  # model used 4
    assert how == "fuzzy"
    assert content[start:end] == "class A:\n        x = 1"


def test_locate_edit_not_found():
    from agent.tools import locate_edit
    res = locate_edit("hello world\n", "nonexistent text")
    assert res[0] is None and res[1] == "not_found"


def test_reindent_to_match_single_line():
    from agent.tools import _reindent_to_match
    # file line is indented 4; model's new_string has no indent → re-indented
    assert _reindent_to_match("    return 1", "return 2") == "    return 2"


def test_reindent_to_match_preserves_relative_indent():
    from agent.tools import _reindent_to_match
    # block at 4-space in the file; model wrote it at column 0 → shift whole block
    out = _reindent_to_match("    def f():", "def f():\n    return 1")
    assert out == "    def f():\n        return 1"


def test_edit_file_fuzzy_preserves_indentation(tmp_path):
    from agent.tools import build_tools
    from core.config import get_config
    get_config().raw()["files"]["confirmation"] = "never"
    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    tools = {t.name: t for t in build_tools(tmp_path)}
    # model supplies old/new WITHOUT the file's 4-space indent (a common 7B error)
    tools["edit_file"].invoke({"path": "m.py", "old_string": "return 1", "new_string": "return 2"})
    assert (tmp_path / "m.py").read_text() == "def f():\n    return 2\n"


# ─── Cross-platform shell quoting ─────────────────────────────────────────────

def test_shell_quote_posix(monkeypatch):
    import agent.tools as t
    monkeypatch.setattr(t.sys, "platform", "linux")
    assert t._shell_quote("my message") == "'my message'"


def test_shell_quote_windows(monkeypatch):
    import agent.tools as t
    monkeypatch.setattr(t.sys, "platform", "win32")
    # cmd.exe needs double quotes, with embedded quotes doubled
    assert t._shell_quote('a "b" c') == '"a ""b"" c"'
    assert " " in t._shell_quote("with space") and t._shell_quote("with space").startswith('"')


# ─── Code index (find_symbol) ─────────────────────────────────────────────────

def test_build_symbol_index_python_and_js(tmp_path):
    from core.code_index import build_symbol_index
    (tmp_path / "a.py").write_text("import os\n\n\nclass Foo:\n    def bar(self):\n        pass\n")
    (tmp_path / "b.js").write_text("export function greet() {}\nconst add = (a, b) => a + b\n")
    idx = build_symbol_index(tmp_path)
    assert idx["Foo"][0] == {"file": "a.py", "line": 4, "kind": "class"}
    assert idx["bar"][0]["kind"] == "function" and idx["bar"][0]["line"] == 5
    assert idx["greet"][0]["kind"] == "function"
    assert idx["add"][0]["kind"] == "function"  # arrow function


def test_build_symbol_index_respects_ignore(tmp_path):
    from core.code_index import build_symbol_index
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("function leaked() {}\n")
    idx = build_symbol_index(tmp_path, ignore_dirs={"node_modules"})
    assert "leaked" not in idx


# ─── Large-file paging (read_file offset/limit) ───────────────────────────────

def test_read_file_paging(tmp_path):
    from agent.tools import build_tools
    big = "\n".join(f"line {i}" for i in range(1, 1201))   # 1200 lines
    (tmp_path / "big.txt").write_text(big)
    tools = {t.name: t for t in build_tools(tmp_path)}

    # default window on a large file
    first = tools["read_file"].invoke({"path": "big.txt"})
    assert first.startswith("line 1\n")
    assert "showing lines 1-500 of 1200" in first

    # explicit window
    mid = tools["read_file"].invoke({"path": "big.txt", "offset": 600, "limit": 3})
    assert "line 600\nline 601\nline 602" in mid
    assert "showing lines 600-602 of 1200" in mid


# ─── Project instructions (AICODER.md) ────────────────────────────────────────

def test_load_instructions_reads_project_file(tmp_path, monkeypatch):
    from agent.loop import _load_instructions
    monkeypatch.setattr("core.config.AICODER_HOME", tmp_path / "home")  # no global file
    (tmp_path / "AICODER.md").write_text("Always use type hints.\nPrefer pathlib.")
    out = _load_instructions(tmp_path)
    assert "Always use type hints." in out
    assert "From AICODER.md" in out


def test_load_instructions_absent_is_empty(tmp_path, monkeypatch):
    from agent.loop import _load_instructions
    monkeypatch.setattr("core.config.AICODER_HOME", tmp_path / "home")
    assert _load_instructions(tmp_path) == ""


def test_instructions_injected_into_system_prompt(tmp_path):
    from agent.prompts import system_prompt
    p = system_prompt(tmp_path, ["read_file"], project_instructions="Use 2-space indent.")
    assert "Project instructions" in p
    assert "Use 2-space indent." in p


# ─── History compaction (context management) ──────────────────────────────────

def _mk_session_with_history(messages, budget):
    from agent.loop import AgentSession
    s = AgentSession.__new__(AgentSession)
    s.messages = messages
    s._history_budget = budget
    return s


def test_compaction_noop_under_budget():
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    msgs = [SystemMessage(content="SYS"), HumanMessage(content="hi"), AIMessage(content="ok")]
    s = _mk_session_with_history(list(msgs), budget=10_000)
    s._compact_history_if_needed()
    assert s.messages == msgs  # unchanged


def test_compaction_summarizes_old_keeps_recent(monkeypatch):
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

    # force the summary model call into its fallback (no Ollama needed)
    def boom(*a, **k):
        raise RuntimeError("no model in test")
    monkeypatch.setattr("agent.loop.get_chat_model", boom)

    msgs = [
        SystemMessage(content="SYSTEM PROMPT"),
        HumanMessage(content="u1 " + "x" * 60),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "1"}]),
        ToolMessage(content="file contents " + "y" * 60, tool_call_id="1"),
        AIMessage(content="a1 " + "z" * 60),
        HumanMessage(content="u2 recent"),
        AIMessage(content="a2 recent answer"),
    ]
    s = _mk_session_with_history(list(msgs), budget=40)
    s._compact_history_if_needed()

    # system prompt preserved as message[0]
    assert s.messages[0].content == "SYSTEM PROMPT"
    # a summary note was inserted right after it
    assert isinstance(s.messages[1], SystemMessage)
    assert s.messages[1].content.startswith("[Summary of earlier conversation]")
    # the kept recent window starts at a user message (no orphaned ToolMessage)
    assert isinstance(s.messages[2], HumanMessage)
    assert not any(isinstance(m, ToolMessage) for m in s.messages[2:])
    # the latest turn is preserved verbatim
    assert s.messages[-1].content == "a2 recent answer"


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


# ─── Lint / typecheck detection ───────────────────────────────────────────────

def test_detect_lint_ruff_and_mypy(tmp_path):
    from core.project import detect_lint_commands
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n\n[tool.mypy]\n")
    labels = [label for _, label in detect_lint_commands(tmp_path)]
    assert "ruff" in labels and "mypy" in labels


def test_detect_lint_python_unconfigured_is_empty(tmp_path):
    from core.project import detect_lint_commands
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")  # no ruff/mypy
    assert detect_lint_commands(tmp_path) == []


def test_detect_lint_node_and_ts(tmp_path):
    import json as _json
    from core.project import detect_lint_commands
    (tmp_path / "package.json").write_text(_json.dumps({"scripts": {"lint": "eslint ."}}))
    (tmp_path / "tsconfig.json").write_text("{}")
    labels = [label for _, label in detect_lint_commands(tmp_path)]
    assert "npm run lint" in labels and "tsc" in labels


def test_detect_lint_rust_and_go(tmp_path):
    from core.project import detect_lint_commands
    (tmp_path / "Cargo.toml").write_text("[package]\n")
    assert [label for _, label in detect_lint_commands(tmp_path)] == ["clippy"]
    (tmp_path / "go.mod").write_text("module x\n")  # both now
    labels = [label for _, label in detect_lint_commands(tmp_path)]
    assert "clippy" in labels and "go vet" in labels


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
