"""Tests for the hooks system."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.hooks import HookRunner


def test_pre_tool_use_blocks_on_nonzero():
    hooks = {"PreToolUse": [{"matcher": "run_shell", "command": "exit 1"}]}
    r = HookRunner(hooks)
    assert r.pre_tool_use("run_shell", {}, Path(".")) is not None      # blocked
    assert r.pre_tool_use("read_file", {}, Path(".")) is None          # matcher misses


def test_pre_tool_use_allows_on_zero():
    r = HookRunner({"PreToolUse": [{"matcher": "*", "command": "exit 0"}]})
    assert r.pre_tool_use("write_file", {"path": "x"}, Path(".")) is None


def test_pre_tool_use_block_reason_is_hook_output():
    r = HookRunner({"PreToolUse": [{"command": "echo nope; exit 2"}]})
    reason = r.pre_tool_use("anything", {}, Path("."))
    assert "nope" in reason


def test_post_tool_use_runs_and_can_have_side_effects(tmp_path):
    marker = tmp_path / "ran"
    r = HookRunner({"PostToolUse": [
        {"matcher": "edit_file", "command": f"touch {marker}"}
    ]})
    r.post_tool_use("edit_file", {}, "ok", tmp_path)
    assert marker.exists()


def test_post_tool_use_matcher_misses(tmp_path):
    marker = tmp_path / "ran"
    r = HookRunner({"PostToolUse": [{"matcher": "edit_file", "command": f"touch {marker}"}]})
    r.post_tool_use("read_file", {}, "ok", tmp_path)
    assert not marker.exists()


def test_stop_hook_runs(tmp_path):
    marker = tmp_path / "stopped"
    r = HookRunner({"Stop": [{"command": f"touch {marker}"}]})
    r.stop(tmp_path)
    assert marker.exists()


def test_no_hooks_is_safe():
    r = HookRunner({})
    assert r.has_any() is False
    assert r.pre_tool_use("x", {}, Path(".")) is None
    assert r.post_tool_use("x", {}, "r", Path(".")) == ""
