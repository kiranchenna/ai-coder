"""Tests for tools/file_tools.py"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile
from tools.file_tools import (
    resolve, read_file, write_file, backup_file,
    generate_diff, file_tree, search_in_files,
)


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


# ─── resolve ──────────────────────────────────────────────────────────────────

def test_resolve_simple(workspace):
    p = resolve(workspace, "foo.py")
    assert p == workspace / "foo.py"


def test_resolve_escapes_workspace(workspace):
    with pytest.raises(PermissionError):
        resolve(workspace, "../../etc/passwd")


# ─── read_file / write_file ───────────────────────────────────────────────────

def test_write_and_read(workspace):
    write_file(workspace, "hello.txt", "world")
    assert read_file(workspace, "hello.txt") == "world"


def test_read_missing_file(workspace):
    with pytest.raises(FileNotFoundError):
        read_file(workspace, "nonexistent.txt")


def test_write_creates_directories(workspace):
    write_file(workspace, "a/b/c/deep.py", "# deep")
    assert (workspace / "a" / "b" / "c" / "deep.py").exists()


# ─── backup_file ─────────────────────────────────────────────────────────────

def test_backup_creates_bak(workspace):
    write_file(workspace, "main.py", "original content")
    bak = backup_file(workspace, "main.py")
    assert bak is not None
    assert bak.exists()
    assert bak.read_text() == "original content"


def test_backup_nonexistent_returns_none(workspace):
    result = backup_file(workspace, "missing.py")
    assert result is None


def test_backup_does_not_modify_original(workspace):
    write_file(workspace, "app.py", "hello")
    backup_file(workspace, "app.py")
    assert read_file(workspace, "app.py") == "hello"


# ─── generate_diff ────────────────────────────────────────────────────────────

def test_diff_identical_files():
    diff = generate_diff("same content\n", "same content\n", "file.py")
    assert diff.strip() == ""


def test_diff_changed_file():
    diff = generate_diff("old line\n", "new line\n", "file.py")
    assert "-old line" in diff
    assert "+new line" in diff


def test_diff_new_file():
    diff = generate_diff("", "new content\n", "file.py")
    assert "+new content" in diff


# ─── file_tree ────────────────────────────────────────────────────────────────

def test_file_tree_lists_files(workspace):
    (workspace / "main.py").write_text("x")
    (workspace / "utils.py").write_text("x")
    tree = file_tree(workspace, max_depth=2)
    assert "main.py" in tree
    assert "utils.py" in tree


def test_file_tree_ignores_dirs(workspace):
    venv = workspace / "venv"
    venv.mkdir()
    (venv / "something.py").write_text("x")
    tree = file_tree(workspace, ignore_dirs=["venv"], max_depth=2)
    assert "venv" not in tree


# ─── search_in_files ─────────────────────────────────────────────────────────

def test_search_finds_match(workspace):
    write_file(workspace, "main.py", "def hello_world():\n    pass\n")
    results = search_in_files(workspace, "hello_world")
    assert len(results) == 1
    assert results[0]["line_number"] == 1


def test_search_no_match(workspace):
    write_file(workspace, "main.py", "print('hello')")
    results = search_in_files(workspace, "nonexistent_token_xyz")
    assert results == []


def test_search_case_insensitive(workspace):
    write_file(workspace, "app.py", "class MyClass:")
    results = search_in_files(workspace, "myclass", case_sensitive=False)
    assert len(results) == 1
