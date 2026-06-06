"""Tests for the shell 'smart'-mode destructive-command heuristic.

This is the gate that decides whether a command auto-runs in smart mode, so its
false-negatives are a safety concern. These pin the cases the gate must catch
(including danger hidden behind shell operators) and the safe ones it must not
flag.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from tools.shell_tools import _is_destructive


@pytest.mark.parametrize("cmd", [
    "rm file.txt",
    "rm -rf build",
    "rm  -rf  build",            # extra spaces
    "sudo rm -rf /",
    "git reset --hard HEAD~1",
    "git clean -fd",
    "git push --force",
    "find . -name '*.py' -delete",
    "echo hi > important.txt",   # truncating redirect
    "cd src && rm -rf node_modules",   # danger behind &&
    "ls | xargs rm",                   # danger behind a pipe
    "true; rm -rf data",               # danger behind ;
    "python -c 'import shutil; shutil.rmtree(\"x\")'",
    "chmod -R 777 /etc",
    "dd if=/dev/zero of=/dev/sda",
])
def test_flags_destructive(cmd):
    assert _is_destructive(cmd) is True, cmd


@pytest.mark.parametrize("cmd", [
    "ls -la",
    "git status",
    "git diff HEAD",
    "npm test",
    "pytest -q",
    "cat file.txt >> log.txt",   # append, not truncate
    "python -m build 2>&1",      # fd-dup, not a file truncate
    "echo hello",
    "grep -rn foo src",
])
def test_allows_safe(cmd):
    assert _is_destructive(cmd) is False, cmd
