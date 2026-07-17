"""Tests for core/context.py — the workspace overview used in the system prompt."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.context import WorkspaceContext


def test_overview_counts_files_and_languages(tmp_path):
    (tmp_path / "a.py").write_text("print(1)")
    (tmp_path / "b.py").write_text("print(2)")
    (tmp_path / "c.js").write_text("console.log(1)")

    overview = WorkspaceContext(tmp_path).overview()
    assert "Source files: 3" in overview
    assert "Python (2)" in overview
    assert "JavaScript (1)" in overview


def test_overview_excludes_files_in_ignored_directories(tmp_path):
    (tmp_path / "real.py").write_text("print(1)")
    node_modules = tmp_path / "node_modules" / "some-pkg"
    node_modules.mkdir(parents=True)
    (node_modules / "index.js").write_text("module.exports = {}")

    overview = WorkspaceContext(tmp_path).overview()
    assert "Source files: 1" in overview
    assert "JavaScript" not in overview


def test_overview_never_descends_into_an_ignored_directory(tmp_path, monkeypatch):
    """The actual regression this guards against: WorkspaceContext.overview()
    used to walk with Path.rglob("*"), which has no way to skip descending
    into a directory — every file under node_modules/.git/venv/... got
    physically walked before ignore_dirs ever filtered anything out.
    Confirmed live: pointed at a large/unrelated tree (a home directory),
    that took well over a minute before the rest of startup (all fast) even
    got a turn. This asserts the *mechanism* (os.walk never yields a path
    inside the ignored directory at all), not just the resulting count —
    a fix that filters correctly but still walks everything would pass a
    count-only test while reintroducing the exact same slowdown."""
    import os

    (tmp_path / "real.py").write_text("print(1)")
    node_modules = tmp_path / "node_modules" / "some-pkg"
    node_modules.mkdir(parents=True)
    (node_modules / "index.js").write_text("module.exports = {}")

    import core.context as context_mod

    real_walk = os.walk
    visited_dirpaths = []

    def spying_walk(top, *args, **kwargs):
        for dirpath, dirnames, filenames in real_walk(top, *args, **kwargs):
            visited_dirpaths.append(dirpath)
            yield dirpath, dirnames, filenames

    monkeypatch.setattr(context_mod.os, "walk", spying_walk)
    WorkspaceContext(tmp_path).overview()

    assert not any("node_modules" in d for d in visited_dirpaths)


def test_overview_respects_aicoderignore(tmp_path):
    (tmp_path / "real.py").write_text("print(1)")
    (tmp_path / "generated.py").write_text("print(2)")
    (tmp_path / ".aicoderignore").write_text("generated.py\n")

    # Language count, not the raw file count: .aicoderignore itself is also
    # a (extensionless, unmatched-language) file under the workspace root,
    # so the total file count includes it regardless — the behavior this
    # test actually cares about is that the *ignored* .py file doesn't
    # inflate the Python count.
    overview = WorkspaceContext(tmp_path).overview()
    assert "Python (1)" in overview


def test_overview_caches_the_language_scan(tmp_path):
    """The file walk only needs to happen once per WorkspaceContext — a
    second overview() call (e.g. from a later system-prompt rebuild) must
    reuse the cached counts instead of re-walking the tree."""
    (tmp_path / "a.py").write_text("print(1)")
    ctx = WorkspaceContext(tmp_path)
    ctx.overview()
    (tmp_path / "b.py").write_text("print(2)")  # added after the first scan
    assert "Source files: 1" in ctx.overview()  # still the cached count
