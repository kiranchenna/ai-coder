"""Tests for Developer Mode (devmode) — engine logic, no model calls."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import devmode.session as S
from devmode.phases import PHASES, PHASES_BY_ID
from devmode.session import DevSession


def test_init_all_phases_pending(tmp_path):
    ds = DevSession(tmp_path, "build a todo app")
    assert ds.state["idea"] == "build a todo app"
    assert all(ds.state["phases"][p.id]["status"] == "pending" for p in PHASES)


def test_write_doc_and_conventions_artifacts(tmp_path):
    ds = DevSession(tmp_path, "x")
    p = ds._write_artifact(PHASES_BY_ID["requirements"], "Decision body", "transcript")
    assert p == tmp_path / "docs" / "dev" / "03_requirements.md"
    assert "Decision body" in p.read_text()

    # conventions phase writes to AICODER.md in the workspace root
    c = ds._write_artifact(PHASES_BY_ID["conventions"], "Use snake_case", "t")
    assert c == tmp_path / "AICODER.md"
    assert "snake_case" in c.read_text()


def test_state_persists_and_reloads(tmp_path):
    ds = DevSession(tmp_path, "x")
    ds._set_status("requirements", "done")
    reloaded = DevSession(tmp_path)
    assert reloaded.state["phases"]["requirements"]["status"] == "done"
    assert reloaded.state["idea"] == "x"


def test_prior_artifacts_grounds_later_phases(tmp_path):
    ds = DevSession(tmp_path, "x")
    ds._write_artifact(PHASES_BY_ID["requirements"], "REQ-BODY-MARKER", "t")
    assert "REQ-BODY-MARKER" in ds._prior_artifacts()


def test_run_marches_through_all_phases(tmp_path, monkeypatch):
    monkeypatch.setattr(S.Confirm, "ask", lambda *a, **k: True)  # auto-yes
    ds = DevSession(tmp_path, "x")

    def fake_phase(spec):
        ds._write_artifact(spec, f"decision for {spec.id}", "t")
        return "done"

    monkeypatch.setattr(ds, "_run_phase", fake_phase)
    ds.run()

    assert all(ds.state["phases"][p.id]["status"] == "done" for p in PHASES)
    assert (tmp_path / "docs" / "dev" / "03_requirements.md").exists()
    assert (tmp_path / "AICODER.md").exists()


def test_resume_skips_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(S.Confirm, "ask", lambda *a, **k: True)
    ds = DevSession(tmp_path, "x")
    ds._set_status("requirements", "done")
    ran = []
    monkeypatch.setattr(ds, "_run_phase", lambda spec: (ran.append(spec.id), "done")[1])
    ds.run(resume=True)
    assert "requirements" not in ran               # already done → not re-run
    assert "architecture" in ran                   # pending → run


# ─── Build hand-off ───────────────────────────────────────────────────────────

def test_parse_files_from_messy_output():
    from devmode.build import _parse_files
    out = 'Sure: [{"path": "app.py", "purpose": "main"}, {"path": "util.py"}] done'
    files = _parse_files(out)
    assert [f["path"] for f in files] == ["app.py", "util.py"]


def test_build_generates_pending_files(tmp_path, monkeypatch):
    import devmode.build as B
    from core.config import get_config
    get_config().raw()["files"]["confirmation"] = "never"
    get_config().raw()["shell"]["confirmation"] = "never"

    (tmp_path / "docs" / "dev").mkdir(parents=True)
    (tmp_path / "docs" / "dev" / "03_requirements.md").write_text("# Requirements\nBuild X.")

    b = B.Builder(tmp_path)
    b._save_plan({"idea": "x", "files": [
        {"path": "app.py", "purpose": "entry", "status": "pending"},
        {"path": "pkg/util.py", "purpose": "helpers", "status": "done"},  # already done → skipped
    ]})
    gen = []
    monkeypatch.setattr(b, "_generate_file",
                        lambda entry, spec, conv, completed: (gen.append(entry["path"]),
                                                              f"# {entry['path']}\nx = 1\n")[1])
    monkeypatch.setattr(b, "_verify", lambda: None)
    monkeypatch.setattr(B.Confirm, "ask", lambda *a, **k: True)
    b.build()

    assert (tmp_path / "app.py").read_text().startswith("# app.py")
    assert gen == ["app.py"]                        # the done file was skipped
    assert all(f["status"] == "done" for f in b._load_plan()["files"])


# ─── Auto-resync (revisit) ────────────────────────────────────────────────────

import json as _json


def _setup_built_session(tmp_path, monkeypatch, old_decision, new_decision):
    monkeypatch.setattr(S.Confirm, "ask", lambda *a, **k: True)
    ds = DevSession(tmp_path, "x")
    spec = PHASES_BY_ID["requirements"]
    ds._write_artifact(spec, old_decision, "t")     # old artifact on disk
    (tmp_path / "docs" / "dev" / "build_plan.json").write_text(
        _json.dumps({"files": [{"path": "a.py", "status": "done"}]}))
    # revisit re-runs the phase → writes the new decision
    monkeypatch.setattr(ds, "_run_phase",
                        lambda s: (ds._write_artifact(s, new_decision, "t"), "done")[1])
    calls = {}
    monkeypatch.setattr("devmode.resync.resync",
                        lambda ws, title, old, new: calls.update(title=title, old=old, new=new))
    return ds, calls


def test_revisit_resyncs_when_decision_changes(tmp_path, monkeypatch):
    ds, calls = _setup_built_session(tmp_path, monkeypatch,
                                     "hello returns 'Hello, X!'", "hello returns 'Hi, X!'")
    ds.revisit("requirements")
    assert calls.get("title") == "Requirements"
    assert "Hello" in calls["old"] and "Hi" in calls["new"]


def test_revisit_no_resync_when_unchanged(tmp_path, monkeypatch):
    ds, calls = _setup_built_session(tmp_path, monkeypatch, "same decision", "same decision")
    ds.revisit("requirements")
    assert calls == {}                              # no change → no resync


def test_revisit_no_resync_without_a_build(tmp_path, monkeypatch):
    monkeypatch.setattr(S.Confirm, "ask", lambda *a, **k: True)
    ds = DevSession(tmp_path, "x")
    ds._write_artifact(PHASES_BY_ID["requirements"], "old", "t")
    monkeypatch.setattr(ds, "_run_phase",
                        lambda s: (ds._write_artifact(s, "new", "t"), "done")[1])
    calls = {}
    monkeypatch.setattr("devmode.resync.resync", lambda *a: calls.update(hit=True))
    ds.revisit("requirements")                       # no build_plan.json → skip resync
    assert calls == {}


# ─── Brownfield (existing-repo awareness) ─────────────────────────────────────

def test_has_existing_code(tmp_path):
    ds = DevSession(tmp_path, "x")
    assert ds._has_existing_code() is False
    (tmp_path / "app.py").write_text("def f():\n    pass\n")
    assert ds._has_existing_code() is True


def test_has_existing_code_ignores_design_and_docs(tmp_path):
    ds = DevSession(tmp_path, "x")
    (tmp_path / "docs" / "dev").mkdir(parents=True)
    (tmp_path / "docs" / "dev" / "03_requirements.md").write_text("# x")
    (tmp_path / "README.md").write_text("# readme")
    assert ds._has_existing_code() is False          # docs/markdown don't count as code


def test_sample_code_collects_sources(tmp_path):
    ds = DevSession(tmp_path, "x")
    (tmp_path / "a.py").write_text("print('hello-sample')")
    s = ds._sample_code()
    assert "a.py" in s and "hello-sample" in s


def test_system_prompt_includes_brownfield_sections(tmp_path):
    ds = DevSession(tmp_path, "x")
    p = ds._system_prompt(PHASES_BY_ID["architecture"], "", "",
                          repo="REPO-OVERVIEW", docs="DOC-X", seed="SEED-Y")
    assert "Existing codebase" in p and "REPO-OVERVIEW" in p
    assert "DOC-X" in p and "SEED-Y" in p


# ─── New roles (PM, Market Analyst, Tech Writer, Design Reviewer) ──────────────

def test_new_roles_and_review_phase_present():
    roles = {p.role for p in PHASES}
    assert {"Product Manager", "Market Analyst", "Technical Writer", "Design Reviewer"} <= roles
    assert len(PHASES) == 14
    review = [p for p in PHASES if p.kind == "review"]
    assert len(review) == 1 and review[0].id == "review"
    assert PHASES[-1].kind == "review"               # review runs last (before build)


def test_summarize_reflection_does_draft_then_improve(tmp_path, monkeypatch):
    from core.config import get_config
    get_config().raw().setdefault("devmode", {})["reflect"] = True
    calls = []
    monkeypatch.setattr(S, "_stream",
                        lambda msgs, precise=False: (calls.append(1),
                                                     "DRAFT" if len(calls) == 1 else "IMPROVED")[1])
    ds = DevSession(tmp_path, "x")
    out = ds._summarize(PHASES_BY_ID["requirements"], [])
    assert len(calls) == 2 and out == "IMPROVED"      # draft then refine


def test_summarize_single_pass_when_reflection_off(tmp_path, monkeypatch):
    from core.config import get_config
    get_config().raw().setdefault("devmode", {})["reflect"] = False
    calls = []
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: (calls.append(1), "DRAFT")[1])
    ds = DevSession(tmp_path, "x")
    out = ds._summarize(PHASES_BY_ID["requirements"], [])
    get_config().raw()["devmode"]["reflect"] = True   # restore for other tests
    assert len(calls) == 1 and out == "DRAFT"


def test_review_artifact_excluded_from_design_context(tmp_path):
    ds = DevSession(tmp_path, "x")
    review = PHASES_BY_ID["review"]
    (tmp_path / "docs" / "dev").mkdir(parents=True)
    (tmp_path / "docs" / "dev" / review.filename).write_text("# Design Review\nREVIEW-MARKER")
    ds._write_artifact(PHASES_BY_ID["requirements"], "REQ-MARKER", "t")
    ctx = ds._prior_artifacts()
    assert "REQ-MARKER" in ctx and "REVIEW-MARKER" not in ctx
