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
    assert p == tmp_path / "docs" / "dev" / "01_requirements.md"
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
    assert (tmp_path / "docs" / "dev" / "01_requirements.md").exists()
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
    (tmp_path / "docs" / "dev" / "01_requirements.md").write_text("# Requirements\nBuild X.")

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
