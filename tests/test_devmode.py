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


def test_generate_file_self_reviews(tmp_path, monkeypatch):
    import devmode.build as B
    from core.config import get_config
    get_config().raw()["devmode"]["build_review"] = True

    seq = iter(["def f():\n    pass  # TODO\n", "def f():\n    return 42\n"])
    monkeypatch.setattr(B, "_stream", lambda msgs, precise=False: next(seq))
    b = B.Builder(tmp_path)
    out = b._generate_file({"path": "f.py", "purpose": "do f"}, "spec", "conv", [])
    assert out == "def f():\n    return 42"            # the reviewed/fixed version won


def test_generate_file_review_keeps_draft_if_truncated(tmp_path, monkeypatch):
    import devmode.build as B
    from core.config import get_config
    get_config().raw()["devmode"]["build_review"] = True

    draft = "def f():\n    return 1\n\n" * 5
    seq = iter([draft, "def f"])                       # degenerate/truncated review
    monkeypatch.setattr(B, "_stream", lambda msgs, precise=False: next(seq))
    b = B.Builder(tmp_path)
    out = b._generate_file({"path": "f.py", "purpose": "do f"}, "spec", "conv", [])
    assert out == draft.strip()                        # guard kept the good draft


def test_generate_file_review_off(tmp_path, monkeypatch):
    import devmode.build as B
    from core.config import get_config
    get_config().raw()["devmode"]["build_review"] = False

    calls = []
    monkeypatch.setattr(B, "_stream",
                        lambda msgs, precise=False: (calls.append(1), "x = 1\n")[1])
    b = B.Builder(tmp_path)
    out = b._generate_file({"path": "f.py", "purpose": "do f"}, "spec", "conv", [])
    assert out == "x = 1" and len(calls) == 1          # no second review call


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
    get_config().raw()["devmode"]["best_of"] = False   # isolate the reflection path
    calls = []
    monkeypatch.setattr(S, "_stream",
                        lambda msgs, precise=False: (calls.append(1),
                                                     "DRAFT" if len(calls) == 1 else "IMPROVED")[1])
    ds = DevSession(tmp_path, "x")
    out = ds._summarize(PHASES_BY_ID["requirements"], [])
    get_config().raw()["devmode"]["best_of"] = True
    assert len(calls) == 2 and out == "IMPROVED"      # draft then refine


def test_best_of_generates_candidates_and_judges(tmp_path, monkeypatch):
    from core.config import get_config
    get_config().raw()["devmode"]["best_of"] = True
    get_config().raw()["devmode"]["reflect"] = False   # one call per candidate
    # security has best_of=3 → 3 candidate calls, then 1 judge call picking #2
    seq = iter(["CANDIDATE-1", "CANDIDATE-2", "CANDIDATE-3", "the best is 2"])
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: next(seq))
    ds = DevSession(tmp_path, "x")
    out = ds._summarize(PHASES_BY_ID["security"], [])
    get_config().raw()["devmode"]["reflect"] = True
    assert out == "CANDIDATE-2"                         # judge chose candidate 2


def test_best_of_disabled_is_single_pass(tmp_path, monkeypatch):
    from core.config import get_config
    get_config().raw()["devmode"]["best_of"] = False
    get_config().raw()["devmode"]["reflect"] = False
    calls = []
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: (calls.append(1), "ONE")[1])
    ds = DevSession(tmp_path, "x")
    out = ds._summarize(PHASES_BY_ID["security"], [])
    get_config().raw()["devmode"]["reflect"] = True
    assert out == "ONE" and len(calls) == 1            # disabled → no candidates, no judge


def test_judge_best_defaults_to_first_on_unparsable(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: "I cannot decide")
    ds = DevSession(tmp_path, "x")
    out = ds._judge_best(PHASES_BY_ID["security"], ["A", "B", "C"])
    assert out == "A"                                  # no number → safe fallback to first


def test_phases_have_best_of_set():
    assert PHASES_BY_ID["security"].best_of == 3
    assert PHASES_BY_ID["requirements"].best_of == 3
    assert PHASES_BY_ID["app_flow"].best_of == 1       # non-critical → single pass


def test_decomposed_summarize_designs_each_unit(tmp_path, monkeypatch):
    # data_model decomposes by "entity": list → overview → detail each
    seq = iter(['["User", "Message"]', 'OVERVIEW-TEXT', 'USER-DETAIL', 'MESSAGE-DETAIL'])
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: next(seq))
    ds = DevSession(tmp_path, "x")
    out = ds._summarize(PHASES_BY_ID["data_model"], [])
    assert "## Overview" in out and "OVERVIEW-TEXT" in out
    assert "## User" in out and "USER-DETAIL" in out
    assert "## Message" in out and "MESSAGE-DETAIL" in out


def test_decomposed_falls_back_when_no_items(tmp_path, monkeypatch):
    from core.config import get_config
    get_config().raw().setdefault("devmode", {})["reflect"] = False
    # list step yields no array → fall through to the normal single-pass summarize
    seq = iter(["no json here", "PLAIN-SUMMARY"])
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: next(seq))
    ds = DevSession(tmp_path, "x")
    out = ds._summarize(PHASES_BY_ID["data_model"], [])
    get_config().raw()["devmode"]["reflect"] = True
    assert out == "PLAIN-SUMMARY"


def test_research_is_multi_query(tmp_path, monkeypatch):
    import core.model as CM
    import rag.research as RR

    class _AI:
        content = '["websocket scaling 2026", "Signal protocol implementation"]'

    class _LLM:
        def invoke(self, msgs):
            return _AI()

    monkeypatch.setattr(CM, "get_chat_model", lambda precise=False: _LLM())
    monkeypatch.setattr(RR, "research_topic",
                        lambda q, project="", fetch_pages=2: {"text": f"FACTS::{q}", "count": 1, "sources": []})
    ds = DevSession(tmp_path, "a messaging app")
    out = ds._research(PHASES_BY_ID["architecture"])
    assert "websocket scaling 2026" in out and "FACTS::websocket scaling 2026" in out
    assert "Signal protocol implementation" in out      # both queries fetched


def test_research_queries_fallback(tmp_path, monkeypatch):
    import core.model as CM

    class _LLM:
        def invoke(self, msgs):
            raise RuntimeError("no model")

    monkeypatch.setattr(CM, "get_chat_model", lambda precise=False: _LLM())
    ds = DevSession(tmp_path, "x")
    qs = ds._research_queries(PHASES_BY_ID["architecture"])
    assert len(qs) == 1 and "Architecture" in qs[0]     # graceful single-query fallback


def test_consistency_check_flags_contradiction(tmp_path, monkeypatch):
    import devmode.session as S
    from core.config import get_config
    get_config().raw()["devmode"]["consistency_check"] = True

    ds = DevSession(tmp_path, "an E2E messaging app")
    # An earlier (security) decision exists on disk.
    sec = PHASES_BY_ID["security"]
    ds._write_artifact(sec, "Server must NEVER see plaintext or private keys.", "t")

    finding = "HIGH — Data Model stores private_key server-side, but Security says the server never holds keys: store keys only on the device."
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: finding)
    out = ds._report_consistency(PHASES_BY_ID["data_model"],
                                 "Devices table has a private_key column on the server.")
    assert out == finding
    note = (tmp_path / "docs" / "dev" / "consistency_notes.md").read_text()
    assert "private_key" in note and "Data Model" in note     # logged to the notes file


def test_consistency_check_clean_when_none(tmp_path, monkeypatch):
    import devmode.session as S
    from core.config import get_config
    get_config().raw()["devmode"]["consistency_check"] = True

    ds = DevSession(tmp_path, "x")
    ds._write_artifact(PHASES_BY_ID["security"], "Use JWT auth.", "t")
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: "NONE")
    out = ds._report_consistency(PHASES_BY_ID["data_model"], "Users table with id, name.")
    assert out == ""
    assert not (tmp_path / "docs" / "dev" / "consistency_notes.md").exists()  # nothing logged


def test_consistency_check_excludes_self_and_first_phase(tmp_path, monkeypatch):
    import devmode.session as S
    from core.config import get_config
    get_config().raw()["devmode"]["consistency_check"] = True

    ds = DevSession(tmp_path, "x")
    # A non-NONE digest, so a contradiction WOULD surface if any comparison ran.
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: "- stores a private_key column")
    notes = tmp_path / "docs" / "dev" / "consistency_notes.md"

    # No prior artifacts at all → nothing to compare against → no findings.
    out = ds._report_consistency(PHASES_BY_ID["requirements"], "Some requirements.")
    assert out == "" and not notes.exists()
    assert ds.state["digests"]["requirements"]              # but its digest was cached for later

    # Fresh session where ONLY the current phase has been decided → self is
    # excluded → still nothing to compare against (no false self-conflict).
    ds2 = DevSession(tmp_path / "ws2", "x")
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: "- stores a private_key column")
    ds2._write_artifact(PHASES_BY_ID["data_model"], "old schema", "t")
    out = ds2._report_consistency(PHASES_BY_ID["data_model"], "new schema")
    assert out == "" and not (tmp_path / "ws2" / "docs" / "dev" / "consistency_notes.md").exists()


def test_consistency_check_off(tmp_path, monkeypatch):
    import devmode.session as S
    from core.config import get_config
    get_config().raw()["devmode"]["consistency_check"] = False

    ds = DevSession(tmp_path, "x")
    ds._write_artifact(PHASES_BY_ID["security"], "no plaintext", "t")
    called = []
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: (called.append(1), "X")[1])
    out = ds._report_consistency(PHASES_BY_ID["data_model"], "anything")
    assert out == "" and called == []                         # disabled → no model call


def test_review_findings_structured_parses_and_sorts(tmp_path, monkeypatch):
    import devmode.session as S
    ds = DevSession(tmp_path, "an E2E messaging app")
    # two decided phases (need >= 2 for a cross-phase review)
    ds._write_artifact(PHASES_BY_ID["security"], "Server never sees plaintext.", "t")
    ds._write_artifact(PHASES_BY_ID["data_model"], "Devices table stores private_key.", "t")
    ds.state["digests"] = {"security": "no plaintext on server",
                           "data_model": "stores private_key server-side"}

    payload = (
        '[{"severity":"LOW","target":"data_model","issue":"minor","fix":"x"},'
        ' {"severity":"HIGH","target":"data_model","issue":"private key on server",'
        '"fix":"keep keys client-side"},'
        ' {"severity":"MEDIUM","target":"bogus_phase","issue":"y","fix":"z"}]'  # invalid target dropped
    )
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: payload)
    out = ds._review_findings_structured()
    assert [f["severity"] for f in out] == ["HIGH", "LOW"]      # sorted; bogus target dropped
    assert out[0]["target"] == "data_model"


def test_apply_fix_rewrites_artifact_and_resyncs(tmp_path, monkeypatch):
    import devmode.session as S
    from core.config import get_config
    get_config().raw()["files"]["confirmation"] = "never"

    ds = DevSession(tmp_path, "x")
    ds._write_artifact(PHASES_BY_ID["data_model"],
                       "Devices table stores private_key server-side.", "orig transcript")
    ds.state["digests"] = {"data_model": "old digest"}

    monkeypatch.setattr(S.Confirm, "ask", lambda *a, **k: True)
    monkeypatch.setattr(S, "_stream",
                        lambda msgs, precise=False: "Devices table stores ONLY public keys; private keys stay on-device.")
    monkeypatch.setattr(ds, "_build_exists", lambda: True)
    resynced = {}
    import devmode.resync as R
    monkeypatch.setattr(R, "resync",
                        lambda ws, title, old, new: resynced.update(title=title, new=new))

    ok = ds._apply_fix({"severity": "HIGH", "target": "data_model",
                        "issue": "private key on server", "fix": "keep keys client-side"})
    assert ok is True
    art = (tmp_path / "docs" / "dev" / "06_data_model.md").read_text()
    assert "stay on-device" in art                              # artifact rewritten
    assert "orig transcript" in art                             # original transcript preserved
    assert resynced["title"] == "Data Model & DB Schema"        # code resync triggered
    assert "data_model" not in ds.state.get("digests", {})      # stale digest invalidated


def test_apply_fix_rejects_truncating_rewrite(tmp_path, monkeypatch):
    import devmode.session as S
    ds = DevSession(tmp_path, "x")
    big = "## Schema\n" + ("- a detailed field line that carries real content\n" * 60)
    ds._write_artifact(PHASES_BY_ID["data_model"], big, "t")
    monkeypatch.setattr(S.Confirm, "ask", lambda *a, **k: True)
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: "Tiny rewrite.")  # truncation
    monkeypatch.setattr(ds, "_build_exists", lambda: True)
    import devmode.resync as R
    hit = {}
    monkeypatch.setattr(R, "resync", lambda *a, **k: hit.setdefault("x", 1))
    ok = ds._apply_fix({"severity": "HIGH", "target": "data_model", "issue": "i", "fix": "f"})
    assert ok is False and "x" not in hit                      # guard blocked the shrink + resync
    assert "detailed field line" in (tmp_path / "docs" / "dev" / "06_data_model.md").read_text()


def test_apply_fix_no_resync_without_build(tmp_path, monkeypatch):
    import devmode.session as S
    from core.config import get_config
    get_config().raw()["files"]["confirmation"] = "never"

    ds = DevSession(tmp_path, "x")
    ds._write_artifact(PHASES_BY_ID["security"], "Use sessions.", "t")
    monkeypatch.setattr(S.Confirm, "ask", lambda *a, **k: True)
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: "Use JWT tokens instead.")
    monkeypatch.setattr(ds, "_build_exists", lambda: False)
    called = {}
    import devmode.resync as R
    monkeypatch.setattr(R, "resync", lambda *a, **k: called.setdefault("hit", True))

    ok = ds._apply_fix({"severity": "HIGH", "target": "security",
                        "issue": "auth", "fix": "use jwt"})
    assert ok is True and "hit" not in called                  # no build → no resync


def test_phases_have_decompose_set():
    assert PHASES_BY_ID["data_model"].decompose == "entity"
    assert PHASES_BY_ID["api"].decompose == "resource"
    assert PHASES_BY_ID["architecture"].decompose == "component"
    assert PHASES_BY_ID["requirements"].decompose == ""   # not decomposed


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
