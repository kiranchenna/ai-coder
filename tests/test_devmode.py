"""Tests for Developer Mode (devmode) — engine logic, no model calls."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.markdown import Markdown

import devmode.session as S
from devmode.phases import PHASES, PHASES_BY_ID
from devmode.session import DevSession


# ── _ask / _confirm — TUI-aware, no popups for /develop's discuss loop ──────────
# Live-reported bug: every question and yes/no confirmation in /develop's
# flow opened as a popup showing nothing but a bare phase id — genuinely
# confusing. In the TUI these now route through the main chat input instead
# (agent/tui.py's ask_inline/ask_inline_confirm); in the plain REPL (or if
# textual isn't installed) they still fall back to a normal Prompt.ask/
# Confirm.ask, unchanged from before.

def test_ask_routes_to_tui_inline_when_tui_active(monkeypatch):
    import agent.tui as tui_mod

    monkeypatch.setattr(tui_mod, "is_tui_active", lambda: True)
    calls = []
    monkeypatch.setattr(tui_mod, "ask_inline",
                        lambda q, d="": calls.append((q, d)) or "typed reply")

    assert S._ask("Your reply?", default="fallback") == "typed reply"
    assert calls == [("Your reply?", "fallback")]


def test_ask_falls_back_to_prompt_ask_when_no_tui(monkeypatch):
    import agent.tui as tui_mod

    monkeypatch.setattr(tui_mod, "is_tui_active", lambda: False)
    monkeypatch.setattr(S.Prompt, "ask", lambda *a, **k: "plain repl reply")

    assert S._ask("Your reply?") == "plain repl reply"


def test_confirm_routes_to_tui_inline_when_tui_active(monkeypatch):
    import agent.tui as tui_mod

    monkeypatch.setattr(tui_mod, "is_tui_active", lambda: True)
    calls = []
    monkeypatch.setattr(tui_mod, "ask_inline_confirm",
                        lambda q, d=True: calls.append((q, d)) or False)

    assert S._confirm("Apply fix?", default=True) is False
    assert calls == [("Apply fix?", True)]


def test_confirm_falls_back_to_confirm_ask_when_no_tui(monkeypatch):
    import agent.tui as tui_mod

    monkeypatch.setattr(tui_mod, "is_tui_active", lambda: False)
    monkeypatch.setattr(S.Confirm, "ask", lambda *a, **k: True)

    assert S._confirm("Apply fix?") is True


# ── _stream — TUI-aware output routing ───────────────────────────────────────
# Live-reproduced bug: _stream() wrote every model token straight to
# sys.stdout for a live "typing" effect — correct in the plain REPL (stdout
# genuinely is the terminal there), but under the TUI stdout isn't connected
# to the RichLog at all, so the entire discuss-loop proposal/critique text
# silently never appeared anywhere a user could see it. Confirmed live: a
# full /develop phase ran to completion with the AI's actual response never
# once landing in the chat log.

class _FakeChunk:
    def __init__(self, text):
        self.content = text


class _FakeStreamingLLM:
    def __init__(self, pieces):
        self._pieces = pieces

    def stream(self, messages):
        return iter(_FakeChunk(p) for p in self._pieces)


def test_stream_writes_to_stdout_when_no_tui(monkeypatch, capsys):
    import agent.tui as tui_mod
    import core.model as CM

    monkeypatch.setattr(tui_mod, "is_tui_active", lambda: False)
    monkeypatch.setattr(CM, "get_chat_model", lambda precise=False, model=None: _FakeStreamingLLM(["Hello", " world."]))

    printed = []
    monkeypatch.setattr(S, "console", type("C", (), {"print": lambda self, *a, **k: printed.append(a)})())

    result = S._stream([])
    assert result == "Hello world."
    assert "Hello world." in capsys.readouterr().out
    # No Markdown re-print in the plain REPL — the live-typed text is already
    # fully visible on screen; a second print would duplicate it.
    assert not any(isinstance(arg, Markdown) for call in printed for arg in call)


def test_stream_routes_through_console_when_tui_active(monkeypatch, capsys):
    import agent.tui as tui_mod
    import core.model as CM

    monkeypatch.setattr(tui_mod, "is_tui_active", lambda: True)
    monkeypatch.setattr(CM, "get_chat_model", lambda precise=False, model=None: _FakeStreamingLLM(["Hello", " world."]))

    printed = []
    monkeypatch.setattr(S, "console", type("C", (), {"print": lambda self, *a, **k: printed.append(a)})())

    result = S._stream([])
    assert result == "Hello world."
    # Nothing raw went to stdout — under the TUI that's invisible to the user.
    assert capsys.readouterr().out == ""
    rendered = [arg for call in printed for arg in call if isinstance(arg, Markdown)]
    assert len(rendered) == 1
    assert rendered[0].markup == "Hello world."


def test_init_all_phases_pending(tmp_path):
    ds = DevSession(tmp_path, "build a todo app")
    assert ds.state["idea"] == "build a todo app"
    assert all(ds.state["phases"][p.id]["status"] == "pending" for p in PHASES)


def test_show_status_includes_active_profile(tmp_path, capsys):
    # 'dev status' must surface which profile/levers are actually active, so a
    # user editing config.yaml can confirm it without guessing.
    from core.config import get_config

    dm = get_config().raw()["devmode"]
    saved = dm.get("profile")
    dm["profile"] = "thorough"
    try:
        DevSession(tmp_path, "x").show_status()
        assert "profile: thorough" in capsys.readouterr().out
    finally:
        if saved is None:
            dm.pop("profile", None)
        else:
            dm["profile"] = saved


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


def test_prior_grounding_prefers_cached_digest(tmp_path):
    # Grounding for the next phase uses the compact digest, not the full artifact,
    # so chaining many phases stays within the context budget.
    ds = DevSession(tmp_path, "x")
    ds._write_artifact(PHASES_BY_ID["requirements"], "FULL-DECISION-BODY", "t")
    ds.state.setdefault("digests", {})["requirements"] = "- a key constraint"
    grounding = ds._prior_grounding()
    assert "a key constraint" in grounding
    assert "FULL-DECISION-BODY" not in grounding   # digest replaces the raw body


def test_prior_grounding_is_bounded(tmp_path):
    # Even with several large artifacts, the grounding stays within budget so the
    # earliest phases are never silently truncated by an overflowing context.
    ds = DevSession(tmp_path, "x")
    digests = ds.state.setdefault("digests", {})
    for pid in ("requirements", "architecture", "data_model"):
        ds._write_artifact(PHASES_BY_ID[pid], "BODY", "t")
        digests[pid] = "X" * 20000
    assert len(ds._prior_grounding(budget=6000)) <= 7000


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


def test_run_phase_end_to_end_writes_artifact(tmp_path, monkeypatch):
    # Exercise the REAL phase glue: _discuss → _summarize → _report_consistency
    # → _write_artifact, with only the interactive prompt and the model stubbed.
    from core.config import get_config
    dm = get_config().raw()["devmode"]
    saved = {k: dm.get(k) for k in ("best_of", "reflect", "consistency_check")}
    dm.update(best_of=False, reflect=False, consistency_check=False)
    try:
        monkeypatch.setattr(S.Prompt, "ask", lambda *a, **k: "done")   # end discussion at once
        monkeypatch.setattr(S.Confirm, "ask", lambda *a, **k: True)
        monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: "The captured DECISION.")
        ds = DevSession(tmp_path, "a tiny CLI app")
        result = ds._run_phase(PHASES_BY_ID["requirements"])
        assert result == "done"
        art = tmp_path / "docs" / "dev" / "03_requirements.md"
        assert art.exists()
        text = art.read_text()
        assert "## Decision" in text and "The captured DECISION." in text  # summarized + written
    finally:
        dm.update(saved)


def test_fast_mode_skips_discussion(tmp_path, monkeypatch):
    # In auto mode, _discuss must NOT call Prompt.ask — it captures the proposal.
    def _boom(*a, **k):
        raise AssertionError("Prompt.ask should not be called in fast mode")
    monkeypatch.setattr(S.Prompt, "ask", _boom)
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: "A complete proposal.")
    ds = DevSession(tmp_path, "x", auto=True)
    result, messages = ds._discuss(PHASES_BY_ID["requirements"])
    assert result == "done" and any("proposal" in str(m.content) for m in messages)


def test_critic_stream_uses_judge_model(monkeypatch):
    import devmode.session as S2
    from core.config import get_config
    get_config().raw()["devmode"]["judge_model"] = "big-model:99b"
    seen = {}
    monkeypatch.setattr(S2, "_stream",
                        lambda msgs, precise=True, model=None: (seen.update(model=model), "ok")[1])
    out = S2._critic_stream([])
    get_config().raw()["devmode"]["judge_model"] = ""
    assert out == "ok" and seen["model"] == "big-model:99b"   # routed to the judge model


def test_critic_stream_falls_back_to_main(monkeypatch):
    import devmode.session as S2
    from core.config import get_config
    get_config().raw()["devmode"]["judge_model"] = ""        # not configured
    seen = {"model": "SENTINEL"}
    monkeypatch.setattr(S2, "_stream",
                        lambda msgs, precise=True: (seen.update(model=None), "ok")[1])  # no model kwarg
    S2._critic_stream([])
    assert seen["model"] is None                             # main model (no override)


def test_run_phase_review_routes_to_review(tmp_path, monkeypatch):
    # The review-kind phase must take the _run_review path, not _discuss.
    ds = DevSession(tmp_path, "x")
    ds._write_artifact(PHASES_BY_ID["requirements"], "some decision", "t")
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: "HIGH — looks fine")
    result = ds._run_phase(PHASES_BY_ID["review"])
    assert result == "done"
    assert (tmp_path / "docs" / "dev" / "design_review.md").exists()


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
    monkeypatch.setattr(b, "_verify_and_fix", lambda spec: None)
    monkeypatch.setattr(B, "_confirm", lambda *a, **k: True)
    b.build()

    assert (tmp_path / "app.py").read_text().startswith("# app.py")
    assert gen == ["app.py"]                        # the done file was skipped
    assert all(f["status"] == "done" for f in b._load_plan()["files"])
    assert (tmp_path / "docs" / "dev" / "build_manifest.json").exists()  # provenance written


def test_build_finds_nested_project_dir(tmp_path):
    import devmode.build as B
    (tmp_path / "docs" / "dev").mkdir(parents=True)
    b = B.Builder(tmp_path)
    # nothing at root → workspace
    assert b._project_dir() == tmp_path
    # a nested project with a test marker → that subdir
    nested = tmp_path / "wordcount-cli"
    (nested / "tests").mkdir(parents=True)
    (nested / "pyproject.toml").write_text("[project]\nname='x'\n")
    assert b._project_dir() == nested


def test_compile_problems_catches_syntax_error(tmp_path):
    import devmode.build as B
    (tmp_path / "docs" / "dev").mkdir(parents=True)
    b = B.Builder(tmp_path)
    (tmp_path / "good.py").write_text("x = 1\n")
    assert b._compile_problems(tmp_path) == ""
    (tmp_path / "bad.py").write_text("def oops(:\n    pass\n")   # syntax error
    problem = b._compile_problems(tmp_path)
    assert "syntax error" in problem.lower() and "bad.py" in problem


def test_plan_keeps_only_valid_implements(tmp_path, monkeypatch):
    import devmode.build as B
    from core.model import get_chat_model  # noqa: F401 — patched below
    (tmp_path / "docs" / "dev").mkdir(parents=True)
    (tmp_path / "docs" / "dev" / "03_requirements.md").write_text("# Requirements\nBuild X.")
    b = B.Builder(tmp_path)

    class _AI:
        content = '[{"path":"m.py","purpose":"model","implements":["data_model","bogus"]}]'
    monkeypatch.setattr(B, "get_chat_model", lambda precise=False: type("M", (), {"invoke": lambda self, msgs: _AI()})())
    plan = b.generate_plan()
    assert plan["files"][0]["implements"] == ["data_model"]     # 'bogus' dropped


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
    get_config().raw()["devmode"]["judge_model"] = "judge:big"  # best_of is gated on a judge
    # security has best_of=3 → 3 candidate calls, then 1 judge call picking #2
    seq = iter(["CANDIDATE-1", "CANDIDATE-2", "CANDIDATE-3", "the best is 2"])
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False, model=None: next(seq))
    ds = DevSession(tmp_path, "x")
    out = ds._summarize(PHASES_BY_ID["security"], [])
    get_config().raw()["devmode"]["reflect"] = True
    get_config().raw()["devmode"]["judge_model"] = ""
    assert out == "CANDIDATE-2"                         # judge chose candidate 2


def test_best_of_gated_off_without_judge_model(tmp_path, monkeypatch):
    # best_of requested but no judge_model → suppressed; a single reflected pass
    # runs instead (the eval showed a self-judge added latency without quality).
    from core.config import get_config
    get_config().raw()["devmode"]["best_of"] = True
    get_config().raw()["devmode"]["reflect"] = False
    get_config().raw()["devmode"]["judge_model"] = ""
    calls = []
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False, model=None: (calls.append(1), "ONE")[1])
    ds = DevSession(tmp_path, "x")
    out = ds._summarize(PHASES_BY_ID["security"], [])
    get_config().raw()["devmode"]["reflect"] = True
    assert out == "ONE" and len(calls) == 1            # gated → single pass, no judge call


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


def test_parse_choice_prefers_explicit_candidate():
    assert S.DevSession._parse_choice("I choose Candidate 2 because…", 3) == 1
    assert S.DevSession._parse_choice("#3 is best", 3) == 2
    assert S.DevSession._parse_choice("the answer is number 1", 3) == 0


def test_parse_choice_ignores_out_of_range_prose_digits():
    # "covers all 5 requirements" must not select a nonexistent candidate 5; the
    # real choice "candidate 2" wins. With only the stray 5, fall back to 0.
    assert S.DevSession._parse_choice("Candidate 2 covers all 5 requirements", 3) == 1
    assert S.DevSession._parse_choice("it covers 5 requirements nicely", 3) == 0


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


def test_decomposed_falls_back_when_units_empty(tmp_path, monkeypatch):
    # items parse fine, but every detail comes back empty → don't ship an
    # overview-only schema; fall back to single-pass.
    seq = iter(['["A", "B", "C"]', 'OVERVIEW', '', '', ''])
    monkeypatch.setattr(S, "_stream", lambda msgs, precise=False: next(seq))
    ds = DevSession(tmp_path, "x")
    out = ds._summarize_decomposed(PHASES_BY_ID["data_model"], [], "entity")
    assert out is None


def test_decision_section_ignores_conventions_timestamp():
    from devmode.session import _decision_section
    a = "# AICODER.md — conventions\n_Generated by Developer Mode: 2026-06-04 10:00_\n\nUse 4-space indent."
    b = "# AICODER.md — conventions\n_Generated by Developer Mode: 2026-06-04 22:00_\n\nUse 4-space indent."
    assert _decision_section(a) == _decision_section(b) == "Use 4-space indent."


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
