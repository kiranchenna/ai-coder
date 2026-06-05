"""Tests for the Developer Mode eval harness (pure logic — no live model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from devmode.phases import PHASES_BY_ID
from evals.rubric import (
    build_judge_messages,
    coverage_fraction,
    parse_score,
    score_decision,
)
from evals.run_eval import _ALL_OFF, LEVER_CONFIGS, resolve_levers
from evals.fixtures import EVAL_IDEAS, get_idea

_SPEC = PHASES_BY_ID["security"]


# ── parse_score ────────────────────────────────────────────────────────────────

def test_parse_score_extracts_overall_and_coverage():
    raw = ('Here is the grade: {"coverage": [{"item": "auth", "score": 1}, '
           '{"item": "e2e", "score": 0.5}], "overall": 7.5, "rationale": "solid"}')
    s = parse_score(raw)
    assert s["overall"] == 7.5
    assert len(s["coverage"]) == 2
    assert s["rationale"] == "solid"


def test_parse_score_none_without_overall():
    assert parse_score('{"coverage": []}') is None
    assert parse_score("no json here") is None
    assert parse_score("") is None


def test_parse_score_clamps_out_of_range_scale():
    # A model that grades on 0-100 shouldn't be able to skew the average.
    assert parse_score('{"overall": 85}')["overall"] == 10.0
    assert parse_score('{"overall": -3}')["overall"] == 0.0


def test_parse_score_ignores_non_numeric_overall():
    assert parse_score('{"overall": "great"}') is None


# ── coverage_fraction ───────────────────────────────────────────────────────────

def test_coverage_fraction_averages_item_scores():
    score = {"coverage": [{"score": 1}, {"score": 0.5}, {"score": 0}]}
    assert coverage_fraction(score) == 0.5


def test_coverage_fraction_none_when_empty():
    assert coverage_fraction({"coverage": []}) is None


# ── build_judge_messages ────────────────────────────────────────────────────────

def test_judge_prompt_includes_must_cover_and_decision():
    msgs = build_judge_messages("a secure chat app", _SPEC, "we use AES and JWT")
    human = msgs[1].content
    assert "we use AES and JWT" in human
    # every checklist item for the phase is surfaced to the grader
    for item in _SPEC.must_cover:
        assert item in human


# ── score_decision (injected invoke) ────────────────────────────────────────────

def test_score_decision_uses_injected_invoke():
    captured = {}

    def fake_invoke(messages):
        captured["called"] = True
        return '{"overall": 6, "coverage": [{"item": "x", "score": 1}], "rationale": "ok"}'

    s = score_decision("idea", _SPEC, "a real decision", invoke=fake_invoke)
    assert captured["called"] is True
    assert s["overall"] == 6.0


def test_score_decision_empty_decision_returns_none():
    # An empty artifact must never reach the judge (and never score).
    called = []
    score_decision("idea", _SPEC, "   ", invoke=lambda m: called.append(1) or "{}")
    assert not called


# ── resolve_levers ──────────────────────────────────────────────────────────────

def test_resolve_levers_baseline_all_off():
    assert resolve_levers("baseline") == _ALL_OFF
    assert all(v is False for v in resolve_levers("baseline").values())


def test_resolve_levers_full_turns_on_the_stack():
    levers = resolve_levers("full")
    assert levers["reflect"] and levers["best_of"] and levers["consistency_check"]


def test_resolve_levers_single_isolates_one_lever():
    levers = resolve_levers("best_of")
    assert levers["best_of"] is True
    assert levers["reflect"] is False  # isolated: only best_of differs from baseline


def test_resolve_levers_unknown_raises():
    import pytest

    with pytest.raises(SystemExit):
        resolve_levers("nope")


def test_every_named_config_only_uses_known_levers():
    # Guards against a typo'd lever key that would silently do nothing.
    for name, overrides in LEVER_CONFIGS.items():
        assert set(overrides) <= set(_ALL_OFF), name


# ── fixtures ─────────────────────────────────────────────────────────────────────

def test_get_idea_prefers_explicit_over_fixture():
    assert get_idea("whatsapp", "my own idea") == "my own idea"


def test_get_idea_resolves_named_fixture():
    assert get_idea("whatsapp", None) == EVAL_IDEAS["whatsapp"]


def test_get_idea_unknown_fixture_raises():
    import pytest

    with pytest.raises(SystemExit):
        get_idea("does-not-exist", None)


# ── consistency-check detection eval ─────────────────────────────────────────────

from evals.consistency_fixtures import CASES
from evals.run_consistency_eval import compute_metrics, detect_case, evaluate


def test_consistency_fixtures_are_well_formed():
    # Every case must name a real phase, a valid label, and have balanced classes.
    labels = []
    for c in CASES:
        assert c["new_phase"] in PHASES_BY_ID, c["id"]
        assert c["label"] in ("contradiction", "clean"), c["id"]
        assert c["difficulty"] in ("blatant", "subtle"), c["id"]
        assert c["prior"].strip() and c["new"].strip(), c["id"]
        labels.append(c["label"])
    # Need both positives and negatives or precision/recall are meaningless.
    assert "contradiction" in labels and "clean" in labels


def test_detect_case_uses_injected_detector():
    case = CASES[0]
    # A detector returning non-empty text means "flagged".
    assert detect_case(case, detector=lambda spec, p, n: "HIGH — conflict") is True
    assert detect_case(case, detector=lambda spec, p, n: "NONE handled -> ''") is True  # non-empty
    assert detect_case(case, detector=lambda spec, p, n: "") is False
    assert detect_case(case, detector=lambda spec, p, n: "   ") is False


def test_compute_metrics_confusion_matrix():
    results = [
        {"label": "contradiction", "flagged": True},    # TP
        {"label": "contradiction", "flagged": False},   # FN
        {"label": "clean", "flagged": True},            # FP
        {"label": "clean", "flagged": False},           # TN
    ]
    m = compute_metrics(results)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5  # 1 / (1+1)
    assert m["recall"] == 0.5     # 1 / (1+1)
    assert m["accuracy"] == 0.5


def test_compute_metrics_perfect_detector():
    results = [
        {"label": "contradiction", "flagged": True},
        {"label": "clean", "flagged": False},
    ]
    m = compute_metrics(results)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["accuracy"] == 1.0
    assert m["fp"] == 0 and m["fn"] == 0


def test_evaluate_majority_vote_and_correctness():
    # A flaky detector: flags on odd-indexed runs. With repeat=3 → flags 2/3 → flagged.
    state = {"n": 0}

    def flaky(spec, prior, new):
        state["n"] += 1
        return "conflict" if state["n"] % 2 == 1 else ""

    one_case = [c for c in CASES if c["label"] == "contradiction"][:1]
    results = evaluate(one_case, judge_model="", repeat=3, detector=flaky)
    assert results[0]["flag_count"] == 2          # runs 1 and 3 flagged
    assert results[0]["flagged"] is True          # majority of 3
    assert results[0]["correct"] is True          # label is contradiction
