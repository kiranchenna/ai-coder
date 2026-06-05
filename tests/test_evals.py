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
