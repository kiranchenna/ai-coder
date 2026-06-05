"""
evals/rubric.py — Judge-model scoring of a single phase decision
================================================================
Grades a phase's decision artifact for substance: how concretely and completely
it decides the hard parts of the product, scored against the phase's own
``must_cover`` checklist plus an overall 0–10.

The model call is injectable (``invoke``) so the parsing and prompt construction
are unit-testable without a live model.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from core.model import balanced_json_objects


def build_judge_messages(idea: str, spec, decision: str) -> list:
    """Construct the grading prompt for one phase decision."""
    must = "\n".join(f"- {m}" for m in spec.must_cover) or "- (no specific checklist for this phase)"
    system = (
        "You are a strict senior engineer grading ONE SDLC design decision. Grade only on "
        "substance: concreteness, technical correctness, completeness against the checklist, "
        "and whether it confronts the genuinely HARD parts of THIS specific product. Heavily "
        "penalize generic boilerplate, vague hand-waving, and silently dropped requirements. "
        "Be harsh — a 10 is reserved for a decision a senior engineer could build from "
        "directly with no gaps.\n"
        "Score each checklist item: 1 = fully and concretely covered, 0.5 = mentioned but "
        "vague/incomplete, 0 = missing or wrong.\n"
        'Output ONLY JSON, no prose:\n'
        '{"coverage": [{"item": "<checklist item>", "score": 0|0.5|1}], '
        '"overall": <number 0-10>, "rationale": "<one terse sentence>"}'
    )
    human = (
        f"# Product\n{idea}\n\n"
        f"# Phase being graded: {spec.title}\n"
        f"It must decide: {spec.focus}\n\n"
        f"# Required checklist\n{must}\n\n"
        f"# The decision to grade\n{decision[:12000]}\n\n"
        "Return the JSON grade now."
    )
    return [SystemMessage(content=system), HumanMessage(content=human)]


def parse_score(raw: str) -> dict | None:
    """Parse the judge's JSON grade tolerantly. Returns a normalized dict or None.

    Accepts the first balanced ``{...}`` span that has a numeric ``overall``;
    clamps it to [0, 10] so a stray scale (e.g. "85") can't skew an average.
    """
    for span in balanced_json_objects(raw or ""):
        try:
            obj = json.loads(span)
        except Exception:
            continue
        if not isinstance(obj, dict) or "overall" not in obj:
            continue
        try:
            overall = float(obj["overall"])
        except (TypeError, ValueError):
            continue
        cov = obj.get("coverage")
        coverage = [c for c in cov if isinstance(c, dict)] if isinstance(cov, list) else []
        return {
            "overall": max(0.0, min(10.0, overall)),
            "coverage": coverage,
            "rationale": str(obj.get("rationale", "")).strip(),
        }
    return None


def coverage_fraction(score: dict) -> float | None:
    """Mean of the per-item checklist scores (0–1), or None if no items were graded."""
    items = [c.get("score") for c in score.get("coverage", [])]
    nums = [float(s) for s in items if isinstance(s, (int, float))]
    return sum(nums) / len(nums) if nums else None


def score_decision(idea: str, spec, decision: str, judge_model: str = "", invoke=None) -> dict | None:
    """Grade ``decision`` for ``spec``. ``invoke(messages) -> str`` is injectable
    for tests; by default it calls the (optionally stronger) judge model."""
    if not decision.strip():
        return None
    messages = build_judge_messages(idea, spec, decision)
    if invoke is None:
        from core.model import get_chat_model

        llm = get_chat_model(precise=True, model=judge_model or None)

        def invoke(msgs):
            out = llm.invoke(msgs).content
            return out if isinstance(out, str) else str(out)

    raw = invoke(messages)
    return parse_score(raw if isinstance(raw, str) else str(raw))
