"""
evals/fixtures.py — Fixed project ideas for the ablation eval
=============================================================
Each fixture is a product idea with genuinely hard, defining parts — the kind a
small model tends to hand-wave. We grade how well a single design phase decides
those hard parts, with the quality levers on vs off.

Keep these stable: the whole point of an eval is a fixed yardstick you can
re-run as the prompts/levers change.
"""

from __future__ import annotations

EVAL_IDEAS: dict[str, str] = {
    "whatsapp": (
        "a privacy-first, end-to-end encrypted group messaging app for mobile and web — "
        "1:1 and group chats, media sharing, offline delivery, read receipts, and "
        "multi-device sync. Encryption must be real (the server never sees plaintext)."
    ),
    "invoicing": (
        "a multi-tenant invoicing SaaS: organizations manage customers, send invoices, "
        "accept online payments, handle taxes and multiple currencies, and reconcile "
        "payments — with per-tenant data isolation and an audit trail."
    ),
    "rideshare": (
        "a regional ride-hailing platform: riders request trips, drivers accept and "
        "navigate, real-time location tracking, surge pricing, in-app payments, and "
        "ratings — with live dispatch matching under load."
    ),
}

# The default phase to grade for each fixture. `security` is a strong default:
# it is a best-of-N phase, it has a demanding must-cover checklist, and it needs
# no web research (so the eval runs offline and deterministically).
DEFAULT_PHASE = "security"

DEFAULT_FIXTURE = "whatsapp"


def get_idea(fixture: str | None, idea: str | None) -> str:
    """Resolve the idea to grade: an explicit --idea wins, else a named fixture."""
    if idea:
        return idea
    key = (fixture or DEFAULT_FIXTURE).lower()
    if key not in EVAL_IDEAS:
        raise SystemExit(
            f"Unknown fixture '{key}'. Choose from: {', '.join(EVAL_IDEAS)} "
            "(or pass --idea '<your own>')."
        )
    return EVAL_IDEAS[key]
