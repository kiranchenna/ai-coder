"""
evals/build_review_fixtures.py — Drafts with planted issues for the build_review eval
=====================================================================================
`build_review` is a build-time, per-file lever: it takes a generated draft and
runs a self-review→fix pass to remove placeholders/TODOs/stubs and obvious bugs
before the file is written. To measure it controllably we hand the real review
pass drafts with a KNOWN planted issue and check whether the issue is gone after
review — plus clean controls to confirm review doesn't mangle good code.

Each case feeds ``Builder._review_file`` (the live build's code path).
"""

from __future__ import annotations

# A minimal stand-in for the conventions/spec the real build passes as `system`.
_CONVENTIONS = (
    "Project conventions: Python 3.11, full type hints, snake_case. Write COMPLETE, "
    "production-ready code — NO placeholders, TODOs, stubs, or NotImplementedError. "
    "Every function must be fully implemented."
)


def _entry(path: str, purpose: str) -> dict:
    return {"path": path, "purpose": purpose, "implements": []}


def _prompt(path: str, purpose: str) -> str:
    return f"Generate {path}.\nPurpose: {purpose}\nOutput ONLY the raw file content."


# Each case:
#   id        — short name
#   entry     — the file entry ({path, purpose, ...})
#   system    — the system message (conventions) the review sees
#   prompt    — the original generation prompt the review sees
#   draft     — the draft handed to the review pass (contains the planted issue)
#   marker    — substring whose ABSENCE after review means the issue was fixed
#               (for "preserved" cases it's a symbol that must REMAIN)
#   expect    — "removed" (placeholder should be gone) | "preserved" (clean code kept)

CASES: list[dict] = [
    {
        "id": "todo_email_validation",
        "entry": _entry("validators.py", "email validation helper"),
        "marker": "TODO",
        "expect": "removed",
        "draft": (
            "import re\n\n\n"
            "def validate_email(email: str) -> bool:\n"
            "    # TODO: implement proper email validation\n"
            "    return True\n"
        ),
    },
    {
        "id": "notimplemented_hash",
        "entry": _entry("security.py", "password hashing and verification"),
        "marker": "NotImplementedError",
        "expect": "removed",
        "draft": (
            "import hashlib\n\n\n"
            "def hash_password(password: str) -> str:\n"
            "    raise NotImplementedError  # fill in later\n\n\n"
            "def verify_password(password: str, hashed: str) -> bool:\n"
            "    raise NotImplementedError\n"
        ),
    },
    {
        "id": "stub_pagination",
        "entry": _entry("pagination.py", "offset/limit pagination of a list"),
        "marker": "pass",
        "expect": "removed",
        "draft": (
            "def paginate(items: list, page: int, per_page: int) -> list:\n"
            "    pass  # stub — not implemented yet\n"
        ),
    },
    {
        "id": "clean_adder_preserved",
        "entry": _entry("math_utils.py", "small arithmetic helpers"),
        "marker": "def add",
        "expect": "preserved",
        "draft": (
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n\n\n"
            "def multiply(a: int, b: int) -> int:\n"
            "    return a * b\n"
        ),
    },
]

# Attach the system/prompt every case shares (built from its entry).
for _c in CASES:
    _c["system"] = _CONVENTIONS
    _c["prompt"] = _prompt(_c["entry"]["path"], _c["entry"]["purpose"])
