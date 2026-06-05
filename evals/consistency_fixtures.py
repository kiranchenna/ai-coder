"""
evals/consistency_fixtures.py — Labeled cross-phase contradiction cases
=======================================================================
The `consistency_check` lever's job is to catch a later phase's decision
contradicting an earlier one. A single-phase ablation can't measure that (there
are no priors to conflict with), so this fixture set drives the checker directly:
each case is a (prior-phase constraints, new-phase decision) pair labeled
``contradiction`` or ``clean``.

We feed these straight into ``DevSession._consistency_findings`` — the same code
path the live build uses — so only the model's judgment varies, not the inputs.
The hard cases (subtle E2E violations) are exactly where a small model is known
to struggle; they're included on purpose so the eval can show the ceiling.
"""

from __future__ import annotations

# Each case:
#   id        — short name
#   new_phase — the phase id whose decision is being checked (PHASES_BY_ID key)
#   prior     — constraint summary from EARLIER phases (what the checker compares against)
#   new       — the new phase's commitments/constraints (what's being judged)
#   label     — "contradiction" (should be flagged) | "clean" (should NOT be flagged)
#   difficulty— "blatant" | "subtle"  (for slicing the results)

CASES: list[dict] = [
    {
        "id": "e2e_plaintext_body",
        "new_phase": "data_model",
        "difficulty": "blatant",
        "label": "contradiction",
        "prior": (
            "## Security & Non-Functional\n"
            "- End-to-end encryption: the server MUST NEVER see plaintext message content.\n"
            "- Per-device keys; private keys never leave the client device.\n"
            "- Auth: short-lived JWT access tokens + refresh tokens."
        ),
        "new": (
            "- messages table: id (uuid PK), chat_id (FK), sender_id (FK), "
            "body TEXT NOT NULL  -- the message text in plaintext, indexed for full-text search\n"
            "- created_at timestamptz"
        ),
    },
    {
        "id": "e2e_ciphertext_ok",
        "new_phase": "data_model",
        "difficulty": "blatant",
        "label": "clean",
        "prior": (
            "## Security & Non-Functional\n"
            "- End-to-end encryption: the server MUST NEVER see plaintext message content.\n"
            "- Per-device keys; private keys never leave the client device."
        ),
        "new": (
            "- messages table: id (uuid PK), chat_id (FK), sender_id (FK), "
            "ciphertext BYTEA NOT NULL, nonce BYTEA, created_at timestamptz\n"
            "- device_keys table: device_id, user_id, public_key  -- only PUBLIC keys server-side\n"
            "- per-recipient message_envelopes(message_id, recipient_device_id, wrapped_key)"
        ),
    },
    {
        "id": "datastore_mismatch",
        "new_phase": "data_model",
        "difficulty": "blatant",
        "label": "contradiction",
        "prior": (
            "## Architecture & Tech Stack\n"
            "- Primary datastore: PostgreSQL 16 for all relational data (users, chats, messages).\n"
            "- Redis for presence/ephemeral state only."
        ),
        "new": (
            "- Messages are stored in MongoDB collections (one collection per chat) for flexibility.\n"
            "- Users and chats live in PostgreSQL."
        ),
    },
    {
        "id": "auth_mismatch",
        "new_phase": "api",
        "difficulty": "blatant",
        "label": "contradiction",
        "prior": (
            "## Security & Non-Functional\n"
            "- Authentication: stateless JWT bearer tokens in the Authorization header.\n"
            "- No server-side session store."
        ),
        "new": (
            "- All endpoints authenticate via a server-side session cookie (connect.sid), "
            "backed by a sessions table in Postgres.\n"
            "- Login sets the cookie; logout deletes the session row."
        ),
    },
    {
        "id": "auth_consistent",
        "new_phase": "api",
        "difficulty": "blatant",
        "label": "clean",
        "prior": (
            "## Security & Non-Functional\n"
            "- Authentication: stateless JWT bearer tokens in the Authorization header."
        ),
        "new": (
            "- Every endpoint expects `Authorization: Bearer <jwt>`; the gateway validates the "
            "signature and expiry.\n"
            "- 401 on missing/invalid token, using the standard error envelope."
        ),
    },
    {
        "id": "dropped_group_feature",
        "new_phase": "data_model",
        "difficulty": "subtle",
        "label": "contradiction",
        "prior": (
            "## Requirements\n"
            "- Group chats with membership management (add/remove members, admins).\n"
            "- Read receipts per recipient.\n"
            "## Security & Non-Functional\n"
            "- E2E encryption with per-device keys."
        ),
        "new": (
            "- users(id, handle), chats(id, type), messages(id, chat_id, sender_id, ciphertext).\n"
            "- A chat has a sender and a ciphertext; that's the whole model."
            "  (no membership table, no per-recipient receipt/delivery state)"
        ),
    },
    {
        "id": "single_key_vs_per_device",
        "new_phase": "data_model",
        "difficulty": "subtle",
        "label": "contradiction",
        "prior": (
            "## Security & Non-Functional\n"
            "- End-to-end encryption with PER-DEVICE keys (a user may have several devices, "
            "each with its own keypair). Messages are encrypted to every recipient device."
        ),
        "new": (
            "- users(id, handle, public_key)  -- exactly one public key per user account.\n"
            "- messages(id, chat_id, sender_id, ciphertext)  -- encrypted once per message."
        ),
    },
    {
        "id": "scale_consistent",
        "new_phase": "deployment",
        "difficulty": "subtle",
        "label": "clean",
        "prior": (
            "## Architecture & Tech Stack\n"
            "- Stateless API services behind a load balancer; horizontal scaling.\n"
            "- PostgreSQL primary with read replicas; Redis for presence.\n"
            "## Security & Non-Functional\n"
            "- Target 50k concurrent websocket connections."
        ),
        "new": (
            "- Kubernetes: API deployment with HPA on CPU, 3+ replicas; a dedicated websocket "
            "gateway deployment scaled to handle 50k connections.\n"
            "- Managed Postgres with read replicas; Redis cluster for presence.\n"
            "- Per-environment secrets via the cluster secret store."
        ),
    },
]
