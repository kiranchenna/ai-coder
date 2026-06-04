"""
devmode/phases.py — The SDLC phase specifications
==================================================
Data-driven: each phase is a role + a focus + an output artifact. The session
engine runs the same discussion loop for every phase.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseSpec:
    id: str
    title: str
    role: str
    goal: str
    focus: str           # the concrete things this phase must decide
    filename: str        # artifact filename (under docs/dev/, or AICODER.md)
    target: str = "doc"  # "doc" → docs/dev/<filename> | "conventions" → AICODER.md
    research: bool = False  # research current versions/best-practices first


PHASES: list[PhaseSpec] = [
    PhaseSpec(
        "requirements", "Requirements", "Requirements Analyst",
        "Agree the functional requirements and scope.",
        "the problem, primary users and their goals, in-scope vs out-of-scope, "
        "key user stories/features, and measurable success criteria",
        "01_requirements.md",
    ),
    PhaseSpec(
        "architecture", "Architecture & Tech Stack", "Software Architect",
        "Choose the architecture and technology stack.",
        "the overall architecture style and major components, the technology stack "
        "with current stable versions, key trade-offs, and third-party services",
        "02_architecture.md", research=True,
    ),
    PhaseSpec(
        "security", "Security & Non-Functional", "Security & Platform Engineer",
        "Decide security and non-functional requirements.",
        "authentication & authorization, secrets/data protection, scalability and "
        "performance targets, reliability/availability, and any compliance needs",
        "03_security_nfr.md",
    ),
    PhaseSpec(
        "data_model", "Data Model & DB Schema", "Database Architect",
        "Design the data model and database schema.",
        "the entities and relationships, each table/collection's fields and types, "
        "keys and indexes, and the migration approach",
        "04_data_model.md",
    ),
    PhaseSpec(
        "api", "API & Interface Contracts", "Backend Engineer",
        "Define the API / interface contracts.",
        "the endpoints (or module interfaces), request/response shapes, status codes "
        "and a consistent error format, auth per endpoint, and versioning",
        "05_api.md", research=True,
    ),
    PhaseSpec(
        "app_flow", "Application Flow & Business Logic", "Domain/Backend Engineer",
        "Map the application flows and core business logic.",
        "the core user/business flows step by step, business rules and validations, "
        "important edge cases, and key state transitions",
        "06_app_flow.md",
    ),
    PhaseSpec(
        "ui_ux", "UI/UX — Screens & Behaviour", "Frontend & UX Engineer",
        "Define the screens and user behaviour.",
        "the screen/page list, what each shows and does, navigation/routing, and key "
        "interactions and states (loading, empty, error)",
        "07_ui_ux.md", research=True,
    ),
    PhaseSpec(
        "testing", "Testing Strategy", "QA Engineer",
        "Decide the testing strategy.",
        "test levels (unit/integration/e2e), tools, what must be covered, test data, "
        "and CI gating",
        "08_testing.md",
    ),
    PhaseSpec(
        "deployment", "Deployment & Infrastructure", "DevOps Engineer",
        "Plan deployment and infrastructure.",
        "runtime/hosting, build and CI/CD, environments and config/secrets, "
        "observability, and the rollout approach",
        "09_deployment.md", research=True,
    ),
    PhaseSpec(
        "conventions", "Coding Conventions", "Tech Lead",
        "Capture the coding conventions the build must follow (written to AICODER.md).",
        "languages and style, the folder structure, file naming, function/variable "
        "naming, formatting rules, error-handling and logging patterns, "
        "comment/docstring style, and the test layout",
        "AICODER.md", target="conventions",
    ),
]

PHASES_BY_ID = {p.id: p for p in PHASES}
