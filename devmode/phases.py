"""
devmode/phases.py — The SDLC phase specifications
==================================================
Data-driven: each phase is a role + a focus + an output artifact. The session
engine runs the same discussion loop for every phase (except "review" kind,
which critiques the other phases instead of producing a decision).
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
    research: bool = False   # research current versions/best-practices first
    kind: str = "discussion"  # "discussion" | "review"


PHASES: list[PhaseSpec] = [
    PhaseSpec(
        "product", "Product Vision", "Product Manager",
        "Frame the product vision and what to build first.",
        "the product vision and value proposition, the target users/market, the core problem "
        "being solved, the MVP scope vs. later, and the success metrics",
        "01_product.md",
    ),
    PhaseSpec(
        "competitors", "Market & Competitors", "Market Analyst",
        "Understand the competitive landscape and how to differentiate.",
        "the main existing solutions/competitors, their strengths and gaps, and how this "
        "product differentiates",
        "02_competitors.md", research=True,
    ),
    PhaseSpec(
        "requirements", "Requirements", "Requirements Analyst",
        "Agree the functional requirements and scope.",
        "the problem, primary users and their goals, in-scope vs out-of-scope, key user "
        "stories/features, and measurable success criteria",
        "03_requirements.md",
    ),
    PhaseSpec(
        "architecture", "Architecture & Tech Stack", "Software Architect",
        "Choose the architecture and technology stack.",
        "the overall architecture style and major components, the technology stack with "
        "current stable versions, key trade-offs, and third-party services",
        "04_architecture.md", research=True,
    ),
    PhaseSpec(
        "security", "Security & Non-Functional", "Security & Platform Engineer",
        "Decide security and non-functional requirements.",
        "authentication & authorization, secrets/data protection, scalability and performance "
        "targets, reliability/availability, and any compliance needs",
        "05_security_nfr.md",
    ),
    PhaseSpec(
        "data_model", "Data Model & DB Schema", "Database Architect",
        "Design the data model and database schema.",
        "the entities and relationships, each table/collection's fields and types, keys and "
        "indexes, and the migration approach",
        "06_data_model.md",
    ),
    PhaseSpec(
        "api", "API & Interface Contracts", "Backend Engineer",
        "Define the API / interface contracts.",
        "the endpoints (or module interfaces), request/response shapes, status codes and a "
        "consistent error format, auth per endpoint, and versioning",
        "07_api.md", research=True,
    ),
    PhaseSpec(
        "app_flow", "Application Flow & Business Logic", "Domain/Backend Engineer",
        "Map the application flows and core business logic.",
        "the core user/business flows step by step, business rules and validations, important "
        "edge cases, and key state transitions",
        "08_app_flow.md",
    ),
    PhaseSpec(
        "ui_ux", "UI/UX — Screens & Behaviour", "Frontend & UX Engineer",
        "Define the screens and user behaviour.",
        "the screen/page list, what each shows and does, navigation/routing, and key "
        "interactions and states (loading, empty, error)",
        "09_ui_ux.md", research=True,
    ),
    PhaseSpec(
        "testing", "Testing Strategy", "QA Engineer",
        "Decide the testing strategy.",
        "test levels (unit/integration/e2e), tools, what must be covered, test data, and CI "
        "gating",
        "10_testing.md",
    ),
    PhaseSpec(
        "deployment", "Deployment & Infrastructure", "DevOps Engineer",
        "Plan deployment and infrastructure.",
        "runtime/hosting, build and CI/CD, environments and config/secrets, observability, and "
        "the rollout approach",
        "11_deployment.md", research=True,
    ),
    PhaseSpec(
        "documentation", "Documentation Plan", "Technical Writer",
        "Decide what documentation the project needs.",
        "the docs to produce (README, setup/usage, API reference, architecture, user guide), "
        "their audience and outline, and where they live",
        "12_documentation.md",
    ),
    PhaseSpec(
        "conventions", "Coding Conventions", "Tech Lead",
        "Capture the coding conventions the build must follow (written to AICODER.md).",
        "languages and style, the folder structure, file naming, function/variable naming, "
        "formatting rules, error-handling and logging patterns, comment/docstring style, and "
        "the test layout",
        "AICODER.md", target="conventions",
    ),
    PhaseSpec(
        "review", "Design Review", "Design Reviewer",
        "Critically review all design decisions before building.",
        "consistency across phases (do the schema, API, and flows match the requirements?), "
        "gaps and missing decisions, security and scalability risks, and anything that will "
        "cause problems at build time",
        "design_review.md", kind="review",
    ),
]

PHASES_BY_ID = {p.id: p for p in PHASES}
