"""
phases/architecture.py — Phase 3: Architecture Design
======================================================
Role: Software Architect
Goal: Choose the right architecture, tech stack, and hosting strategy
      based on the feature requirements from Phases 1 & 2.
"""

from __future__ import annotations
from phases.base import BasePhase


class ArchitecturePhase(BasePhase):
    PHASE_NUM  = 3
    PHASE_NAME = "Architecture Design"
    ROLE       = "software_architect"
    PHASE_FILE = "03_architecture_design.md"

    def get_research_queries(self) -> list[str]:
        return [
            f"{self.idea} architecture best practices 2025",
            f"{self.idea} tech stack modern scalable",
            f"{self.idea} microservices vs monolith 2025",
            f"best backend framework {self.idea} 2025",
            f"best database for {self.idea} app",
        ]

    def build_initial_prompt(self, research: str, context: str) -> str:
        return (
            f"We're building: **{self.idea}**\n\n"
            f"Based on the features and requirements from earlier phases, "
            f"as my Software Architect please:\n\n"
            f"1. Present **2–3 architecture options** (e.g. monolith, microservices, serverless) "
            f"   with pros, cons, and which scale each suits\n"
            f"2. Recommend the best option for THIS project and explain why\n"
            f"3. Propose the full **tech stack** with specific versions:\n"
            f"   - Backend language + framework\n"
            f"   - Database(s) + caching\n"
            f"   - Frontend framework\n"
            f"   - Auth provider\n"
            f"   - Hosting / cloud (cheapest option for startup)\n"
            f"4. Draw a simple **ASCII architecture diagram** showing how services connect\n"
            f"5. Estimate rough monthly infrastructure cost at 100, 10K, 100K users\n\n"
            f"Use the latest stable versions of everything. Prefer proven, production-ready choices."
        )

    def build_summary_prompt(self, transcript: str) -> str:
        return (
            f"Based on the following Software Architect conversation, produce a structured "
            f"**Architecture Decision Record (ADR)** in Markdown:\n\n"
            f"## Architecture: {self.project}\n"
            f"### Architecture Style & Justification\n"
            f"### Tech Stack (with versions)\n"
            f"| Layer | Technology | Version | Why |\n"
            f"|---|---|---|---|\n"
            f"### System Architecture Diagram (ASCII)\n"
            f"### Data Flow\n"
            f"### Hosting & Deployment Strategy\n"
            f"### Key Trade-offs\n"
            f"### Infrastructure Cost Estimate\n"
            f"### What This Architecture Makes Easy vs Hard\n\n"
            f"Conversation:\n\n{transcript}"
        )
