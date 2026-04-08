"""
phases/frontend.py — Phase 6: Frontend Planning
================================================
Role: Frontend Engineer + UX Designer
Goal: Plan all pages, components, framework, state management,
      routing, and key user flows.
"""

from __future__ import annotations
from phases.base import BasePhase


class FrontendPhase(BasePhase):
    PHASE_NUM  = 6
    PHASE_NAME = "Frontend Planning"
    ROLE       = "frontend_engineer"
    PHASE_FILE = "06_frontend_planning.md"

    def get_research_queries(self) -> list[str]:
        return [
            f"best frontend framework {self.idea} app 2025",
            f"React Next.js {self.idea} tutorial setup 2025",
            f"UI component library modern web app 2025",
            f"{self.idea} UX design patterns best practices",
            f"state management React 2025 best practice",
        ]

    def build_initial_prompt(self, research: str, context: str) -> str:
        return (
            f"We're building: **{self.idea}**\n\n"
            f"You have the full API design, data models, and feature list. "
            f"As my Frontend Engineer and UX Designer, please:\n\n"
            f"1. Recommend the **frontend framework** with justification "
            f"   (Next.js / Vite+React / Vue / SvelteKit / Flutter, etc.)\n"
            f"2. Recommend the **UI library / component system** "
            f"   (Tailwind CSS / shadcn-ui / MUI / Chakra, etc.)\n"
            f"3. Recommend **state management** approach\n"
            f"4. List ALL pages/screens with:\n"
            f"   | Page | Route | Auth Required? | Key Components | API Calls |\n"
            f"5. Describe the **routing structure** (nested routes, layouts)\n"
            f"6. Walk through the **user flow** for the 3 most important features step-by-step\n"
            f"7. Describe how **authentication** is handled client-side "
            f"   (token storage, protected routes, redirects)\n"
            f"8. List any **custom components** that need to be built\n\n"
            f"Use the latest framework versions and official setup methods."
        )

    def build_summary_prompt(self, transcript: str) -> str:
        return (
            f"Based on the following Frontend planning conversation, produce a structured "
            f"**Frontend Specification** in Markdown:\n\n"
            f"## Frontend: {self.project}\n"
            f"### Framework & Tooling\n"
            f"| Tool | Choice | Version | Why |\n"
            f"|---|---|---|---|\n"
            f"### UI Library & Design System\n"
            f"### State Management\n"
            f"### Pages / Screens\n"
            f"(Table: Page name, Route, Auth required, Key components, API calls)\n"
            f"### Routing Structure\n"
            f"### Authentication (Client Side)\n"
            f"### Key User Flows\n"
            f"### Custom Components to Build\n"
            f"### Project Setup Commands\n\n"
            f"Conversation:\n\n{transcript}"
        )
