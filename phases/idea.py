"""
phases/idea.py — Phase 1: Idea Refinement
==========================================
Role: Product Manager
Goal: Turn a rough idea into a clear, scoped product definition
      with a finalised feature list.
"""

from __future__ import annotations
from phases.base import BasePhase


class IdeaPhase(BasePhase):
    PHASE_NUM  = 1
    PHASE_NAME = "Idea Refinement"
    ROLE       = "product_manager"
    PHASE_FILE = "01_idea_refinement.md"

    def get_research_queries(self) -> list[str]:
        return [
            f"{self.idea} must-have features 2025",
            f"{self.idea} product requirements modern app",
            f"{self.idea} user expectations latest trends",
        ]

    def build_initial_prompt(self, research: str, context: str) -> str:
        return (
            f"I have an idea: **{self.idea}**\n\n"
            f"As my Product Manager, please:\n"
            f"1. Ask me the 5–7 most important clarifying questions to understand the product fully\n"
            f"   (target users, platform, core problem, monetisation, scope)\n"
            f"2. Based on the research data you have, suggest features I may not have thought of\n"
            f"3. Help me arrive at a clear product brief with a finalised feature list\n\n"
            f"Start with your clarifying questions."
        )

    def build_summary_prompt(self, transcript: str) -> str:
        return (
            f"Based on the following Product Manager conversation, produce a structured "
            f"**Product Brief** in Markdown with these sections:\n\n"
            f"## Product Brief: {self.project}\n"
            f"### Elevator Pitch (1–2 sentences)\n"
            f"### Target Users\n"
            f"### Problem Being Solved\n"
            f"### Core MVP Features (essential for launch)\n"
            f"### Future Roadmap Features (v2+)\n"
            f"### Platform & Constraints\n"
            f"### Success Metrics\n\n"
            f"Conversation:\n\n{transcript}"
        )
