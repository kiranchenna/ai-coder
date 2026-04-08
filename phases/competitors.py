"""
phases/competitors.py — Phase 2: Competitor Analysis
=====================================================
Role: Market Analyst
Goal: Research the competitive landscape, build a feature comparison table,
      and identify gaps and opportunities.
"""

from __future__ import annotations
from phases.base import BasePhase


class CompetitorPhase(BasePhase):
    PHASE_NUM  = 2
    PHASE_NAME = "Competitor Analysis"
    ROLE       = "market_analyst"
    PHASE_FILE = "02_competitor_analysis.md"

    def get_research_queries(self) -> list[str]:
        return [
            f"top {self.idea} apps 2025 comparison",
            f"best {self.idea} software market leaders",
            f"{self.idea} user reviews complaints missing features",
            f"{self.idea} alternative tools ProductHunt",
        ]

    def build_initial_prompt(self, research: str, context: str) -> str:
        return (
            f"We're building: **{self.idea}**\n\n"
            f"As my Market Analyst, please:\n"
            f"1. Identify the top 3–5 competitors based on your research\n"
            f"2. Create a **feature comparison table** (Feature | Our App | Competitor A | B | C)\n"
            f"3. Highlight what competitors do WELL that we must match\n"
            f"4. Highlight their WEAKNESSES and gaps we can exploit\n"
            f"5. Recommend which features from competitors we should include in our feature list\n"
            f"6. Suggest our key differentiator based on the gaps you see\n\n"
            f"Use the research data you have and be specific — name real apps and real features."
        )

    def build_summary_prompt(self, transcript: str) -> str:
        return (
            f"Based on the following Market Analyst conversation, produce a structured "
            f"**Competitor Analysis Report** in Markdown:\n\n"
            f"## Competitor Analysis: {self.project}\n"
            f"### Market Overview\n"
            f"### Competitor Profiles (one sub-section each: name, key features, pricing, weaknesses)\n"
            f"### Feature Comparison Table\n"
            f"### Market Gaps & Opportunities\n"
            f"### Our Differentiator\n"
            f"### Final Recommended Feature List (updated from Phase 1)\n\n"
            f"Conversation:\n\n{transcript}"
        )
