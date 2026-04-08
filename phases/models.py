"""
phases/models.py — Phase 4: Data Models
========================================
Role: Database Architect (DBA)
Goal: Define all entities, fields, types, relationships, and DB engine choice.
"""

from __future__ import annotations
from phases.base import BasePhase


class ModelsPhase(BasePhase):
    PHASE_NUM  = 4
    PHASE_NAME = "Data Models"
    ROLE       = "dba"
    PHASE_FILE = "04_data_models.md"

    def get_research_queries(self) -> list[str]:
        return [
            f"{self.idea} database schema design best practices",
            f"{self.idea} data model entities relationships",
            f"schema design patterns {self.idea} PostgreSQL",
        ]

    def build_initial_prompt(self, research: str, context: str) -> str:
        return (
            f"We're building: **{self.idea}**\n\n"
            f"You have the full feature list and architecture from previous phases. "
            f"As my Database Architect, please:\n\n"
            f"1. Identify ALL entities the app needs from the feature list\n"
            f"2. For EACH entity provide:\n"
            f"   - Field name | Type | Constraints | Description\n"
            f"   - Primary key, foreign keys, unique constraints\n"
            f"   - Important indexes (which fields, why)\n"
            f"3. Define all RELATIONSHIPS with cardinality\n"
            f"4. Provide an ERD diagram in ASCII/text format\n"
            f"5. Confirm the database engine choice from architecture phase\n"
            f"   or recommend a better fit if needed\n"
            f"6. Flag any N+1 query risks or schema design concerns\n"
            f"7. Note any denormalisation decisions and why\n\n"
            f"Derive entities directly from the feature list — don't add unnecessary complexity."
        )

    def build_summary_prompt(self, transcript: str) -> str:
        return (
            f"Based on the following Database Architect conversation, produce a structured "
            f"**Data Model Specification** in Markdown:\n\n"
            f"## Data Models: {self.project}\n"
            f"### Database Engine & Justification\n"
            f"### Entity List\n"
            f"### Entity Schemas\n"
            f"(For each entity: table name, fields table, indexes, constraints)\n"
            f"### Entity Relationship Diagram (ASCII)\n"
            f"### Relationships Summary\n"
            f"### Migration Notes\n"
            f"### Performance Considerations\n\n"
            f"Conversation:\n\n{transcript}"
        )
