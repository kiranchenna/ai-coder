"""
phases/api_design.py — Phase 5: API Design
===========================================
Role: Backend Engineer
Goal: Define all REST API endpoints, auth strategy, error format,
      rate limiting, and pagination.
"""

from __future__ import annotations
from phases.base import BasePhase


class APIPhase(BasePhase):
    PHASE_NUM  = 5
    PHASE_NAME = "API Design"
    ROLE       = "backend_engineer"
    PHASE_FILE = "05_api_design.md"

    def get_research_queries(self) -> list[str]:
        return [
            f"REST API design best practices 2025",
            f"API authentication JWT OAuth2 best practices",
            f"API versioning strategy 2025",
            f"API rate limiting best practices",
        ]

    def build_initial_prompt(self, research: str, context: str) -> str:
        return (
            f"We're building: **{self.idea}**\n\n"
            f"You have the full data models and architecture from previous phases. "
            f"As my Backend Engineer, please:\n\n"
            f"1. Define the **base URL structure** and versioning strategy\n"
            f"2. Choose and justify the **authentication method** "
            f"   (JWT / OAuth2 / API keys / session-based)\n"
            f"3. List ALL endpoints organised by resource:\n"
            f"   | Method | Path | Auth? | Description | Request Body | Response |\n"
            f"4. Define the **standard error response format** (JSON)\n"
            f"5. Define **pagination** approach (cursor or offset+limit)\n"
            f"6. Specify **rate limiting** strategy\n"
            f"7. List any **WebSocket or real-time endpoints** if needed\n"
            f"8. Define CORS policy\n\n"
            f"Derive every endpoint from the feature list and data models. "
            f"Don't add endpoints we don't need."
        )

    def build_summary_prompt(self, transcript: str) -> str:
        return (
            f"Based on the following Backend Engineer conversation, produce a structured "
            f"**API Specification** in Markdown:\n\n"
            f"## API Design: {self.project}\n"
            f"### Base URL & Versioning\n"
            f"### Authentication Strategy\n"
            f"### Standard Response & Error Format\n"
            f"### Endpoints by Resource\n"
            f"(For each resource: heading + endpoint table with Method, Path, Auth, Description, "
            f"Request, Response)\n"
            f"### Pagination\n"
            f"### Rate Limiting\n"
            f"### Real-time / WebSocket Endpoints (if any)\n"
            f"### CORS Policy\n\n"
            f"Conversation:\n\n{transcript}"
        )
