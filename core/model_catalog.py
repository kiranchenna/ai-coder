"""
core/model_catalog.py — Curated model recommendations for `/model`
====================================================================
A small, hand-verified catalog of Ollama models well-suited to a local coding
agent, grouped into tiers by hardware / preference (fast vs. balanced vs.
powerful). Every tag here was confirmed to exist on ollama.com/library before
being added — an unverified tag would make `ollama pull` fail for the user, so
don't add one without checking it resolves to a real, pullable model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    tag: str        # exact Ollama tag, e.g. "qwen2.5-coder:7b"
    tier: str        # "fast" | "balanced" | "powerful"
    size_gb: float   # approximate download size
    note: str        # one-line reason it's recommended


TIER_ORDER = ("fast", "balanced", "powerful")

TIER_LABELS = {
    "fast": "Fast & light (~8GB RAM/VRAM)",
    "balanced": "Balanced (~16GB) — the sweet spot",
    "powerful": "Powerful (24GB+)",
}

RECOMMENDED_MODELS: list[ModelSpec] = [
    # ── Fast & light ──
    ModelSpec("qwen2.5-coder:3b", "fast", 1.9,
              "Lightest capable coder in the qwen2.5-coder line"),
    ModelSpec("qwen3:4b", "fast", 2.5,
              "Long 256K context for its size; native tool-calling"),
    ModelSpec("granite4:micro", "fast", 2.1,
              "IBM Granite 4, tuned for tool-calling and code completion"),

    # ── Balanced ──
    ModelSpec("qwen2.5-coder:7b", "balanced", 4.7,
              "The default — best reliability/quality tradeoff at this size"),
    ModelSpec("qwen2.5-coder:14b", "balanced", 9.0,
              "Stronger code reasoning, still comfortable in 16GB"),
    ModelSpec("deepseek-coder-v2:16b", "balanced", 8.9,
              "MoE; strong code-quality claims, 160K context"),
    ModelSpec("gpt-oss:20b", "balanced", 14.0,
              "OpenAI's open-weights model; strong agentic tool-calling"),

    # ── Powerful ──
    ModelSpec("qwen3-coder:30b", "powerful", 19.0,
              "2026 consensus top pick for local coding agents; 256K context"),
    ModelSpec("devstral:24b", "powerful", 14.0,
              "Purpose-built for agentic coding — multi-file edits, debugging"),
    ModelSpec("codestral:latest", "powerful", 13.0,
              "Mistral's dedicated code model, 80+ languages, 32K context"),
    ModelSpec("qwen2.5-coder:32b", "powerful", 20.0,
              "Largest dense model in the qwen2.5-coder family"),
]
