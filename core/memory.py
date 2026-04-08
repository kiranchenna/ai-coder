"""
core/memory.py — Persistent session memory for aicoder
=======================================================
Stores conversation history and project notes in:
    ~/.aicoder/memory/<project_hash>/

Each project (identified by its absolute path) has its own memory,
so context is preserved across sessions within the same project.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from core.config import MEMORY_DIR


def _project_id(root: Path) -> str:
    """Create a stable short identifier for a project path."""
    digest = hashlib.md5(str(root.resolve()).encode()).hexdigest()[:8]
    return f"{root.resolve().name}_{digest}"


def _memory_path(root: Path) -> Path:
    dir_ = MEMORY_DIR / _project_id(root)
    dir_.mkdir(parents=True, exist_ok=True)
    return dir_ / "session.json"


# ─── Save / Load ──────────────────────────────────────────────────────────────

def save_history(root: Path, history: list[BaseMessage], notes: str = "") -> None:
    """
    Persist conversation history to disk.

    Args:
        root:    Project root path
        history: LangChain message objects
        notes:   Optional free-form project notes
    """
    from core.config import get_config
    cfg = get_config()
    if not cfg.memory_enabled:
        return

    serialized = []
    for msg in history:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, HumanMessage):
            role = "human"
        elif isinstance(msg, AIMessage):
            role = "ai"
        else:
            continue
        serialized.append({"role": role, "content": msg.content})

    # Keep only the most recent N messages
    max_h = cfg.memory_max_history
    if len(serialized) > max_h:
        serialized = serialized[-max_h:]

    data = {
        "project_path": str(root.resolve()),
        "saved_at": datetime.now().isoformat(),
        "notes": notes,
        "history": serialized,
    }

    path = _memory_path(root)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_history(root: Path) -> tuple[list[BaseMessage], str]:
    """
    Load conversation history from disk.

    Returns:
        (history, notes) — history is empty list if no saved session found
    """
    from core.config import get_config
    if not get_config().memory_enabled:
        return [], ""

    path = _memory_path(root)
    if not path.exists():
        return [], ""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        role_map: dict[str, type[BaseMessage]] = {
            "system": SystemMessage,
            "human": HumanMessage,
            "ai": AIMessage,
        }
        history: list[BaseMessage] = [
            role_map[e["role"]](content=e["content"])
            for e in data.get("history", [])
            if e.get("role") in role_map
        ]
        return history, data.get("notes", "")
    except Exception:
        return [], ""


def clear_memory(root: Path) -> None:
    """Delete the saved session for this project."""
    path = _memory_path(root)
    if path.exists():
        path.unlink()


def memory_info(root: Path) -> str:
    """Return a human-readable summary of what's stored in memory."""
    path = _memory_path(root)
    if not path.exists():
        return "No saved memory for this project."

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        saved_at = data.get("saved_at", "unknown")[:19].replace("T", " ")
        count = len(data.get("history", []))
        notes = data.get("notes", "")
        lines = [
            f"  Project memory: {_project_id(root)}",
            f"  Last saved:     {saved_at}",
            f"  Messages stored:{count}",
        ]
        if notes:
            lines.append(f"  Notes:          {notes[:120]}")
        return "\n".join(lines)
    except Exception as e:
        return f"Memory file exists but could not be read: {e}"


# ─── Context window management ────────────────────────────────────────────────

def summarize_history_if_needed(
    history: list[BaseMessage],
    threshold_chars: int = 20_000,
) -> list[BaseMessage]:
    """
    If the conversation history is getting too long for the model's context
    window, summarize the oldest messages into a compact SystemMessage and
    replace them, keeping the recent messages intact.

    Args:
        history:          Full conversation history
        threshold_chars:  Total non-system character count that triggers summary

    Returns:
        Possibly shortened history list
    """
    non_system = [m for m in history if not isinstance(m, SystemMessage)]
    total_chars = sum(len(m.content) for m in non_system)

    if total_chars <= threshold_chars or len(non_system) <= 6:
        return history  # Nothing to do

    system_msgs  = [m for m in history if isinstance(m, SystemMessage)]
    # Keep the most recent half; summarize the older half
    split        = len(non_system) // 2
    to_summarize = non_system[:split]
    to_keep      = non_system[split:]

    # Build readable text from messages to summarize
    text_to_summarize = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: "
        + m.content[:400]
        for m in to_summarize
    )

    # One-shot AI call (no streaming — this is background housekeeping)
    from core.streaming import quick_ask
    from rich.console import Console
    Console().print("[dim]📝 Summarizing old conversation to save context…[/dim]")

    summary_text = quick_ask(
        f"Summarize this conversation in ~150 words, preserving all key technical "
        f"decisions, file names, and choices made:\n\n{text_to_summarize}",
        system="You are a conversation summarizer. Be factual and concise.",
    )

    if not summary_text:
        return history  # Summarization failed — keep as is

    summary_msg = SystemMessage(
        content=f"[Summary of earlier conversation]: {summary_text}"
    )
    return system_msgs + [summary_msg] + to_keep

