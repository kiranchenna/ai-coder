"""
core/model.py — Chat model factory with native tool calling
============================================================
Single place that constructs the local Ollama chat model and binds tools to
it for native function/tool calling (the agentic core depends on this).

Replaces the prompt-and-regex-parse approach of the old pipeline: the model
now returns structured tool calls that the agent loop executes directly.
"""

from __future__ import annotations

from typing import Sequence

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


def get_chat_model(precise: bool = False, tools: Sequence | None = None):
    """
    Build a ChatOllama instance from the active config.

    Args:
        precise: Use the low-temperature setting (for code/edits) instead of
                 the conversational temperature.
        tools:   Optional list of LangChain tools to bind for native tool
                 calling. When provided, the returned runnable yields
                 AIMessages whose ``.tool_calls`` lists the model's requests.

    Returns:
        A ChatOllama (or its tool-bound runnable) ready for ``.invoke()``.
    """
    from core.config import get_config

    cfg = get_config()
    llm = ChatOllama(
        model=cfg.model_name,
        base_url=cfg.model_base_url,
        temperature=cfg.model_temperature_precise if precise else cfg.model_temperature,
        num_ctx=cfg.model_context_length,
    )
    if tools:
        return llm.bind_tools(list(tools))
    return llm


def selftest() -> bool:
    """
    Phase 0 smoke test: verify the configured model can do native tool calling.

    Binds a trivial ``get_time`` tool and checks that the model chooses to call
    it. Returns True on success. Prints a human-readable result.
    """
    from langchain_core.tools import tool
    from rich.console import Console

    console = Console()

    @tool
    def get_time() -> str:
        """Return the current server time as an ISO 8601 string."""
        return "2026-01-01T00:00:00Z"

    from core.config import get_config

    model_name = get_config().model_name
    console.print(f"[dim]Tool-calling self-test against [bold]{model_name}[/bold]…[/dim]")

    try:
        llm = get_chat_model(tools=[get_time])
        ai = llm.invoke(
            [HumanMessage(content="What time is it right now? Use the get_time tool to find out.")]
        )
    except Exception as e:
        console.print(f"[red]✗ Model call failed: {e}[/red]")
        console.print("[dim]Is Ollama running, and is the model pulled?[/dim]")
        return False

    calls = getattr(ai, "tool_calls", None) or []
    if calls:
        console.print(f"[green]✓ Native tool calling works[/green] — model requested: "
                      f"{', '.join(c.get('name', '?') for c in calls)}")
        return True

    console.print(
        "[yellow]✗ Model responded without calling the tool.[/yellow]\n"
        "[dim]This model may not support native tool calling well. "
        "Try a stronger one, e.g.: ollama pull qwen2.5-coder:7b[/dim]"
    )
    return False
