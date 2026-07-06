"""
core/model.py — Chat model factory with native tool calling
============================================================
Single place that constructs the local Ollama chat model and binds tools to
it for native function/tool calling (the agentic core depends on this).

Replaces the prompt-and-regex-parse approach of the old pipeline: the model
now returns structured tool calls that the agent loop executes directly.
"""

from __future__ import annotations

import json
from typing import Sequence

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


# ── Tool-call recovery from text ────────────────────────────────────────────────
# Some local models (e.g. qwen2.5-coder over Ollama) emit tool calls as JSON text
# in the message content instead of via native tool calling. These helpers recover
# them so the agent loop can execute them anyway.

def _balanced_spans(text: str, open_ch: str, close_ch: str) -> list[str]:
    """Extract top-level open_ch...close_ch substrings via matching (string-aware)."""
    spans: list[str] = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch:
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append(text[start : i + 1])
    return spans


def balanced_json_objects(text: str) -> list[str]:
    """Extract top-level {...} substrings via brace matching (string-aware)."""
    return _balanced_spans(text, "{", "}")


def balanced_json_arrays(text: str) -> list[str]:
    """Extract top-level [...] substrings via bracket matching (string-aware)."""
    return _balanced_spans(text, "[", "]")


def extract_text_tool_calls(content: str) -> list[dict]:
    """
    Recover tool calls a model emitted as JSON *text* rather than natively.
    Looks for {"name": ..., "arguments"/"args"/"parameters": {...}} objects.
    Returns dicts shaped like native tool calls: {name, args, id}.
    """
    calls: list[dict] = []
    for candidate in balanced_json_objects(content or ""):
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        args = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
        if args is None:  # explicit null args (e.g. a zero-arg tool) → treat as empty
            args = {}
        if isinstance(args, dict):
            calls.append({"name": obj["name"], "args": args, "id": ""})
    return calls


# A dummy key placeholder for openai_compatible servers that don't check
# authentication at all (most local runtimes) — the OpenAI SDK requires a
# non-empty string be passed even when the server ignores it.
_NO_AUTH_PLACEHOLDER = "not-needed"


def _build_openai_compatible(model_name: str, base_url: str, api_key: str, temperature: float):
    """Build a ChatOpenAI pointed at any OpenAI-compatible endpoint — a local
    server (llama.cpp server, vLLM, LM Studio, text-generation-webui,
    LocalAI, ...) or a hosted API (OpenAI, OpenRouter, Groq, Together, ...).

    Raises RuntimeError with an actionable message if the optional
    `langchain-openai` package isn't installed, rather than silently falling
    back to a different provider than the one explicitly configured.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise RuntimeError(
            "model.provider is 'openai_compatible' but the 'langchain-openai' package "
            'isn\'t installed. Run: pip install "ai-coder[openai]"'
        ) from e

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key or _NO_AUTH_PLACEHOLDER,
        temperature=temperature,
    )


def get_chat_model(precise: bool = False, tools: Sequence | None = None, model: str | None = None):
    """
    Build a chat model from the active config's `model.provider`:
    "ollama" (default, via ChatOllama) or "openai_compatible" (via ChatOpenAI
    pointed at a custom base_url — any local server or hosted API that speaks
    the OpenAI chat-completions protocol).

    Args:
        precise: Use the low-temperature setting (for code/edits) instead of
                 the conversational temperature.
        tools:   Optional list of LangChain tools to bind for native tool
                 calling. When provided, the returned runnable yields
                 AIMessages whose ``.tool_calls`` lists the model's requests.

    Returns:
        A chat model (or its tool-bound runnable) ready for ``.invoke()``.
    """
    from core.config import get_config

    cfg = get_config()
    name = model or cfg.model_name
    temperature = cfg.model_temperature_precise if precise else cfg.model_temperature

    if cfg.model_provider == "openai_compatible":
        llm = _build_openai_compatible(name, cfg.model_base_url, cfg.model_api_key, temperature)
    else:
        llm = ChatOllama(
            model=name,
            base_url=cfg.model_base_url,
            temperature=temperature,
            num_ctx=cfg.model_context_length,  # default 16384; see DEFAULT_CONFIG
        )
    if tools:
        return llm.bind_tools(list(tools))
    return llm


# ── Model discovery (for `/model` and the startup pull-check) ──────────────────

def list_ollama_models(base_url: str) -> list[dict]:
    """
    Query Ollama for the models pulled locally. Returns a list of
    ``{"name": str, "size": int (bytes)}`` sorted by name. Raises (network
    error, bad status) so callers can distinguish "no models" from
    "couldn't reach Ollama" instead of getting an empty list either way.
    """
    import httpx

    resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
    resp.raise_for_status()
    models = resp.json().get("models", [])
    return sorted(
        (
            {"name": m.get("name") or m.get("model", ""), "size": int(m.get("size", 0))}
            for m in models
        ),
        key=lambda m: m["name"],
    )


def is_model_pulled(base_url: str, model_name: str) -> bool | None:
    """
    Whether ``model_name`` (or a close match, ignoring the ``:tag`` suffix) is
    among Ollama's locally pulled models. Returns None if Ollama can't be
    reached — callers should treat that as "unknown", not "not pulled".
    """
    try:
        models = list_ollama_models(base_url)
    except Exception:
        return None
    base = model_name.split(":")[0]
    return any(base in m["name"] for m in models)


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

    native = getattr(ai, "tool_calls", None) or []
    if native:
        console.print(f"[green]✓ Native tool calling works[/green] — model requested: "
                      f"{', '.join(c.get('name', '?') for c in native)}")
        return True

    # Many local models emit the tool call as JSON text instead — the agent loop
    # recovers and executes these, so this still counts as working.
    text_calls = [c for c in extract_text_tool_calls(ai.content or "") if c["name"] == "get_time"]
    if text_calls:
        console.print(
            "[green]✓ Tool calling works via text-call recovery[/green] — this model emits "
            "tool calls as text rather than natively, and the agent handles that."
        )
        return True

    console.print(
        "[yellow]✗ Model responded without calling the tool (natively or as text).[/yellow]\n"
        "[dim]This model may not support tool calling well. "
        "Try a stronger one, e.g.: ollama pull qwen2.5-coder:7b[/dim]"
    )
    return False
