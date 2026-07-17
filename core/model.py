"""
core/model.py — Chat model factory with native tool calling
============================================================
Single place that constructs the chat model (LM Studio, or any other
OpenAI-compatible local server/hosted API) and binds tools to it for native
function/tool calling (the agentic core depends on this).

Replaces the prompt-and-regex-parse approach of the old pipeline: the model
now returns structured tool calls that the agent loop executes directly.
"""

from __future__ import annotations

import json
from typing import Callable, Sequence

from langchain_core.messages import HumanMessage


# ── Tool-call recovery from text ────────────────────────────────────────────────
# Some local models (e.g. qwen2.5-coder) emit tool calls as JSON text
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
    """Build a ChatOpenAI pointed at any OpenAI-compatible endpoint — LM
    Studio's default, but also any other local server (llama.cpp server,
    vLLM, text-generation-webui, LocalAI, ...) or hosted API (OpenAI,
    OpenRouter, Groq, Together, ...) via model.base_url.

    Raises RuntimeError with an actionable message if the `langchain-openai`
    package isn't installed, rather than crashing on an obscure ImportError.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise RuntimeError(
            "the 'langchain-openai' package isn't installed. Run: pip install langchain-openai"
        ) from e

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key or _NO_AUTH_PLACEHOLDER,
        temperature=temperature,
    )


def get_chat_model(precise: bool = False, tools: Sequence | None = None, model: str | None = None):
    """
    Build a chat model from the active config — a ChatOpenAI pointed at
    model.base_url (LM Studio's default local server, or any other
    OpenAI-compatible endpoint).

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

    llm = _build_openai_compatible(name, cfg.model_base_url, cfg.model_api_key, temperature)
    if tools:
        return llm.bind_tools(list(tools))
    return llm


# ── LM Studio discovery (for `/model`, `/vision model`, and the startup
#    reachability check) ─────────────────────────────────────────────────────
# This shells out to the `lms` CLI: LM Studio's OpenAI-compatible /v1/models
# endpoint does list every model downloaded to disk (confirmed live —
# includes idle, not just loaded, ones), but only returns bare ids, no size
# or the `vision` flag /vision model's picker needs — `lms ls --json` is the
# source for that richer data. Every function here degrades to "unavailable"
# (raises, or returns None) rather than assuming LM Studio specifically is
# what's listening on base_url — model.base_url can still point at any other
# OpenAI-compatible server, where `lms` legitimately won't be on PATH.

def _run_lms(*args: str, timeout: float = 15) -> str:
    """Run an `lms` subcommand and return its stdout. Raises RuntimeError with
    an actionable message on any failure (not on PATH, non-zero exit, or a
    timeout) so callers can fall back to the generic openai_compatible UI."""
    import subprocess

    try:
        result = subprocess.run(
            ["lms", *args], capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "'lms' isn't on your PATH — install LM Studio (lmstudio.ai), which "
            "installs its CLI, or set it up manually with `lms bootstrap`."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"'lms {' '.join(args)}' timed out.") from e
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "lms failed").strip())
    return result.stdout


def list_lmstudio_models(vision_only: bool = False) -> list[dict]:
    """
    Every LLM downloaded to disk in LM Studio, via `lms ls --llm --json`.
    Returns ``{"name": modelKey, "size": bytes, "vision": bool}`` sorted by
    name — ``modelKey`` is what both `lms load` and the OpenAI-compatible
    API's ``model`` field expect. Raises (see ``_run_lms``) so callers can
    distinguish "no models downloaded" from "can't reach LM Studio".
    """
    import json

    out = _run_lms("ls", "--llm", "--json")
    models = json.loads(out or "[]")
    entries = [
        {
            "name": m["modelKey"],
            "size": int(m.get("sizeBytes", 0)),
            "vision": bool(m.get("vision", False)),
        }
        for m in models
    ]
    if vision_only:
        entries = [e for e in entries if e["vision"]]
    return sorted(entries, key=lambda m: m["name"])


def list_lmstudio_embedding_models() -> list[dict]:
    """Every embedding model downloaded to disk, via `lms ls --embedding --json`
    — used to suggest a sensible default when switching to LM Studio, since an
    empty knowledge.embedding_model falls back to the *chat* model (core.config's
    Config.embedding_model), which isn't an embedding model at all."""
    import json

    out = _run_lms("ls", "--embedding", "--json")
    models = json.loads(out or "[]")
    return sorted(
        ({"name": m["modelKey"], "size": int(m.get("sizeBytes", 0))} for m in models),
        key=lambda m: m["name"],
    )


def is_lmstudio_model_downloaded(model_name: str) -> bool | None:
    """Whether ``model_name`` is among LM Studio's locally downloaded models.
    Returns None if LM Studio/`lms` can't be reached."""
    try:
        models = list_lmstudio_models()
    except Exception:
        return None
    return any(m["name"] == model_name for m in models)


def is_lmstudio_model_loaded(name: str) -> bool:
    """Whether `name` is currently loaded *in memory* right now — distinct
    from is_lmstudio_model_downloaded (on disk, whether loaded or not).
    Read-only: never loads/unloads anything itself, just reports current
    state, so a caller can decide whether a (re)load is actually needed
    before paying for one. Returns False (not raises) if LM Studio/`lms`
    can't be reached — "can't tell" is treated the same as "not loaded" so
    callers fall through to their own load attempt rather than silently
    assuming it's fine."""
    import json

    try:
        loaded = json.loads(_run_lms("ps", "--json") or "[]")
    except RuntimeError:
        return False
    return any(m.get("type") == "llm" and m.get("identifier") == name for m in loaded)


LMSTUDIO_DEFAULT_BASE_URL = "http://localhost:1234/v1"

# How long a model can sit unused (no requests — not tied to whether
# `aicoder` itself is still running) before LM Studio unloads it on its own,
# via `lms load`'s native --ttl. This is LM Studio's own idle-tracking, not
# something aicoder polls for: it survives aicoder exiting or crashing, and
# — importantly — won't unload a model something *else* (another aicoder
# session, LM Studio's own chat UI) is still actively using, since real
# requests keep resetting its clock.
LMSTUDIO_IDLE_UNLOAD_SECONDS = 600  # 10 minutes


def is_lmstudio_endpoint(base_url: str) -> bool:
    """Whether `base_url` matches LM Studio's default local server — gates
    every LM-Studio-specific `lms` CLI shellout (auto-start, load/unload,
    model discovery) so none of it is attempted against a different
    OpenAI-compatible server (a custom local server, a hosted API), where
    `lms` legitimately wouldn't apply."""
    return base_url.rstrip("/") == LMSTUDIO_DEFAULT_BASE_URL


def switch_lmstudio_model(name: str, *, timeout: float = 120) -> None:
    """
    Load ``name`` in LM Studio at the configured context length and with the
    idle-unload TTL (LMSTUDIO_IDLE_UNLOAD_SECONDS), unloading any other
    currently-loaded LLM first (embedding models are left alone — RAG needs
    those to stay up) so only one coding model occupies RAM at a time.
    Raises RuntimeError on failure — the caller is responsible for deciding
    whether that's fatal or just a warning (see agent/loop.py's
    _switch_model).

    Always loads with an explicit --context-length matching
    Config.model_context_length. Without this, LM Studio loads at its own
    default (often 4096-8192) regardless of what config.yaml specifies —
    confirmed live, twice, with two different models: aicoder happily builds
    a prompt sized for the configured context, LM Studio silently truncates
    to its own smaller loaded window, and the request fails with a cryptic
    "tokens to keep from the initial prompt is greater than the context
    length" the moment a real system prompt (tools + repo overview +
    memory) doesn't fit — not on `aicoder` startup, on the first real turn.
    If the model's already loaded but at a *different* context length or TTL
    than configured, it's unloaded and reloaded rather than left mismatched.
    """
    import json

    from core.config import get_config

    context_length = get_config().model_context_length
    ttl_ms = LMSTUDIO_IDLE_UNLOAD_SECONDS * 1000

    try:
        loaded = json.loads(_run_lms("ps", "--json") or "[]")
    except RuntimeError:
        loaded = []  # `lms ps` failing isn't fatal here — just skip the unload step

    # `lms load` on an already-loaded modelKey doesn't reuse it — confirmed
    # live: it spins up a second, separately-identified instance (a ":2"
    # suffix) rather than being a no-op, silently doubling RAM usage. Only
    # skip the reload if it's already loaded *and* already at the configured
    # context length and TTL — otherwise it's exactly the mismatch this
    # function exists to prevent (a model loaded without a TTL, e.g. by hand
    # or by an older aicoder run, would otherwise never pick one up and just
    # sit there indefinitely).
    already_loaded = any(
        m.get("type") == "llm" and m.get("identifier") == name
        and m.get("contextLength") == context_length
        and m.get("ttlMs") == ttl_ms
        for m in loaded
    )

    for m in loaded:
        if m.get("type") == "llm" and m.get("identifier") != name:
            try:
                _run_lms("unload", m["identifier"], timeout=timeout)
            except RuntimeError:
                pass  # best-effort — a stuck old model shouldn't block loading the new one

    if not already_loaded:
        # The loop above only unloads *other* models — the target itself
        # needs an explicit unload too if it's the one that's mismatched.
        if any(m.get("type") == "llm" and m.get("identifier") == name for m in loaded):
            try:
                _run_lms("unload", name, timeout=timeout)
            except RuntimeError:
                pass  # best-effort — `lms load` below will still surface any real failure
        _run_lms(
            "load", name,
            "--context-length", str(context_length),
            "--ttl", str(LMSTUDIO_IDLE_UNLOAD_SECONDS),
            "-y",
            timeout=timeout,
        )


def unload_lmstudio_model(name: str, *, timeout: float = 15) -> None:
    """Unload a specific model right away. Used on a clean `aicoder` exit
    (/exit, Ctrl+C/Ctrl+D, the TUI's quit action) — a deliberate "I'm done"
    signal, distinct from LM Studio's own idle-TTL unload (see
    switch_lmstudio_model's --ttl), which is the right default for "still
    running but idle" but shouldn't be the *only* way a model comes down:
    an explicit exit means unload now, not wait out the idle window. Raises
    RuntimeError on failure — the caller (agent/loop.py's
    _try_lmstudio_unload) treats this as fully best-effort, since a stuck or
    failed unload must never block the app from actually exiting."""
    _run_lms("unload", name, timeout=timeout)


def is_lmstudio_reachable(base_url: str) -> set[str] | None:
    """The set of model ids LM Studio reports via its OpenAI-compatible
    /v1/models (every model downloaded to disk, whether idle or loaded —
    see the module docstring above), or None if the server can't be reached
    at all."""
    import httpx

    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=5)
        resp.raise_for_status()
        return {m.get("id", "") for m in resp.json().get("data", [])}
    except Exception:
        return None


def ensure_lmstudio_running(
    model_name: str, *, on_status: Callable[[str], None] | None = None,
) -> bool:
    """Best-effort auto-start: if LM Studio's local server isn't reachable,
    start it (`lms server start`) and load `model_name` — so a fresh
    `aicoder` launch doesn't require manually opening LM Studio and clicking
    Start Server first. Returns True once the server is confirmed reachable
    (whether it needed starting or not); False if it couldn't be brought up
    (LM Studio/`lms` not installed, app not running, ...) — the caller
    (cli.py's preflight check) falls back to its existing warning either way.
    Only ever called against LM Studio's default endpoint (see
    is_lmstudio_endpoint) — never attempted against a custom/remote server.

    `on_status`, if given, is called with short progress messages ("starting
    the server", "loading the model") as each step runs — this is a
    multi-second, otherwise-silent operation (subprocess shellouts + a model
    load), so a caller can surface what's happening instead of it looking
    like the app hung. This module deliberately has no console/UI dependency
    of its own — that's left to the caller (e.g. cli.py) via this callback.
    """
    def emit(msg: str) -> None:
        if on_status is not None:
            on_status(msg)

    if is_lmstudio_reachable(LMSTUDIO_DEFAULT_BASE_URL) is not None:
        return True

    emit("LM Studio's local server isn't running — starting it…")
    try:
        _run_lms("server", "start", timeout=20)
    except RuntimeError as e:
        # Surface *why* (e.g. `lms` not on PATH at all — meaning LM Studio
        # likely isn't installed) rather than silently returning False and
        # leaving the caller's generic "open LM Studio" fallback warning as
        # the only thing shown, which wrongly presumes it's installed.
        emit(str(e))
        return False
    if is_lmstudio_reachable(LMSTUDIO_DEFAULT_BASE_URL) is None:
        emit("Server start reported success, but it's still not reachable.")
        return False

    emit(f"Server started — loading '{model_name}'…")
    try:
        switch_lmstudio_model(model_name)
    except RuntimeError:
        pass  # server's up even if this specific model failed to load — the
              # "isn't available on that server" check right after reports that
    return True


def selftest(console=None) -> bool:
    """
    Phase 0 smoke test: verify the configured model can do native tool calling.

    Binds a trivial ``get_time`` tool and checks that the model chooses to call
    it. Returns True on success. Prints a human-readable result.

    ``console`` defaults to a fresh SafeConsole (the ``aicoder --selftest``
    CLI path, writing straight to the real terminal). ``/doctor`` passes
    ``agent.loop.console`` instead — that singleton gets swapped for a
    RichLog-backed console when the TUI is active, so a locally-created
    console here would silently write to real stdout underneath the TUI's
    alternate screen buffer instead of the visible chat log.
    """
    from langchain_core.tools import tool

    if console is None:
        from core.console import SafeConsole

        console = SafeConsole()

    @tool
    def get_time() -> str:
        """Return the current server time as an ISO 8601 string."""
        return "2026-01-01T00:00:00Z"

    from core.config import get_config

    cfg = get_config()
    model_name = cfg.model_name
    console.print(f"[dim]Tool-calling self-test against [bold]{model_name}[/bold]…[/dim]")

    try:
        llm = get_chat_model(tools=[get_time])
        ai = llm.invoke(
            [HumanMessage(content="What time is it right now? Use the get_time tool to find out.")]
        )
    except Exception as e:
        console.print(f"[red]✗ Model call failed: {e}[/red]")
        console.print(f"[dim]Is the server at {cfg.model_base_url} running, and is "
                      f"'{model_name}' loaded? (LM Studio: `lms load {model_name}`)[/dim]")
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
        "[dim]This model may not support tool calling well. Try a stronger one, e.g.: "
        "lms get qwen2.5-coder-7b-instruct[/dim]"
    )
    return False
