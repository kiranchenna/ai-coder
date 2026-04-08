"""
core/streaming.py — Streaming LLM output helpers
=================================================
Wraps LangChain Ollama with live token-by-token streaming to the terminal.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage
from rich.console import Console

console = Console()

# Module-level LLM instances — created lazily on first use
_llm: ChatOllama | None = None
_llm_precise: ChatOllama | None = None


def _get_llm(precise: bool = False) -> ChatOllama:
    """Return (and lazily create) the appropriate LLM instance."""
    global _llm, _llm_precise
    from core.config import get_config
    cfg = get_config()

    if precise:
        if _llm_precise is None:
            _llm_precise = ChatOllama(
                model=cfg.model_name,
                base_url=cfg.model_base_url,
                temperature=cfg.model_temperature_precise,
                num_ctx=cfg.model_context_length,
            )
        return _llm_precise
    else:
        if _llm is None:
            _llm = ChatOllama(
                model=cfg.model_name,
                base_url=cfg.model_base_url,
                temperature=cfg.model_temperature,
                num_ctx=cfg.model_context_length,
            )
        return _llm


def reset_llm() -> None:
    """Reset cached LLM instances (call after config changes)."""
    global _llm, _llm_precise
    _llm = None
    _llm_precise = None


def stream_response(
    messages: list[BaseMessage],
    label: str = "🤖 AI",
    precise: bool = False,
    show_label: bool = True,
) -> str:
    """
    Stream an LLM response token-by-token to the terminal.

    Args:
        messages:   Conversation history (LangChain message objects)
        label:      Label shown above the streamed response
        precise:    Use low-temperature model (for code/spec output)
        show_label: Whether to print the label header

    Returns:
        The full response text as a single string
    """
    llm = _get_llm(precise=precise)

    if show_label:
        console.print(f"\n[bold cyan]{label}:[/bold cyan]")

    full_response = ""
    
    from rich.status import Status
    from rich.markup import escape
    import sys

    first_token = True
    # Match the Ollama desktop app's standard loading phrase
    status = Status("[dim italic]💭 Thinking...[/dim italic]", console=console)
    status.start()

    in_think = False
    buffer = ""

    try:
        for chunk in llm.stream(messages):
            if first_token:
                status.stop()
                first_token = False

            # LangChain sometimes separates reasoning from content in newer Ollama builds
            text = chunk.content or ""
            reasoning = chunk.additional_kwargs.get("reasoning_content", "")
            
            # Construct a streamable text block
            if reasoning:
                text = f"[dim]{escape(reasoning)}[/dim]"

            if reasoning:
                # Bypass the <think> parser if we already get formatted reasoning
                console.print(text, end="", markup=False)
                sys.stdout.flush()
                continue
            
            full_response += text
            buffer += text
            
            while buffer:
                idx = buffer.find("<")
                if idx > 0:
                    # Print everything up to the first '<'
                    chunk_to_print = buffer[:idx]
                    if in_think:
                        console.print(f"[dim]{escape(chunk_to_print)}[/dim]", end="")
                    else:
                        console.print(chunk_to_print, end="", markup=False)
                    # Flush to prevent terminal buffering block
                    sys.stdout.flush()
                    buffer = buffer[idx:]
                    continue
                
                if idx == 0:
                    # Buffer starts with '<'
                    if buffer.startswith("<think>"):
                        in_think = True
                        console.print("\n[dim italic]💭 Thinking...[/dim italic]\n[dim]", end="")
                        buffer = buffer[7:]
                        continue
                    elif buffer.startswith("</think>"):
                        in_think = False
                        console.print("[/dim]\n", end="")
                        buffer = buffer[8:]
                        continue
                    
                    # Check if it's a partial tag that could become <think> or </think>
                    is_partial = False
                    for tag in ("<think>", "</think>"):
                        if tag.startswith(buffer):
                            is_partial = True
                            break
                            
                    if is_partial:
                        # Break out and wait for more characters from the stream
                        break
                    else:
                        # False alarm, not a think tag. Print the '<' and move on.
                        if in_think:
                            console.print("[dim]<[/dim]", end="")
                        else:
                            console.print("<", end="", markup=False)
                        sys.stdout.flush()
                        buffer = buffer[1:]
                else:
                    # No '<' left in buffer
                    if in_think:
                        console.print(f"[dim]{escape(buffer)}[/dim]", end="")
                    else:
                        console.print(buffer, end="", markup=False)
                    sys.stdout.flush()
                    buffer = ""

        # Print any leftover buffer (if stream ended mid-partial-tag)
        if buffer:
            if in_think:
                console.print(f"[dim]{escape(buffer)}[/dim]", end="")
            else:
                console.print(buffer, end="", markup=False)

    except Exception as e:
        status.stop()
        console.print(f"\n[red]⚠ LLM error: {e}[/red]")
        console.print("[dim]Is Ollama running? Try: ollama serve[/dim]")
        return ""

    if not first_token:
        console.print()  # newline after stream ends
    return full_response


def quick_ask(prompt_text: str, system: str = "", precise: bool = False) -> str:
    """
    Single-turn question — no history, just a prompt and response.
    Returns the full response string.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages: list[BaseMessage] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt_text))

    return stream_response(messages, label="🤖 AI", precise=precise, show_label=False)
