"""
core/repl.py — Main REPL loop for aicoder
==========================================
The heart of the CLI: reads user input, dispatches slash commands or
handles natural language conversation with full workspace context.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.markdown import Markdown

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

console = Console()


# ─── System prompt for the natural-language chat mode ─────────────────────────

REPL_SYSTEM = """\
You are AICoder — an expert AI coding assistant running locally via Ollama.

Your role:
- Answer coding questions clearly and concisely
- Write, debug, and improve code
- Explain code and architectural decisions
- Suggest the best tools, libraries, and approaches

When you need to create or modify files, use this EXACT format:
===FILE: relative/path/to/file===
<complete file content here>
===END===

After all files, always add:
===SUMMARY===
What changed: <brief description>
Files: <list>
===END===

Rules:
- STRICT REQUIREMENT: Before you provide your final output, you MUST wrap your step-by-step internal reasoning and planning inside `<think>` and `</think>` tags.
- Write COMPLETE file content, not partial snippets.
- Be direct and action-oriented outside of the think block.
- If asked about the workspace, use the provided context.
- For questions, just answer — no need for file blocks.\
"""


# ─── Banner ───────────────────────────────────────────────────────────────────

def print_banner(workspace: Path, model_name: str) -> None:
    console.print(
        Panel.fit(
            "[bold magenta]✨ AICoder[/bold magenta] [dim]— AI Coding Assistant[/dim]\n"
            f"[dim]Model:[/dim]     [cyan]{model_name}[/cyan]\n"
            f"[dim]Workspace:[/dim] [bold]{workspace}[/bold]\n"
            "[dim]Type [bold]/help[/bold] to see all commands. Type anything to chat.[/dim]",
            border_style="magenta",
        )
    )


# ─── Command dispatch ─────────────────────────────────────────────────────────

def dispatch_slash_command(raw_input: str, ctx) -> bool:
    """
    Parse and run a slash command.
    Returns True if a slash command was matched, False otherwise.
    """
    from commands.registry import registry

    if not raw_input.startswith("/"):
        return False

    # Split: /command args
    parts    = raw_input[1:].split(None, 1)
    cmd_name = parts[0].lower() if parts else ""
    args     = parts[1] if len(parts) > 1 else ""

    cmd = registry.get(cmd_name)
    if cmd is None:
        console.print(f"[red]Unknown command: /{cmd_name}[/red]  [dim]Type /help to see all commands[/dim]")
        return True

    try:
        cmd.handler(args, ctx)
    except KeyboardInterrupt:
        console.print("\n[yellow]Command interrupted.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error in /{cmd_name}: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")

    return True


# ─── Natural language handler ─────────────────────────────────────────────────

def handle_natural_language(user_input: str, ctx, ws_summary: str) -> None:
    """
    Handle a natural language message: build context, stream AI response,
    parse any file blocks, and write them to disk.
    """
    from core.streaming import stream_response
    from core.memory import summarize_history_if_needed
    from tools.file_tools import parse_file_blocks, extract_summary_block, write_files_with_review

    # Build the conversation history for this turn
    messages: list[BaseMessage] = []

    if not ctx.history:
        # First message — inject workspace context
        system_with_context = (
            REPL_SYSTEM
            + f"\n\n--- CURRENT WORKSPACE ---\n{ws_summary}\n--- END WORKSPACE ---"
        )
        messages.append(SystemMessage(content=system_with_context))
    else:
        # Auto-summarize if history is getting too long
        ctx.history = summarize_history_if_needed(ctx.history)
        messages = list(ctx.history)

    # Add this user message
    messages.append(HumanMessage(content=user_input))
    ctx.history.append(HumanMessage(content=user_input))

    # Optional: Auto-inject relevant cached RAG knowledge for this query
    try:
        from core.knowledge import KnowledgeBase
        kb = KnowledgeBase()
        rag_results = kb.search(user_input, n=2, ttl_hours=8760)
        if rag_results:
            rag_context = "\n\n".join(r["content"] for r in rag_results)
            # We inject it as a hidden SystemMessage just for this turn so the AI knows it,
            # but we don't save this bulky text to the permanent scrollback history.
            messages.append(SystemMessage(
                content=f"Use this relevant documentation from your knowledge base if it helps answer the query:\n{rag_context}"
            ))
    except Exception:
        pass

    # Stream response
    response = stream_response(messages, label="🤖 AI")
    ctx.history.append(AIMessage(content=response))

    # Check if AI output any files
    files = parse_file_blocks(response)
    if files:
        console.print(Rule("[dim]Reviewing changes…[/dim]"))
        written = write_files_with_review(files, ctx.workspace)
        summary = extract_summary_block(response)
        if summary and written:
            console.print(Panel(summary, title="[bold green]Files Written[/bold green]", border_style="green"))


# ─── Main REPL function ───────────────────────────────────────────────────────

def run_repl(workspace: Path) -> None:
    """
    Start the interactive REPL loop.

    Args:
        workspace: The project root directory to work in
    """
    from core.config   import get_config
    from core.context  import WorkspaceContext
    from core.memory   import load_history, save_history
    from commands.handlers import REPLContext

    cfg = get_config()

    # Register all commands
    _register_all_commands()

    # Build workspace context
    console.print("[dim]Scanning workspace…[/dim]")
    ws_ctx = WorkspaceContext(workspace)
    ws_ctx.build()

    # Load persistent memory
    history, notes = load_history(workspace)
    if history:
        turns = len([m for m in history if isinstance(m, HumanMessage)])
        console.print(f"[dim]✔ Resumed session — {turns} previous turns[/dim]")

    # Print banner
    console.print()
    print_banner(workspace, cfg.model_name)
    console.print()

    # Create context object shared across handlers
    ctx = REPLContext(workspace=workspace, history=history, ws_context=ws_ctx)
    ctx.notes = notes

    # Pre-build the workspace summary for the first AI message
    ws_summary = ws_ctx.summary

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        console.print()
        try:
            user_input = Prompt.ask("[bold yellow]>[/bold yellow]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Saving session…[/dim]")
            save_history(workspace, ctx.history, notes=ctx.notes)
            console.print("[bold green]Goodbye! 👋[/bold green]")
            break

        if not user_input:
            continue

        # ── Exit commands ──────────────────────────────────────────────────────
        if user_input.lower() in ("exit", "quit", "bye", "q"):
            save_history(workspace, ctx.history, notes=ctx.notes)
            console.print("[bold green]Session saved. Goodbye! 👋[/bold green]")
            break

        # ── Slash commands ─────────────────────────────────────────────────────
        if user_input.startswith("/"):
            dispatch_slash_command(user_input, ctx)
            save_history(workspace, ctx.history, notes=ctx.notes)
            continue

        # ── Natural language ───────────────────────────────────────────────────
        handle_natural_language(user_input, ctx, ws_summary)
        save_history(workspace, ctx.history, notes=ctx.notes)


# ─── Command registration ─────────────────────────────────────────────────────

def _register_all_commands() -> None:
    """Register all slash commands with the global registry."""
    from commands.registry import registry
    from commands.handlers import (
        handle_help, handle_build, handle_fix, handle_improve,
        handle_research, handle_versions, handle_stack,
        handle_explain, handle_review, handle_run,
        handle_context, handle_memory, handle_shell_mode,
        handle_config, handle_checkdeps, handle_docs,
        # New commands
        handle_new, handle_test, handle_git, handle_load,
        handle_github, handle_diff_mode,
        # Pipeline commands
        handle_project, handle_knowledge,
    )

    def reg(name, desc, usage, handler, aliases=None):
        registry.register(name, desc, usage, handler, aliases)

    reg("help",       "Show all commands",                               "/help",                          handle_help,      ["h", "?"])
    reg("project",    "Launch the 7-phase project planning pipeline",     "/project \"idea\" | resume | list", handle_project,  ["plan", "pipeline"])
    reg("build",      "Launch the 4-phase app builder wizard",           "/build",                         handle_build,     ["wizard"])
    reg("new",        "Create and scaffold a new project",               "/new <name> [description]",      handle_new)
    reg("fix",        "Fix bugs in a file or workspace",                 "/fix [file] [description]",      handle_fix)
    reg("improve",    "Improve code quality and structure",              "/improve [file] [what]",         handle_improve,   ["refactor"])
    reg("research",   "Web-search a topic and get AI summary",           "/research <topic>",              handle_research,  ["r"])
    reg("versions",   "Get latest versions of packages (pip/npm)",       "/versions <pkg> [pkg2]",         handle_versions,  ["ver", "v"])
    reg("stack",      "Recommend tech stack for an app type",            "/stack <app type>",              handle_stack)
    reg("explain",    "Explain a file or concept",                       "/explain [file] [what]",         handle_explain,   ["why"])
    reg("review",     "Code review a file or the workspace",             "/review [file]",                 handle_review)
    reg("test",       "Run tests; /test analyse for AI failure analysis", "/test [analyse|cmd]",           handle_test)
    reg("run",        "Run a shell command in the workspace",            "/run <command>",                  handle_run,       ["exec", "shell"])
    reg("git",        "Git status/diff/log; /git context injects info", "/git [diff|log|context]",        handle_git)
    reg("load",       "Load files into AI conversation context",         "/load <file> [file2]",           handle_load)
    reg("github",     "Get GitHub latest release or search repos",       "/github <owner/repo|search X>",  handle_github,    ["gh"])
    reg("context",    "Show the AI's workspace context",                 "/context [--refresh]",           handle_context,   ["ctx"])
    reg("memory",     "View or clear session memory",                    "/memory [clear]",                handle_memory,    ["mem"])
    reg("shell-mode", "Set shell confirmation mode",                     "/shell-mode [always|never|smart]",handle_shell_mode)
    reg("diff-mode",  "Set file diff review mode",                       "/diff-mode [always|auto|never]", handle_diff_mode)
    reg("config",     "Show current configuration",                      "/config",                        handle_config,    ["cfg"])
    reg("checkdeps",  "Check if dependencies are up to date",            "/checkdeps [file]",              handle_checkdeps, ["deps"])
    reg("docs",       "Fetch and summarize library documentation",        "/docs <library>",                handle_docs)
    reg("knowledge",  "Manage the local RAG knowledge base",              "/knowledge [search|learn|clear]",   handle_knowledge, ["kb"])
