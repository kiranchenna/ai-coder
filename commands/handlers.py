"""
commands/handlers.py — Slash command implementations for aicoder
================================================================
Each handler receives (args: str, ctx: REPLContext) and returns None.

Handler conventions:
  - Print output via console (Rich)
  - Write files via write_file() + console.print()
  - Long-running ops show progress indicators
  - All web/shell ops go through their respective tool modules
"""

from __future__ import annotations

from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.table import Table

console = Console()


# ─── Type alias for the REPL context passed to each handler ──────────────────

class REPLContext:
    """Shared state passed to command handlers."""
    def __init__(self, workspace: Path, history: list, ws_context):
        self.workspace   = workspace
        self.history     = history      # Mutable list of LangChain messages
        self.ws_context  = ws_context   # core.context.WorkspaceContext
        self.notes: str  = ""


# ─────────────────────────────────────────────────────────────────────────────
# /help
# ─────────────────────────────────────────────────────────────────────────────

def handle_help(args: str, ctx: REPLContext) -> None:
    from commands.registry import registry

    table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
    table.add_column("Command", style="bold cyan", no_wrap=True, min_width=22)
    table.add_column("Description")

    for cmd in registry.all_commands():
        alias_str = f" [{', '.join(cmd.aliases)}]" if cmd.aliases else ""
        table.add_row(f"{cmd.usage}{alias_str}", cmd.description)

    console.print()
    console.print(Panel(table, title="[bold magenta]✨ AICoder Commands[/bold magenta]", border_style="magenta"))
    console.print()
    console.print("[dim]Type anything without a / to chat with the AI.[/dim]")
    console.print("[dim]The AI understands your project and can write/fix code directly.[/dim]")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# /build
# ─────────────────────────────────────────────────────────────────────────────

def handle_build(args: str, ctx: REPLContext) -> None:
    """Launch the original 4-phase wizard."""
    import importlib.util, sys
    from pathlib import Path as _Path

    # Find and import main.py from the ai-coder package directory
    pkg_root = _Path(__file__).parent.parent
    main_path = pkg_root / "main.py"

    if not main_path.exists():
        console.print("[red]Could not find main.py wizard.[/red]")
        return

    console.print(Rule("[bold magenta]Launching Wizard Mode[/bold magenta]"))
    spec = importlib.util.spec_from_file_location("main_wizard", main_path)
    mod  = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.main()


# ─────────────────────────────────────────────────────────────────────────────
# /fix
# ─────────────────────────────────────────────────────────────────────────────

def handle_fix(args: str, ctx: REPLContext) -> None:
    """
    Fix bugs in a specific file or across the workspace.
    Usage: /fix [file] [description of the bug]
    """
    from core.streaming import stream_response
    from tools.file_tools import parse_file_blocks, extract_summary_block, write_files_with_review
    from langchain_core.messages import SystemMessage, HumanMessage

    SYSTEM = """You are an expert debugging assistant. Fix the bug described by the user.

When outputting changed files, use this EXACT format:
===FILE: relative/path/to/file===
<complete updated file content>
===END===

Rules:
- Write COMPLETE file content, not just the changed lines
- Only output files that actually changed
- Add ===SUMMARY=== block at the end explaining what was fixed
===SUMMARY===
<description of fix>
Files changed: <list>
===END==="""

    # Parse args: optional file path at start
    parts = args.strip().split(None, 1)
    target_file = ""
    bug_description = args.strip()

    if parts:
        candidate = ctx.workspace / parts[0]
        if candidate.exists() and candidate.is_file():
            target_file = parts[0]
            bug_description = parts[1] if len(parts) > 1 else "Fix any bugs you find."

    # Build context
    if target_file:
        try:
            file_content = (ctx.workspace / target_file).read_text(encoding="utf-8", errors="replace")
            code_context = f"===FILE: {target_file}===\n{file_content}\n===END==="
        except Exception:
            code_context = f"[Could not read {target_file}]"
    else:
        console.print("[dim]Scanning workspace for context…[/dim]")
        code_context = ctx.ws_context.collect_files_for_ai(max_files=15)

    prompt = (
        f"Bug description: {bug_description or 'Fix any obvious bugs.'}\n\n"
        f"Current code:\n\n{code_context}"
    )

    console.print(Rule("[bold red]🔧 Fixing Bug[/bold red]"))
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]
    response = stream_response(messages, label="🔧 AI Fix", precise=True)

    files = parse_file_blocks(response)
    if files:
        console.print(Rule("[dim]Reviewing changes…[/dim]"))
        written = write_files_with_review(files, ctx.workspace)
        summary = extract_summary_block(response)
        if summary:
            console.print(Panel(summary, title="[bold green]Fix Applied[/bold green]", border_style="green"))
        console.print(f"  [green]✔ {len(written)} file(s) written[/green]")
    else:
        console.print("[dim](No file changes — see AI response above)[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# /improve
# ─────────────────────────────────────────────────────────────────────────────

def handle_improve(args: str, ctx: REPLContext) -> None:
    """
    Improve code quality, add error handling, refactor, etc.
    Usage: /improve [file] [what to improve]
    """
    from core.streaming import stream_response
    from tools.file_tools import parse_file_blocks, extract_summary_block, write_files_with_review
    from langchain_core.messages import SystemMessage, HumanMessage

    SYSTEM = """You are a senior engineer doing a code improvement pass.
Improve the provided code: better error handling, cleaner structure, type hints,
documentation, performance, security — based on user instructions.

Output format:
===FILE: path===
<complete improved file>
===END===

===SUMMARY===
Improvements made: <list>
===END==="""

    parts = args.strip().split(None, 1)
    target_file = ""
    instruction = args.strip() or "Improve code quality, add error handling, improve readability."

    if parts:
        candidate = ctx.workspace / parts[0]
        if candidate.exists() and candidate.is_file():
            target_file = parts[0]
            instruction = parts[1] if len(parts) > 1 else instruction

    if target_file:
        try:
            content = (ctx.workspace / target_file).read_text(encoding="utf-8", errors="replace")
            code_context = f"===FILE: {target_file}===\n{content}\n===END==="
        except Exception:
            code_context = "[Could not read file]"
    else:
        console.print("[dim]Scanning workspace…[/dim]")
        code_context = ctx.ws_context.collect_files_for_ai(max_files=10)

    prompt = f"Instruction: {instruction}\n\nCode:\n\n{code_context}"

    console.print(Rule("[bold cyan]✨ Improving Code[/bold cyan]"))
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]
    response = stream_response(messages, label="✨ Improvement", precise=True)

    files = parse_file_blocks(response)
    if files:
        console.print(Rule("[dim]Reviewing changes…[/dim]"))
        written = write_files_with_review(files, ctx.workspace)
        summary = extract_summary_block(response)
        if summary:
            console.print(Panel(summary, title="[bold cyan]Improvements Applied[/bold cyan]", border_style="cyan"))
        console.print(f"  [green]✔ {len(written)} file(s) written[/green]")


# ─────────────────────────────────────────────────────────────────────────────
# /research
# ─────────────────────────────────────────────────────────────────────────────

def handle_research(args: str, ctx: REPLContext) -> None:
    """
    Research a topic: searches the web and synthesizes findings.
    Usage: /research <topic>
    """
    from core.streaming import stream_response
    from tools.web_tools import search_web, format_search_results, fetch_url
    from langchain_core.messages import SystemMessage, HumanMessage

    topic = args.strip()
    if not topic:
        console.print("[yellow]Usage: /research <topic>[/yellow]")
        return

    console.print(Rule(f"[bold blue]🔍 Researching: {topic}[/bold blue]"))

    # Search
    console.print("[dim]Searching the web…[/dim]")
    results = search_web(topic)
    search_text = format_search_results(results)

    # Fetch top result for more detail
    extra = ""
    if results:
        url = results[0].get("href") or results[0].get("url", "")
        if url:
            console.print(f"[dim]Fetching top result: {url}[/dim]")
            extra = fetch_url(url)

    SYSTEM = """You are a technical researcher. Synthesize the web research below into a 
clear, actionable summary. Focus on:
- The key findings and current best practices
- Specific versions, tools, or libraries mentioned
- Practical recommendations for a developer
Keep it concise but comprehensive. Use markdown formatting."""

    prompt = (
        f"Research topic: {topic}\n\n"
        f"Search results:\n{search_text}\n\n"
        f"Top source content:\n{extra[:3000] if extra else 'N/A'}"
    )

    console.print()
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]
    stream_response(messages, label="📚 Research Summary", precise=False)


# ─────────────────────────────────────────────────────────────────────────────
# /versions
# ─────────────────────────────────────────────────────────────────────────────

def handle_versions(args: str, ctx: REPLContext) -> None:
    """
    Get the latest version of one or more packages.
    Usage: /versions <package1> [package2 ...]
    Supports both pip (PyPI) and npm packages.
    """
    from tools.tech_tools import get_pypi_version, get_npm_version

    packages = args.strip().split()
    if not packages:
        console.print("[yellow]Usage: /versions <package> [package2 ...][/yellow]")
        return

    console.print(Rule("[bold green]📦 Package Versions[/bold green]"))

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Package",  style="bold cyan")
    table.add_column("Registry", style="dim")
    table.add_column("Latest Version", style="bold green")
    table.add_column("Details")

    for pkg in packages:
        # Try PyPI first, then npm
        info = get_pypi_version(pkg)
        registry = "PyPI"
        if "error" in info:
            info = get_npm_version(pkg)
            registry = "npm"

        if "error" in info:
            table.add_row(pkg, "—", "[red]not found[/red]", info["error"])
        else:
            details = info.get("summary") or info.get("description") or ""
            if len(details) > 60:
                details = details[:60] + "..."
            table.add_row(pkg, registry, info["version"], details)

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# /stack
# ─────────────────────────────────────────────────────────────────────────────

def handle_stack(args: str, ctx: REPLContext) -> None:
    """
    Recommend the best tech stack for a given application type.
    Usage: /stack <app type>
    """
    from core.streaming import stream_response
    from tools.tech_tools import recommend_stack
    from langchain_core.messages import SystemMessage, HumanMessage

    app_type = args.strip()
    if not app_type:
        console.print("[yellow]Usage: /stack <app type>[/yellow]")
        console.print("[dim]Examples: /stack REST API, /stack mobile app, /stack realtime chat[/dim]")
        return

    console.print(Rule(f"[bold yellow]🏗  Stack for: {app_type}[/bold yellow]"))
    console.print("[dim]Researching current best practices…[/dim]")

    research_data = recommend_stack(app_type)

    SYSTEM = """You are a senior software architect. Based on the research data below, 
recommend the best technology stack for the described application type.

Format your response as:
## Recommended Stack: <App Type>

### Primary Recommendation
| Layer | Technology | Version | Why |
|-------|-----------|---------|-----|
...

### Alternative Options
Brief mention of 1-2 alternatives with tradeoffs.

### Quick Start
The exact commands to scaffold the project.

Be specific about versions. Prefer stable, production-proven choices."""

    prompt = f"App type: {app_type}\n\nResearch data:\n{research_data}"

    console.print()
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]
    stream_response(messages, label="🏗 Stack Recommendation")


# ─────────────────────────────────────────────────────────────────────────────
# /explain
# ─────────────────────────────────────────────────────────────────────────────

def handle_explain(args: str, ctx: REPLContext) -> None:
    """
    Explain a file or a concept from the codebase.
    Usage: /explain [file] [what to explain]
    """
    from core.streaming import stream_response
    from langchain_core.messages import SystemMessage, HumanMessage

    parts = args.strip().split(None, 1)
    target_file = ""
    question = args.strip() or "Explain how this code works."

    if parts:
        candidate = ctx.workspace / parts[0]
        if candidate.exists() and candidate.is_file():
            target_file = parts[0]
            question = parts[1] if len(parts) > 1 else "Explain how this file works."

    if target_file:
        try:
            content = (ctx.workspace / target_file).read_text(encoding="utf-8", errors="replace")
            code_context = f"===FILE: {target_file}===\n{content}\n===END==="
        except Exception:
            code_context = "[Could not read file]"
    else:
        code_context = ctx.ws_context.summary

    SYSTEM = "You are an expert code explainer. Give clear, concise explanations suitable for developers. Use examples."

    prompt = f"{question}\n\nCode context:\n{code_context}"

    console.print(Rule("[bold blue]💡 Explanation[/bold blue]"))
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]
    stream_response(messages, label="💡 Explanation")


# ─────────────────────────────────────────────────────────────────────────────
# /review
# ─────────────────────────────────────────────────────────────────────────────

def handle_review(args: str, ctx: REPLContext) -> None:
    """
    Code review a file or the whole codebase.
    Usage: /review [file]
    """
    from core.streaming import stream_response
    from langchain_core.messages import SystemMessage, HumanMessage

    target_file = args.strip()

    if target_file:
        candidate = ctx.workspace / target_file
        if candidate.exists():
            try:
                content = candidate.read_text(encoding="utf-8", errors="replace")
                code_context = f"===FILE: {target_file}===\n{content}\n===END==="
            except Exception:
                code_context = "[Could not read file]"
        else:
            console.print(f"[yellow]File not found: {target_file}[/yellow]")
            return
    else:
        console.print("[dim]Collecting workspace files for review…[/dim]")
        code_context = ctx.ws_context.collect_files_for_ai(max_files=10)

    SYSTEM = """You are a senior code reviewer. Perform a thorough review covering:
- Bugs and potential errors
- Security vulnerabilities  
- Performance issues
- Code quality and readability
- Missing error handling
- Best practice violations

Format: use ## sections, rate severity as 🔴 Critical, 🟡 Warning, 🟢 Suggestion."""

    prompt = f"Please review the following code:\n\n{code_context}"

    console.print(Rule("[bold yellow]🔍 Code Review[/bold yellow]"))
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]
    stream_response(messages, label="🔍 Review")


# ─────────────────────────────────────────────────────────────────────────────
# /run
# ─────────────────────────────────────────────────────────────────────────────

def handle_run(args: str, ctx: REPLContext) -> None:
    """
    Run a shell command in the workspace directory.
    Usage: /run <command>
    """
    from tools.shell_tools import run_with_confirmation

    command = args.strip()
    if not command:
        console.print("[yellow]Usage: /run <command>[/yellow]")
        return

    console.print(Rule("[bold]🔧 Shell Execution[/bold]"))
    result = run_with_confirmation(command, cwd=ctx.workspace)
    if result is None:
        return

    stdout, stderr, code = result
    if stderr.strip():
        console.print(f"\n[red]stderr:[/red]")
        console.print(stderr[:2000], markup=False)
    if code != 0:
        console.print(f"\n[red]Exit code: {code}[/red]")
    else:
        console.print(f"\n[green]✔ Exit code: 0[/green]")


# ─────────────────────────────────────────────────────────────────────────────
# /context
# ─────────────────────────────────────────────────────────────────────────────

def handle_context(args: str, ctx: REPLContext) -> None:
    """Show the AI's current workspace context."""
    console.print(Rule("[bold]📁 Workspace Context[/bold]"))
    force_rebuild = args.strip().lower() in ("--refresh", "-r", "refresh")
    if force_rebuild:
        console.print("[dim]Rebuilding context…[/dim]")
        ctx.ws_context.build()
    console.print(Markdown(ctx.ws_context.summary))


# ─────────────────────────────────────────────────────────────────────────────
# /memory
# ─────────────────────────────────────────────────────────────────────────────

def handle_memory(args: str, ctx: REPLContext) -> None:
    """
    View or clear session memory.
    Usage: /memory          — show memory info
           /memory clear    — wipe memory for this project
    """
    from core.memory import memory_info, clear_memory, save_history

    sub = args.strip().lower()

    if sub == "clear":
        clear_memory(ctx.workspace)
        ctx.history.clear()
        console.print("[green]✔ Memory cleared for this project.[/green]")
    else:
        info = memory_info(ctx.workspace)
        console.print(Panel(info, title="[bold]🧠 Memory[/bold]", border_style="dim"))
        console.print(f"  Conversation turns this session: {len(ctx.history)}")


# ─────────────────────────────────────────────────────────────────────────────
# /shell-mode
# ─────────────────────────────────────────────────────────────────────────────

def handle_shell_mode(args: str, ctx: REPLContext) -> None:
    """
    View or change the shell command confirmation mode.
    Usage: /shell-mode              — show current mode
           /shell-mode always       — always ask before running
           /shell-mode never        — auto-run without asking
           /shell-mode smart        — ask only for destructive commands
    """
    from core.config import get_config
    from tools.shell_tools import show_shell_mode

    mode = args.strip().lower()
    if not mode:
        show_shell_mode()
        console.print()
        console.print("  Change with: [bold]/shell-mode always[/bold] | [bold]never[/bold] | [bold]smart[/bold]")
        return

    try:
        get_config().set_shell_confirmation(mode)
        console.print(f"[green]✔ Shell mode set to:[/green] [bold cyan]{mode}[/bold cyan]")
        show_shell_mode()
    except ValueError as e:
        console.print(f"[red]{e}[/red]")


# ─────────────────────────────────────────────────────────────────────────────
# /config
# ─────────────────────────────────────────────────────────────────────────────

def handle_config(args: str, ctx: REPLContext) -> None:
    """Show current configuration."""
    from core.config import get_config, CONFIG_PATH
    import yaml

    cfg = get_config()
    console.print(Rule("[bold]⚙ Configuration[/bold]"))
    console.print(f"[dim]Config file: {CONFIG_PATH}[/dim]\n")

    raw = cfg.raw()
    yaml_str = __import__("yaml").dump(raw, default_flow_style=False, sort_keys=False)
    console.print(f"```yaml\n{yaml_str}```", markup=False)
    console.print()
    console.print(f"[dim]Edit config file to change settings: {CONFIG_PATH}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# /checkdeps
# ─────────────────────────────────────────────────────────────────────────────

def handle_checkdeps(args: str, ctx: REPLContext) -> None:
    """
    Check if project dependencies are up to date.
    Usage: /checkdeps [requirements.txt | package.json]
    """
    from tools.tech_tools import check_requirements_versions

    target = args.strip() or "requirements.txt"
    dep_file = ctx.workspace / target

    if not dep_file.exists():
        console.print(f"[yellow]File not found: {target}[/yellow]")
        return

    console.print(Rule(f"[bold green]📦 Checking: {target}[/bold green]"))

    if "requirements" in target or target.endswith(".txt"):
        content = dep_file.read_text(encoding="utf-8")
        results = check_requirements_versions(content)

        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("Package",        style="bold cyan")
        table.add_column("Pinned",         style="dim")
        table.add_column("Latest",         style="bold")
        table.add_column("Status")

        for r in results:
            if "error" in r:
                table.add_row(r["package"], "—", "—", f"[red]{r['error']}[/red]")
            elif r.get("outdated"):
                table.add_row(
                    r["package"], r["pinned_version"], r["latest_version"],
                    "[yellow]⬆ Update available[/yellow]"
                )
            else:
                table.add_row(
                    r["package"], r["pinned_version"], r["latest_version"],
                    "[green]✔ Up to date[/green]"
                )

        console.print(table)
    else:
        console.print("[yellow]Currently only requirements.txt is supported.[/yellow]")


# ─────────────────────────────────────────────────────────────────────────────
# /docs
# ─────────────────────────────────────────────────────────────────────────────

def handle_docs(args: str, ctx: REPLContext) -> None:
    """
    Fetch and display official documentation for a library.
    Usage: /docs <library>
    """
    from core.streaming import stream_response
    from tools.web_tools import fetch_docs
    from langchain_core.messages import SystemMessage, HumanMessage

    library = args.strip()
    if not library:
        console.print("[yellow]Usage: /docs <library>[/yellow]")
        return

    console.print(Rule(f"[bold blue]📖 Docs: {library}[/bold blue]"))
    console.print("[dim]Fetching documentation…[/dim]")

    raw_docs = fetch_docs(library)

    SYSTEM = "You are a technical writer. Summarize the key points from this documentation in a developer-friendly way. Focus on: installation, basic usage, key concepts, and common patterns."
    prompt = f"Library: {library}\n\nRaw docs:\n{raw_docs[:5000]}"

    console.print()
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]
    stream_response(messages, label=f"📖 {library} Docs")


# ───────────────────────────────────────────────────────────────────────────────
# /new
# ───────────────────────────────────────────────────────────────────────────────

def handle_new(args: str, ctx: REPLContext) -> None:
    """
    Create a new project directory and scaffold it with AI.
    Usage: /new <project-name> [description]
    Example: /new my-api FastAPI REST API with PostgreSQL
    """
    from core.streaming import stream_response
    from tools.file_tools import parse_file_blocks, extract_summary_block, write_files_with_review
    from langchain_core.messages import SystemMessage, HumanMessage

    parts = args.strip().split(None, 1)
    if not parts:
        console.print("[yellow]Usage: /new <project-name> [description][/yellow]")
        return

    project_name = parts[0].strip()
    description  = parts[1].strip() if len(parts) > 1 else ""

    # Create project directory inside workspace
    project_dir = ctx.workspace / project_name
    if project_dir.exists():
        console.print(f"[yellow]Directory already exists: {project_dir}[/yellow]")
        return

    project_dir.mkdir(parents=True)
    console.print(f"[green]✔ Created directory:[/green] {project_dir}")

    desc_str = description or f"a {project_name} project"
    console.print(Rule(f"[bold magenta]🏗 Scaffolding: {project_name}[/bold magenta]"))

    SYSTEM = """You are a senior developer scaffolding a new project.
Create a minimal but production-ready project structure.

Output each file using:
===FILE: filename===
<complete file content>
===END===

Always include: README.md, .gitignore, and the main entry file.
For Python: requirements.txt + main entry file.
For Node.js: package.json + index file.
Keep files concise — this is a starter scaffold, not a complete app.

===SUMMARY===
Project type: <type>
Files created: <list>
To run: <exact command>
===END==="""

    prompt = (
        f"Create a scaffold for: {desc_str}\n"
        f"Project name: {project_name}\n"
        "Generate only the essential starter files."
    )

    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]
    response  = stream_response(messages, label="🏗 Scaffolding", precise=True)

    files = parse_file_blocks(response)
    if files:
        console.print(Rule("[dim]Writing scaffold files…[/dim]"))
        # Write into the new project subdirectory
        write_files_with_review(files, project_dir, force_accept=True)
        summary = extract_summary_block(response)
        if summary:
            console.print(
                Panel(
                    summary,
                    title=f"[bold magenta]✨ {project_name} created[/bold magenta]",
                    border_style="magenta",
                )
            )
    else:
        console.print("[yellow]No files were generated. Try again with more detail.[/yellow]")


# ───────────────────────────────────────────────────────────────────────────────
# /test
# ───────────────────────────────────────────────────────────────────────────────

def handle_test(args: str, ctx: REPLContext) -> None:
    """
    Detect and run the project's test suite; optionally analyse failures with AI.
    Usage: /test          — auto-detect and run tests
           /test analyse  — run tests and ask AI to explain failures
           /test <cmd>    — run a specific test command
    """
    from tools.shell_tools import run_with_confirmation, run_command
    from core.streaming import stream_response
    from langchain_core.messages import SystemMessage, HumanMessage

    sub = args.strip()
    analyse = sub.lower() == "analyse" or sub.lower() == "analyze"

    # Auto-detect test command if not given
    if not sub or analyse:
        ws = ctx.workspace
        if (ws / "pytest.ini").exists() or (ws / "setup.cfg").exists() or any(ws.glob("tests/test_*.py")):
            cmd = "pytest -v"
        elif (ws / "package.json").exists():
            cmd = "npm test"
        elif (ws / "Makefile").exists():
            cmd = "make test"
        else:
            cmd = "pytest -v"  # Default fallback
        console.print(f"[dim]Auto-detected test command: {cmd}[/dim]")
    else:
        cmd = sub

    console.print(Rule("[bold green]🧪 Running Tests[/bold green]"))
    result = run_with_confirmation(cmd, cwd=ctx.workspace)
    if result is None:
        return

    stdout, stderr, code = result
    combined = (stdout + "\n" + stderr).strip()

    if code == 0:
        console.print("[bold green]\n✔ All tests passed![/bold green]")
    else:
        console.print(f"[bold red]\n✗ Tests failed (exit code {code})[/bold red]")

        if analyse:
            console.print("[dim]Asking AI to analyse failures…[/dim]")
            SYSTEM = "You are a debugging expert. Analyse these test failures and suggest exact fixes. Be concise."
            prompt = f"Test output:\n{combined[:4000]}"
            messages = [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]
            stream_response(messages, label="🧪 Failure Analysis")


# ───────────────────────────────────────────────────────────────────────────────
# /git
# ───────────────────────────────────────────────────────────────────────────────

def handle_git(args: str, ctx: REPLContext) -> None:
    """
    Git status, diff, log viewer — and inject git context into AI conversation.
    Usage: /git             — show status + recent commits
           /git diff        — show current diff
           /git log         — show recent commit log
           /git context     — inject git info into AI conversation context
    """
    from tools.shell_tools import run_command
    from langchain_core.messages import HumanMessage
    from rich.syntax import Syntax

    sub = args.strip().lower()

    def _run_git(cmd: str) -> str:
        """Run a git command silently and return output."""
        stdout, stderr, _ = run_command(
            cmd, cwd=ctx.workspace, stream_output=False
        )
        return (stdout or stderr or "").strip()

    if not sub or sub == "status":
        console.print(Rule("[bold]🌿 Git Status[/bold]"))
        status = _run_git("git status")
        log    = _run_git("git log --oneline -8")
        console.print(Syntax(status, "text", theme="monokai"))
        if log:
            console.print("\n[bold]Recent commits:[/bold]")
            console.print(Syntax(log, "text", theme="monokai"))

    elif sub == "diff":
        console.print(Rule("[bold yellow]🔀 Git Diff[/bold yellow]"))
        diff = _run_git("git diff")
        if diff:
            console.print(Syntax(diff, "diff", theme="monokai"))
        else:
            console.print("[dim]No unstaged changes.[/dim]")

    elif sub == "log":
        console.print(Rule("[bold]📜 Git Log[/bold]"))
        log = _run_git("git log --oneline -20")
        console.print(Syntax(log or "(no commits)", "text", theme="monokai"))

    elif sub == "context":
        # Inject git info into the AI conversation as a HumanMessage
        status = _run_git("git status")
        diff   = _run_git("git diff --stat")
        log    = _run_git("git log --oneline -5")
        context_msg = (
            f"Git context for the current workspace:\n"
            f"STATUS:\n{status}\n\n"
            f"DIFF STAT:\n{diff}\n\n"
            f"RECENT COMMITS:\n{log}"
        )
        ctx.history.append(HumanMessage(content=context_msg))
        console.print("[green]✔ Git context injected into conversation.[/green]")
        console.print(Panel(context_msg, title="[bold]Git Context[/bold]", border_style="dim"))

    else:
        # Run arbitrary git subcommand
        result = _run_git(f"git {sub}")
        console.print(Syntax(result, "text", theme="monokai"))


# ───────────────────────────────────────────────────────────────────────────────
# /load
# ───────────────────────────────────────────────────────────────────────────────

def handle_load(args: str, ctx: REPLContext) -> None:
    """
    Explicitly load one or more files into the AI conversation context.
    Usage: /load <file1> [file2 ...]
    Example: /load src/auth.py src/models.py
    """
    from langchain_core.messages import HumanMessage

    files = args.strip().split()
    if not files:
        console.print("[yellow]Usage: /load <file1> [file2 ...][/yellow]")
        return

    loaded: list[str] = []
    blocks: list[str] = []

    for rel_path in files:
        path = ctx.workspace / rel_path
        if not path.exists():
            console.print(f"  [yellow]Not found:[/yellow] {rel_path}")
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            blocks.append(f"===FILE: {rel_path}===\n{content}\n===END===")
            loaded.append(rel_path)
            console.print(f"  [green]✔[/green] Loaded: [bold]{rel_path}[/bold]")
        except Exception as e:
            console.print(f"  [red]Error reading {rel_path}: {e}[/red]")

    if blocks:
        msg = (
            f"The user has loaded the following file(s) for your context:\n\n"
            + "\n\n".join(blocks)
        )
        ctx.history.append(HumanMessage(content=msg))
        console.print(f"[green]✔ {len(loaded)} file(s) added to conversation context.[/green]")


# ───────────────────────────────────────────────────────────────────────────────
# /github
# ───────────────────────────────────────────────────────────────────────────────

def handle_github(args: str, ctx: REPLContext) -> None:
    """
    Get GitHub repo latest release or search for repos.
    Usage: /github <owner>/<repo>     — get latest release
           /github search <query>     — search GitHub via DuckDuckGo
    """
    from tools.tech_tools import get_github_latest
    from tools.web_tools import search_web, format_search_results

    sub = args.strip()
    if not sub:
        console.print("[yellow]Usage: /github <owner/repo> OR /github search <query>[/yellow]")
        return

    if sub.lower().startswith("search "):
        query = sub[7:].strip()
        console.print(Rule(f"[bold]🐙 GitHub Search: {query}[/bold]"))
        console.print("[dim]Searching…[/dim]")
        results = search_web(f"site:github.com {query}", max_results=6)
        if results:
            for i, r in enumerate(results, 1):
                title = r.get("title", "Untitled")
                url   = r.get("href") or r.get("url", "")
                body  = r.get("body", "")[:120]
                console.print(f"  [bold cyan]{i}.[/bold cyan] [bold]{title}[/bold]")
                console.print(f"     [dim]{url}[/dim]")
                console.print(f"     {body}")
                console.print()
        else:
            console.print("[dim]No results found.[/dim]")
        return

    # Expect owner/repo format
    if "/" not in sub:
        console.print("[yellow]Expected format: /github owner/repo[/yellow]")
        return

    owner, repo = sub.split("/", 1)
    console.print(Rule(f"[bold]🐙 GitHub: {owner}/{repo}[/bold]"))
    console.print("[dim]Fetching latest release…[/dim]")

    info = get_github_latest(owner.strip(), repo.strip())
    if "error" in info:
        console.print(f"[red]{info['error']}[/red]")
        return

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold dim")
    table.add_column(style="bold")
    table.add_row("Repository",   f"{owner}/{repo}")
    table.add_row("Latest tag",   info.get("tag_name", "unknown"))
    table.add_row("Release name", info.get("name", ""))
    table.add_row("Published",    (info.get("published_at") or "")[:10])
    table.add_row("Release URL",  info.get("url", ""))
    console.print(Panel(table, title="[bold]Latest Release[/bold]", border_style="dim"))


# ───────────────────────────────────────────────────────────────────────────────
# /diff-mode
# ───────────────────────────────────────────────────────────────────────────────

def handle_diff_mode(args: str, ctx: REPLContext) -> None:
    """
    View or change how AI-generated file changes are reviewed.
    Usage: /diff-mode             — show current mode
           /diff-mode always      — show diff and ask [y/N] before each change
           /diff-mode auto        — show diff but apply automatically (default)
           /diff-mode never       — skip diff, write immediately
    """
    from core.config import get_config

    mode = args.strip().lower()
    cfg  = get_config()

    descriptions = {
        "always": "show diff and ask [y/N] before each changed file",
        "auto":   "show diff but apply automatically (informational)",
        "never":  "skip diff preview, write immediately",
    }

    if not mode:
        current = cfg.file_confirmation
        console.print(f"  File diff mode: [bold cyan]{current}[/bold cyan] — {descriptions.get(current, '')}")
        console.print()
        console.print("  Change with: [bold]/diff-mode always[/bold] | [bold]auto[/bold] | [bold]never[/bold]")
        return

    try:
        cfg.set_file_confirmation(mode)
        console.print(f"[green]✔ Diff mode set to:[/green] [bold cyan]{mode}[/bold cyan] — {descriptions.get(mode, '')}")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")


# ───────────────────────────────────────────────────────────────────────────────
# /project
# ───────────────────────────────────────────────────────────────────────────────

def handle_project(args: str, ctx: REPLContext) -> None:
    """
    Launch the professional 7-phase project planning pipeline.

    Usage:
        /project                                 — auto-resume if in-progress, else show help
        /project "social media for developers"   — start a new project
        /project resume [name]                   — resume an in-progress project
        /project list                            — list all in-progress projects
    """
    from core.pipeline import Pipeline, find_resumable_projects
    import json as _json, re as _re

    sub = args.strip()

    # ── No args: auto-detect in-progress session ──────────────────────────────
    if not sub:
        projects = find_resumable_projects(ctx.workspace)

        if not projects:
            console.print(
                Panel(
                    "[bold]Start a new project:[/bold]\n"
                    "  [cyan]/project[/cyan] [dim]\"your idea description\"[/dim]\n\n"
                    "[dim]Example:[/dim]\n"
                    "  [cyan]/project[/cyan] [dim]\"task manager for developer teams\"[/dim]",
                    title="[bold magenta]🚀 Project Pipeline[/bold magenta]",
                    border_style="magenta",
                )
            )
            return

        if len(projects) == 1:
            # Single in-progress project — auto-offer to resume
            p = projects[0]
            console.print(
                Panel(
                    f"  Project:  [bold cyan]{p['name']}[/bold cyan]\n"
                    f"  Idea:     {p['idea']}\n"
                    f"  Progress: [green]{p['progress']}[/green]\n"
                    f"  Updated:  [dim]{p['updated_at']}[/dim]",
                    title="[bold magenta]🚀 Resume Previous Session?[/bold magenta]",
                    border_style="magenta",
                )
            )
            from rich.prompt import Confirm
            if Confirm.ask("Continue from where you left off?", default=True):
                sf    = ctx.workspace / "process" / p["name"] / "state.json"
                state = _json.loads(sf.read_text())
                Pipeline(ctx.workspace, p["name"], p["idea"]).run(
                    start_phase=state.get("current_phase", 1)
                )
            return

        # Multiple — show a numbered picker
        console.print()
        console.print("[bold magenta]🚀 In-progress projects:[/bold magenta]")
        for i, p in enumerate(projects, 1):
            console.print(
                f"  [bold cyan]{i}.[/bold cyan] [bold]{p['name']}[/bold] — "
                f"{p['idea'][:50]} [dim]({p['progress']})[/dim]"
            )
        console.print()
        console.print("[dim]  n  Start a new project instead[/dim]")
        console.print()
        choice = console.input("Pick a number (or 'n' for new): ").strip().lower()
        if choice == "n":
            idea = console.input("Describe your idea: ").strip()
            if idea:
                sub = f'"{idea}"'
            else:
                return
        else:
            try:
                p = projects[int(choice) - 1]
            except (ValueError, IndexError):
                console.print("[red]Invalid choice.[/red]")
                return
            sf    = ctx.workspace / "process" / p["name"] / "state.json"
            state = _json.loads(sf.read_text())
            Pipeline(ctx.workspace, p["name"], p["idea"]).run(
                start_phase=state.get("current_phase", 1)
            )
            return

    # ── List ──────────────────────────────────────────────────────────────────
    if sub.lower() == "list":

        projects = find_resumable_projects(ctx.workspace)
        if not projects:
            console.print("[dim]No in-progress projects found in this workspace.[/dim]")
            return

        table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
        table.add_column("Project",       style="bold cyan")
        table.add_column("Idea",          style="dim", max_width=40)
        table.add_column("Progress",      style="green")
        table.add_column("Current Phase", style="yellow")
        table.add_column("Last Updated",  style="dim")
        for p in projects:
            table.add_row(
                p["name"], p["idea"], p["progress"],
                str(p["current_phase"]), p["updated_at"]
            )
        console.print()
        console.print(Panel(table, title="[bold]🚀 In-Progress Projects[/bold]", border_style="magenta"))
        console.print("[dim]Resume with: /project resume <name>[/dim]")
        return

    # ── Resume ────────────────────────────────────────────────────────────────
    if sub.lower().startswith("resume"):
        project_name = sub[6:].strip().strip('"\'') or ""

        if not project_name:
            projects = find_resumable_projects(ctx.workspace)
            if not projects:
                console.print("[yellow]No in-progress projects found.[/yellow]")
                return
            if len(projects) == 1:
                project_name = projects[0]["name"]
                console.print(f"[dim]Resuming: {project_name}[/dim]")
            else:
                console.print("Multiple projects found:")
                for i, p in enumerate(projects, 1):
                    console.print(f"  {i}. {p['name']} — {p['idea']}")
                idx = console.input("Which one? (number) ").strip()
                try:
                    project_name = projects[int(idx) - 1]["name"]
                except (ValueError, IndexError):
                    console.print("[red]Invalid selection.[/red]")
                    return

        # Load state to find the idea and current phase
        state_file = ctx.workspace / "process" / project_name / "state.json"
        if not state_file.exists():
            console.print(f"[red]No state file found for project: {project_name}[/red]")
            return

        import json
        state        = json.loads(state_file.read_text())
        idea         = state.get("idea", project_name)
        start_phase  = state.get("current_phase", 1)

        pipeline = Pipeline(ctx.workspace, project_name, idea)
        pipeline.run(start_phase=start_phase)
        return

    # ── New project ───────────────────────────────────────────────────────────
    idea = sub.strip('"\'').strip()
    if not idea:
        console.print("[yellow]Usage: /project \"your idea description\"[/yellow]")
        console.print("[dim]  or:   /project resume[/dim]")
        console.print("[dim]  or:   /project list[/dim]")
        return

    # Derive a clean project name from the idea
    import re
    project_name = re.sub(r"[^a-zA-Z0-9\-_]", "-", idea[:40]).strip("-").lower()
    project_name = re.sub(r"-+", "-", project_name)

    # Check for existing project
    state_file = ctx.workspace / "process" / project_name / "state.json"
    if state_file.exists():
        from rich.prompt import Confirm
        import json
        state       = json.loads(state_file.read_text())
        done_phases = sum(1 for p in state["phases"].values() if p["status"] == "done")
        console.print(
            f"[yellow]A project named '{project_name}' already exists "
            f"({done_phases}/7 phases done).[/yellow]"
        )
        if Confirm.ask("Resume existing project?", default=True):
            pipeline = Pipeline(ctx.workspace, project_name, idea)
            pipeline.run(start_phase=state.get("current_phase", 1))
            return
        else:
            # Let user pick a new name
            from rich.prompt import Prompt as P
            project_name = P.ask("Enter a different project name").strip().lower()
            project_name = re.sub(r"[^a-zA-Z0-9\-_]", "-", project_name)

    pipeline = Pipeline(ctx.workspace, project_name, idea)
    pipeline.run()


# ───────────────────────────────────────────────────────────────────────────────
# /knowledge
# ───────────────────────────────────────────────────────────────────────────────

def handle_knowledge(args: str, ctx: REPLContext) -> None:
    """
    Manage the local vector knowledge base (RAG cache).

    Usage:
        /knowledge              — show stats
        /knowledge search <q>   — search stored knowledge
        /knowledge learn <q>    — fetch URL or search topic and store in knowledge base
        /knowledge clear        — clear all cached knowledge
    """
    from core.knowledge import KnowledgeBase

    sub = args.strip()
    kb  = KnowledgeBase.get()

    if not sub or sub == "info":
        info = kb.info()
        console.print(
            Panel(
                f"  Total cached documents: [bold green]{info['total_documents']}[/bold green]\n"
                f"  Storage path: [dim]{info['storage_path']}[/dim]\n\n"
                f"  Manage with:\n"
                f"  [dim]/knowledge search <query>[/dim]\n"
                f"  [dim]/knowledge learn <topic_or_url>[/dim]\n"
                f"  [dim]/knowledge clear[/dim]",
                title="[bold]🧠 Knowledge Base[/bold]",
                border_style="dim",
            )
        )

    elif sub.lower().startswith("search "):
        query   = sub[7:].strip()
        results = kb.search(query, n=5, ttl_hours=8760)   # 1 year — show all
        if not results:
            console.print("[dim]No results found.[/dim]")
            return
        console.print(Rule(f"[bold]Knowledge Search: {query}[/bold]"))
        for i, r in enumerate(results, 1):
            meta    = r["metadata"]
            preview = r["content"][:200].replace("\n", " ")
            console.print(
                f"  [bold cyan]{i}.[/bold cyan] "
                f"[dim]source={meta.get('source')} | phase={meta.get('phase', '—')} | "
                f"dist={r['distance']:.3f}[/dim]"
            )
            console.print(f"     {preview}…")
            console.print()

    elif sub.lower().startswith("learn "):
        query = sub[6:].strip()
        console.print(Rule(f"[bold]🧠 Learning: {query}[/bold]"))
        console.print("[dim]Fetching and storing knowledge…[/dim]")

        # Check if it's a URL
        import re
        url_match = re.match(r'^https?://[^\s]+', query)
        if url_match:
            from tools.web_tools import fetch_url
            url = url_match.group(0)
            console.print(f"  [dim]📄 Fetching URL: {url}[/dim]")
            content = fetch_url(url)
            if content and len(content) > 100:
                kb.store_document(
                    url=url,
                    content=content[:10000], # Store a good chunk
                    title=f"Learned from {url}",
                    ttl_hours=8760,  # 1 year
                )
                console.print(f"[green]✔ Successfully learned content from {url}[/green]")
            else:
                console.print(f"[red]Failed to fetch or extract meaningful content from {url}[/red]")
        else:
            # It's a topic
            result = kb.fetch_and_store(
                query=query,
                ttl_hours=8760,  # 1 year
                max_results=3,
                fetch_top_url=True
            )
            if "[No search results" in result:
                console.print(f"[red]Could not find reliable information for: {query}[/red]")
            else:
                console.print(f"[green]✔ Successfully researched and stored knowledge about: {query}[/green]")

    elif sub.lower() == "clear":
        from rich.prompt import Confirm
        if Confirm.ask("[bold red]Clear ALL cached knowledge?[/bold red]", default=False):
            import shutil
            from core.knowledge import CHROMA_DIR
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
            KnowledgeBase._instance = None
            console.print("[green]✔ Knowledge base cleared.[/green]")
        else:
            console.print("[dim]Cancelled.[/dim]")

    else:
        console.print("[yellow]Usage: /knowledge | /knowledge search <q> | /knowledge learn <q> | /knowledge clear[/yellow]")

