"""
main.py — Interactive Idea-to-App Terminal Tool
=================================================
Powered by: qwen2.5-coder:14b (runs locally via Ollama — no internet/API cost)

HOW IT WORKS (3 Phases + Ongoing Iteration):
─────────────────────────────────────────────
  Phase 1 — DISCUSSION
      You describe your app idea. The AI asks focused questions to gather
      requirements (features, tech stack, architecture, etc.). You can share
      opinions and the AI either agrees or suggests alternatives with reasoning.
      Type 'done' when the discussion feels complete.

  Phase 2 — SPEC GENERATION
      The AI reads the full conversation and produces a detailed Markdown
      specification document. Saved to specs/<timestamp>_<name>.md.
      The session (conversation history) is also saved to specs/<name>_session.json
      so you can resume it later.

  Phase 3 — CODE GENERATION
      The AI reads the spec and writes all source files into output/<name>/.
      Every token the AI writes is streamed live to your terminal — you see
      exactly what's being generated in real time.

  Phase 4 — ITERATION LOOP
      After code is generated, the AI waits for your next instruction.
      You can ask it to add features, fix bugs, refactor, improve code, etc.
      The AI will update/create files accordingly. Type 'exit' to stop.

USAGE:
    python main.py              → Start a new session or resume a saved one
    python main.py --new        → Force a fresh new session

COMMANDS DURING DISCUSSION:
    done      → End discussion and move to spec generation
    show      → Print the conversation history so far
    quit      → Exit the program
"""

import os
import re
import json
import datetime
import argparse
from pathlib import Path

# LangChain's Ollama integration — talks to the local Ollama server
from langchain_ollama import ChatOllama

# LangChain message types used to build conversation history:
#   SystemMessage  — hidden instructions that shape AI behaviour
#   HumanMessage   — messages from the user (or our scripted prompts)
#   AIMessage      — messages from the AI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Rich — beautiful terminal UI library (colors, panels, Markdown rendering, etc.)
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.rule import Rule


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL = "qwen2.5-coder:14b"            # The local Ollama model
BASE_URL = "http://localhost:11434"    # Default Ollama server (running locally)
SPECS_DIR = Path("specs")             # Folder for spec docs + session files
OUTPUT_DIR = Path("output")           # Folder where generated code is written

# Global Rich console — all terminal I/O goes through this for consistent styling
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# LLM Instances
# ─────────────────────────────────────────────────────────────────────────────
# Two temperature settings serve different purposes:
#   temperature=0.4 → slightly creative, natural for back-and-forth conversation
#   temperature=0.1 → very deterministic, best for structured output (specs/code)

llm          = ChatOllama(model=MODEL, base_url=BASE_URL, temperature=0.4)
llm_precise  = ChatOllama(model=MODEL, base_url=BASE_URL, temperature=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Streaming output
# ─────────────────────────────────────────────────────────────────────────────

def stream_response(messages: list, label: str = "🤖 AI", precise: bool = False) -> str:
    """
    Send messages to the AI and stream every token to the terminal as it arrives.

    This replaces the old spinner approach — instead of a "thinking..." block,
    you see the AI's output appear word-by-word in real time, just like in a
    chat interface.

    Args:
        messages: Full conversation history (list of System/Human/AI messages)
        label:    Label prefix shown above the response (e.g. "🤖 AI", "💻 Code")
        precise:  If True, uses the low-temperature model (for spec/code output)

    Returns:
        The complete AI response as a single string (accumulated from chunks)
    """
    model = llm_precise if precise else llm
    console.print(f"\n[bold cyan]{label}:[/bold cyan]")

    full_response = ""
    for chunk in model.stream(messages):        # .stream() yields tokens one-by-one
        text = chunk.content
        full_response += text
        # markup=False prevents Rich from trying to parse < > [] in code as markup
        console.print(text, end="", markup=False)

    console.print()  # newline after the stream ends
    return full_response


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Session persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_session(history: list, app_name: str, spec_text: str = ""):
    """
    Persist the conversation history and spec to a JSON file so it can be resumed.

    The session file is saved at: specs/<app_name>_session.json

    Each message is serialized as {"role": "human"|"ai"|"system", "content": "..."}

    Args:
        history:   Full conversation history
        app_name:  Used to name the session file
        spec_text: The generated spec (saved alongside history so resuming works)
    """
    SPECS_DIR.mkdir(exist_ok=True)
    session_path = SPECS_DIR / f"{app_name}_session.json"

    # Convert LangChain message objects to plain dicts for JSON serialization
    serialized = []
    for msg in history:
        if isinstance(msg, SystemMessage):
            serialized.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            serialized.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            serialized.append({"role": "ai", "content": msg.content})

    data = {
        "app_name": app_name,
        "saved_at": datetime.datetime.now().isoformat(),
        "spec_text": spec_text,
        "history": serialized,
    }
    session_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    console.print(f"[dim]💾 Session saved → {session_path}[/dim]")


def load_session(session_path: Path) -> tuple[list, str, str]:
    """
    Load a previously saved session from disk.

    Args:
        session_path: Path to the _session.json file

    Returns:
        history:  Reconstructed list of LangChain message objects
        app_name: The app name stored in the session
        spec_text: The spec document (empty string if not yet generated)
    """
    data = json.loads(session_path.read_text(encoding="utf-8"))

    # Reconstruct LangChain message objects from the serialized dicts
    history = []
    role_map = {"system": SystemMessage, "human": HumanMessage, "ai": AIMessage}
    for entry in data["history"]:
        cls = role_map.get(entry["role"])
        if cls:
            history.append(cls(content=entry["content"]))

    return history, data["app_name"], data.get("spec_text", "")


def list_saved_sessions() -> list[Path]:
    """Return all _session.json files found in the specs/ folder, sorted newest first."""
    if not SPECS_DIR.exists():
        return []
    return sorted(SPECS_DIR.glob("*_session.json"), key=lambda p: p.stat().st_mtime, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Misc UI
# ─────────────────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Convert an arbitrary string to a safe lowercase filename (max 50 chars)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text[:50]


def print_banner():
    console.print(
        Panel.fit(
            "[bold magenta]✨ Idea-to-App Builder[/bold magenta]\n"
            "[dim]Powered by qwen2.5-coder:14b (local · offline)[/dim]\n"
            "[dim]Discuss → Spec → Build → Iterate[/dim]",
            border_style="magenta",
        )
    )


def print_discussion_commands():
    console.print(
        Panel(
            "[green]done[/green]    → Finalize discussion & generate spec\n"
            "[green]show[/green]    → Print conversation so far\n"
            "[green]quit[/green]    → Exit without saving",
            title="[bold]Commands[/bold]",
            border_style="dim",
        )
    )


def print_iteration_commands():
    console.print(
        Panel(
            "Give instructions like:\n"
            "  • [italic]Add user authentication[/italic]\n"
            "  • [italic]Fix the bug in the login function[/italic]\n"
            "  • [italic]Refactor the database layer[/italic]\n\n"
            "[green]show spec[/green]   → Display the spec document\n"
            "[green]show files[/green]  → List generated files\n"
            "[green]exit[/green]        → Finish session",
            title="[bold]Iteration Mode — What would you like to change?[/bold]",
            border_style="cyan",
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0: Session resume / new session selection
# ─────────────────────────────────────────────────────────────────────────────

def pick_session(force_new: bool) -> tuple[list | None, str, str]:
    """
    On startup, let the user resume a saved session or start fresh.

    If there are no saved sessions, or force_new=True, start fresh.

    Args:
        force_new: If True, skip the resume prompt and start a new session

    Returns:
        history:   Loaded history (or None if starting fresh)
        app_name:  The app name (empty string if fresh start)
        spec_text: Loaded spec (empty string if fresh start)
    """
    sessions = list_saved_sessions()

    # If no sessions exist or user forced new, go straight to new session
    if force_new or not sessions:
        return None, "", ""

    console.print(Rule("[bold]Resume a Previous Session[/bold]"))
    console.print("[dim]Found saved sessions:[/dim]\n")

    # Show up to 5 most recent sessions
    shown = sessions[:5]
    for i, path in enumerate(shown):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            saved_at = data.get("saved_at", "unknown")[:19].replace("T", " ")
            console.print(f"  [bold cyan]{i + 1}.[/bold cyan] [bold]{data['app_name']}[/bold]  [dim]({saved_at})[/dim]")
        except Exception:
            console.print(f"  [bold cyan]{i + 1}.[/bold cyan] {path.name}")

    console.print(f"  [bold cyan]{len(shown) + 1}.[/bold cyan] [italic]Start a new session[/italic]")
    console.print()

    choice = Prompt.ask(
        "[bold yellow]Choose[/bold yellow]",
        choices=[str(i + 1) for i in range(len(shown) + 1)],
        default=str(len(shown) + 1),
    )
    choice_idx = int(choice) - 1

    if choice_idx == len(shown):
        # User picked "new session"
        return None, "", ""

    # Load the selected session
    selected = shown[choice_idx]
    history, app_name, spec_text = load_session(selected)
    console.print(f"\n[bold green]✔ Resumed session:[/bold green] [bold]{app_name}[/bold]\n")
    return history, app_name, spec_text


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Discussion
# ─────────────────────────────────────────────────────────────────────────────

DISCUSSION_SYSTEM = """You are a senior software architect and product consultant. 
Your role is to have an in-depth discussion with the user about their app idea. 

Guidelines:
- Ask ONE or TWO focused questions at a time (don't overwhelm the user)
- When the user shares an opinion, acknowledge it, give honest feedback (pros/cons), 
  and either confirm their choice or suggest a better alternative with clear reasoning
- Keep track of all decisions made (stack, architecture, features, etc.)
- Be conversational and concise — avoid walls of text
- Cover: app type, core features, tech stack, architecture, data storage, APIs, auth, deployment
- Once enough details are gathered, encourage the user to type 'done' to generate the spec"""


def run_discussion(idea: str, existing_history: list | None = None) -> tuple[list, str]:
    """
    Run the interactive discussion loop between the user and the AI.

    If 'existing_history' is provided, the conversation is resumed from where
    it left off (the AI is reminded of the previous context before continuing).

    Args:
        idea:             The user's initial idea (used if starting fresh)
        existing_history: History loaded from a saved session (or None)

    Returns:
        history:  Complete conversation history after the discussion ends
        app_name: Slugified app name for use as filename/folder
    """
    console.print(Rule("[bold magenta]Phase 1: Discussion[/bold magenta]"))
    print_discussion_commands()

    # ── Resume path ──────────────────────────────────────────────────────────
    if existing_history:
        # We already have context — just continue the conversation
        history = existing_history
        app_name = slugify(idea) if idea else "app"

        # Add a HumanMessage telling the AI we're continuing
        resume_prompt = "We are resuming our conversation. Please summarize the key decisions we made so far and ask what we should discuss next."
        history.append(HumanMessage(content=resume_prompt))
        stream_response(history, label="🤖 AI (resuming)")
        history.append(AIMessage(content="[resume summary above]"))  # placeholder for memory

    # ── Fresh start path ─────────────────────────────────────────────────────
    else:
        history = [SystemMessage(content=DISCUSSION_SYSTEM)]
        app_name = slugify(idea)

        # Internal opening — instructs AI how to greet the user.
        # Note: this HumanMessage is scripted by us, not typed by the user.
        opening = (
            f'The user wants to build: "{idea}"\n\n'
            "Start by briefly acknowledging their idea enthusiastically, "
            "then ask your first focused question to understand the core requirements. Be concise."
        )
        history.append(HumanMessage(content=opening))
        response = stream_response(history, label="🤖 AI")
        history.append(AIMessage(content=response))

    # ── Main conversation loop ───────────────────────────────────────────────
    while True:
        console.print()
        user_input = Prompt.ask("[bold yellow]You[/bold yellow]").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            console.print("[dim]Goodbye![/dim]")
            raise SystemExit(0)

        if user_input.lower() == "show":
            # Print full readable transcript
            console.print(Rule("[dim]Conversation So Far[/dim]"))
            for msg in history:
                if isinstance(msg, HumanMessage):
                    console.print(f"[bold yellow]You:[/bold yellow] {msg.content}\n")
                elif isinstance(msg, AIMessage):
                    console.print(f"[bold cyan]AI:[/bold cyan] {msg.content}\n")
            console.print(Rule())
            continue

        if user_input.lower() in ("done", "finalize", "finish"):
            console.print("\n[bold green]✔ Discussion complete. Generating spec…[/bold green]")
            # Auto-save session so user can resume later if needed
            save_session(history, app_name)
            break

        # Regular conversation turn
        history.append(HumanMessage(content=user_input))
        response = stream_response(history, label="🤖 AI")
        history.append(AIMessage(content=response))

    return history, app_name


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Spec Generation
# ─────────────────────────────────────────────────────────────────────────────

SPEC_SYSTEM = """You are a senior software architect. Based on the conversation provided, 
generate a COMPLETE, DETAILED specification document in Markdown. 

The document MUST include these sections:
# App Specification: <App Name>

## 1. Overview
## 2. App Type & Platform
## 3. Core Features
## 4. Technology Stack
   Table: Layer | Technology | Version | Reason
## 5. Architecture
   - Design pattern
   - Folder/project structure (as a tree)
   - Data flow description
## 6. Data Models
## 7. API Design (if applicable)
## 8. User Authentication (if applicable)
## 9. Development Environment Setup
## 10. Build & Deployment
## 11. Open Questions / Future Enhancements

Be extremely thorough. Include exact library versions. 
Do NOT use placeholder values — fill in everything from the discussion."""


def generate_spec(history: list, app_name: str) -> tuple[str, str]:
    """
    Generate and save a detailed Markdown spec document from the discussion.

    Everything the AI writes is streamed live to the terminal so you can
    watch the spec being built token-by-token.

    Args:
        history:  Full discussion history from Phase 1
        app_name: Slugified app name for the output filename

    Returns:
        spec_text:    The full Markdown spec as a string
        spec_filename: Path where the spec was saved
    """
    console.print(Rule("[bold magenta]Phase 2: Specification Document[/bold magenta]"))
    console.print("[dim]Streaming spec generation — watch it build in real time…[/dim]\n")

    # Flatten message history into a readable text block for the prompt
    conversation_summary = "\n".join(
        f"User: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
        for m in history
        if isinstance(m, (HumanMessage, AIMessage))
    )

    spec_messages = [
        SystemMessage(content=SPEC_SYSTEM),
        HumanMessage(
            content=f"Here is the full discussion:\n\n{conversation_summary}\n\n"
                    "Now generate the complete specification document."
        ),
    ]

    # Stream the spec directly — no spinner, you see every word as it's written
    spec_text = stream_response(spec_messages, label="📄 Spec", precise=True)

    # Save to disk
    SPECS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    spec_filename = SPECS_DIR / f"{timestamp}_{app_name}.md"
    spec_filename.write_text(spec_text, encoding="utf-8")

    # Save session with spec included so it can be resumed post-spec
    save_session(history, app_name, spec_text=spec_text)

    console.print()
    console.print(
        Panel(
            f"[green]✔ Spec saved →[/green] [bold]{spec_filename}[/bold]",
            border_style="green",
        )
    )

    return spec_text, str(spec_filename)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Code Generation
# ─────────────────────────────────────────────────────────────────────────────

# The AI is instructed to wrap every file in structured delimiters:
#
#   ===FILE: path/to/file.ext===
#   <file content>
#   ===END===
#
# This lets us reliably split the stream into individual files.
# The AI streams everything live, so you see the code being written in real time.

CODEGEN_SYSTEM = """You are a senior software engineer. Build the application from the spec.

Output each file using this EXACT format — do not deviate:

===FILE: relative/path/to/file.ext===
<complete file content>
===END===

Rules:
- Include ALL files: source, configs, package.json/requirements.txt, README.md, .gitignore, etc.
- Write complete working code — no TODOs, no stubs, no placeholders
- Follow the spec architecture exactly
- After all files, add:

===SUMMARY===
Files created: X
Run instructions: <exact commands to run the app>
===END==="""


ITERATION_SYSTEM = """You are a senior software engineer maintaining a codebase.
The user will give you instructions to modify/improve/fix the application.

When outputting changed or new files, use this EXACT format:

===FILE: relative/path/to/file.ext===
<complete updated file content>
===END===

Rules:
- Only output files that actually changed or are new
- Write the FULL content of every changed file (not just diffs)
- After all files, add:

===SUMMARY===
Changes made: <brief description of what changed>
Files modified: <list>
===END===

If no file changes are needed (e.g. answering a question), just respond normally."""


def parse_file_blocks(response: str) -> dict[str, str]:
    """
    Extract ===FILE: path=== ... ===END=== blocks from AI output.

    Returns a dict mapping relative file path → file content.
    """
    pattern = re.compile(r"===FILE:\s*(.+?)===\n(.*?)===END===", re.DOTALL)
    return {
        match.group(1).strip(): match.group(2)
        for match in pattern.finditer(response)
    }


def extract_summary_block(response: str) -> str:
    """Extract the ===SUMMARY=== block from AI output, or empty string if not found."""
    match = re.search(r"===SUMMARY===\n(.*?)===END===", response, re.DOTALL)
    return match.group(1).strip() if match else ""


def write_files(files: dict[str, str], app_output_dir: Path) -> list[str]:
    """
    Write parsed file blocks to disk under the app output directory.

    Args:
        files:          Dict of {relative_path: content} from parse_file_blocks()
        app_output_dir: Base output directory (output/<app_name>/)

    Returns:
        List of full paths that were written
    """
    written = []
    for rel_path, content in files.items():
        full_path = app_output_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        written.append(str(full_path))
        console.print(f"  [green]✔[/green] [bold]{rel_path}[/bold] written")
    return written


def generate_code(spec_text: str, app_name: str) -> Path:
    """
    Generate all application source files by streaming the AI's output live.

    You see every character the AI types — comments, code, file markers, all of it.
    Once the stream ends, the files are parsed out and written to disk.

    Args:
        spec_text: The full Markdown spec from Phase 2
        app_name:  Slugified app name used as the output sub-folder

    Returns:
        app_output_dir: Path to the folder where all files were written
    """
    console.print(Rule("[bold magenta]Phase 3: Code Generation[/bold magenta]"))
    console.print("[dim]Streaming code — you'll see every file being written as it happens…[/dim]\n")

    app_output_dir = OUTPUT_DIR / app_name
    app_output_dir.mkdir(parents=True, exist_ok=True)

    codegen_messages = [
        SystemMessage(content=CODEGEN_SYSTEM),
        HumanMessage(
            content=f"Build the application from this specification:\n\n{spec_text}"
        ),
    ]

    # Stream live — the user sees the AI "typing" the files in real time
    full_response = stream_response(codegen_messages, label="💻 Generating Code", precise=True)

    console.print(Rule("[dim]Writing files to disk…[/dim]"))

    files = parse_file_blocks(full_response)

    if not files:
        # Fallback if AI ignored the structured format
        raw_path = app_output_dir / "ai_output.txt"
        raw_path.write_text(full_response, encoding="utf-8")
        console.print(f"[yellow]⚠ Couldn't parse file blocks. Raw output saved → {raw_path}[/yellow]")
        return app_output_dir

    written = write_files(files, app_output_dir)
    summary = extract_summary_block(full_response)

    console.print()
    console.print(
        Panel(
            f"[bold green]✨ App generated![/bold green]\n\n"
            f"[dim]Output:[/dim] [bold]{app_output_dir}[/bold]\n"
            f"[dim]Files:[/dim] {len(written)}\n\n"
            + (f"[bold]How to run:[/bold]\n{summary}" if summary else ""),
            title="[bold magenta]Build Complete[/bold magenta]",
            border_style="magenta",
        )
    )

    return app_output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Iteration Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_iteration_loop(spec_text: str, app_name: str, app_output_dir: Path):
    """
    After initial code generation, wait for further instructions from the user.

    The user can ask to:
    - Add new features
    - Fix bugs
    - Refactor or improve existing code
    - Ask questions about the codebase

    The AI streams its response live. If it outputs file blocks, those files
    are automatically updated on disk.

    Args:
        spec_text:      The spec document (for context)
        app_name:       The app name
        app_output_dir: Where the app files live
    """
    console.print(Rule("[bold magenta]Phase 4: Iteration[/bold magenta]"))
    print_iteration_commands()

    # Build context for the iteration AI: it needs to know the spec + all current files
    def build_iteration_context() -> str:
        """Collect all current file contents and return as a formatted string."""
        file_contents = []
        for path in sorted(app_output_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(app_output_dir)
                try:
                    content = path.read_text(encoding="utf-8")
                    file_contents.append(f"===FILE: {rel}===\n{content}\n===END===")
                except Exception:
                    file_contents.append(f"===FILE: {rel}===\n[binary or unreadable]\n===END===")
        return "\n\n".join(file_contents)

    # Iteration conversation — starts with spec + current code as context
    iteration_history = [
        SystemMessage(content=ITERATION_SYSTEM),
        HumanMessage(
            content=(
                f"Here is the app specification:\n\n{spec_text}\n\n"
                f"Here are the current source files:\n\n{build_iteration_context()}\n\n"
                "I'm ready for your instructions."
            )
        ),
        AIMessage(content="I have reviewed the spec and all current source files. What would you like me to change or improve?"),
    ]

    while True:
        console.print()
        user_input = Prompt.ask("[bold yellow]Your instruction[/bold yellow]").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "done", "stop"):
            console.print(
                Panel(
                    f"[bold green]Session complete![/bold green]\n\n"
                    f"Your app is at: [bold]{app_output_dir}[/bold]",
                    border_style="green",
                )
            )
            break

        if user_input.lower() == "show spec":
            console.print(Markdown(spec_text))
            continue

        if user_input.lower() == "show files":
            files = list(app_output_dir.rglob("*"))
            file_list = "\n".join(f"  • {f.relative_to(app_output_dir)}" for f in files if f.is_file())
            console.print(Panel(file_list or "[dim]No files yet[/dim]", title="Generated Files", border_style="dim"))
            continue

        # Add the user's instruction to iteration history and get AI response
        iteration_history.append(HumanMessage(content=user_input))
        console.print("\n[dim]Streaming AI response…[/dim]")
        response = stream_response(iteration_history, label="💻 AI", precise=True)
        iteration_history.append(AIMessage(content=response))

        # Check if AI made file changes and apply them
        files = parse_file_blocks(response)
        if files:
            console.print(Rule("[dim]Applying file changes…[/dim]"))
            write_files(files, app_output_dir)
            summary = extract_summary_block(response)
            if summary:
                console.print(Panel(summary, title="[bold]Changes Applied[/bold]", border_style="green"))
        else:
            # No file blocks = AI answered a question or gave advice without code changes
            console.print("[dim](No file changes in this response)[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Coordinates all phases in sequence:
      0. Resume saved session or start new
      1. Discussion loop (unless resuming past spec)
      2. Spec generation (unless already have spec)
      3. Code generation
      4. Iteration loop (until user exits)
    """
    parser = argparse.ArgumentParser(description="Idea-to-App Builder")
    parser.add_argument("--new", action="store_true", help="Force a new session (ignore saved sessions)")
    args = parser.parse_args()

    print_banner()
    console.print()

    # ── Phase 0: Resume or new session ──────────────────────────────────────
    existing_history, app_name, spec_text = pick_session(force_new=args.new)

    # ── Get initial idea (always, even for resumes — used as app_name source) ─
    if not existing_history:
        console.print("[bold]Describe your app idea[/bold] [dim](as detailed or brief as you like)[/dim]")
        idea = Prompt.ask("[bold yellow]Your idea[/bold yellow]").strip()
        if not idea:
            console.print("[red]No idea provided. Exiting.[/red]")
            return
    else:
        # For resumed sessions we already have app_name; idea is just informational
        idea = app_name.replace("_", " ")

    # ── Phase 1: Discussion ──────────────────────────────────────────────────
    # Skip if we already have a spec (session was resumed past the discussion)
    if not spec_text:
        history, app_name = run_discussion(idea, existing_history=existing_history)

        # ── Phase 2: Spec Generation ─────────────────────────────────────────
        spec_text, spec_path = generate_spec(history, app_name)
    else:
        console.print(
            Panel(
                f"[green]✔ Using existing spec from saved session[/green]\n"
                f"App: [bold]{app_name}[/bold]",
                border_style="green",
            )
        )

    # Confirm before starting code generation
    console.print()
    proceed = Prompt.ask(
        "[bold]Generate code now?[/bold] [dim](yes/no)[/dim]",
        choices=["yes", "no", "y", "n"],
        default="yes",
    )

    if proceed.lower() in ("no", "n"):
        console.print(f"[dim]Session saved. Run main.py again anytime to resume.[/dim]")
        return

    # ── Phase 3: Code Generation ─────────────────────────────────────────────
    app_output_dir = generate_code(spec_text, app_name)

    # ── Phase 4: Iteration Loop ───────────────────────────────────────────────
    # AI waits here for more instructions — add features, fix bugs, etc.
    console.print()
    run_iteration_loop(spec_text, app_name, app_output_dir)


if __name__ == "__main__":
    main()
