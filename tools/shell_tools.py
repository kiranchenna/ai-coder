"""
tools/shell_tools.py — Shell command execution with configurable confirmation
=============================================================================
Three confirmation modes (user-toggleable at runtime via /shell-mode):
  always  — always prompt [y/N] before running
  never   — auto-run without asking
  smart   — auto-approve safe commands, ask for destructive ones
"""

import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

console = Console()

# ─── Patterns for "smart" mode ────────────────────────────────────────────────

_DESTRUCTIVE_STARTS = (
    "rm ", "rm\t", "del ", "del\t",
    "rmdir", "rd ", "rd\t",
    "format", "mkfs", "fdisk",
    "drop ", "drop\t",
    "truncate",
    "shutdown", "reboot",
    "git clean",
    "git reset --hard",
)

_DESTRUCTIVE_CONTAINS = (
    " -rf ", " -fr ", "--force", " /f", " /q /f",
    "> /dev/", "| rm", "| del",
)


def _is_destructive(command: str) -> bool:
    """Heuristic check: does this command look dangerous?"""
    cmd_lower = command.lower().strip()
    if any(cmd_lower.startswith(p) for p in _DESTRUCTIVE_STARTS):
        return True
    if any(p in cmd_lower for p in _DESTRUCTIVE_CONTAINS):
        return True
    return False


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_command(
    command: str,
    cwd: Path | None = None,
    timeout: int = 120,
    stream_output: bool = True,
) -> tuple[str, str, int]:
    """
    Execute a shell command and return (stdout, stderr, returncode).

    Args:
        command:       Shell command string
        cwd:           Working directory (defaults to current dir)
        timeout:       Seconds before killing the process
        stream_output: Print output live to terminal as it's produced

    Returns:
        (stdout, stderr, returncode)
    """
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if stream_output:
            # Stream stdout live
            for line in process.stdout:  # type: ignore[union-attr]
                stdout_lines.append(line)
                console.print(line, end="", markup=False)

            # Collect stderr after
            stderr_text = process.stderr.read()  # type: ignore[union-attr]
            stderr_lines = [stderr_text] if stderr_text else []
            process.wait(timeout=timeout)
        else:
            stdout_text, stderr_text = process.communicate(timeout=timeout)
            stdout_lines = [stdout_text]
            stderr_lines = [stderr_text] if stderr_text else []

    except subprocess.TimeoutExpired:
        process.kill()
        console.print(f"[yellow]⚠ Command timed out after {timeout}s[/yellow]")
        stderr_lines.append(f"TimeoutExpired after {timeout}s")
        return "".join(stdout_lines), "".join(stderr_lines), -1
    except Exception as e:
        console.print(f"[red]⚠ Failed to run command: {e}[/red]")
        return "", str(e), -1

    returncode = process.returncode
    return "".join(stdout_lines), "".join(stderr_lines), returncode


def run_with_confirmation(
    command: str,
    cwd: Path | None = None,
    timeout: int = 120,
) -> tuple[str, str, int] | None:
    """
    Run a command, respecting the configured confirmation mode.

    Returns:
        (stdout, stderr, returncode) if the command ran
        None if the user declined to run it
    """
    from core.config import get_config
    mode = get_config().shell_confirmation

    needs_confirm = False
    if mode == "always":
        needs_confirm = True
    elif mode == "smart":
        needs_confirm = _is_destructive(command)

    if needs_confirm:
        console.print()
        console.print(
            Panel(
                f"[bold yellow]{command}[/bold yellow]",
                title="[bold]🔧 Shell Command[/bold]",
                subtitle=f"[dim]cwd: {cwd or Path.cwd()}[/dim]",
                border_style="yellow",
            )
        )
        if not Confirm.ask("[bold]Run this command?[/bold]", default=False):
            console.print("[dim]Command skipped.[/dim]")
            return None
    else:
        console.print(f"\n[bold dim]$ {command}[/bold dim]")

    return run_command(command, cwd=cwd, timeout=timeout, stream_output=True)


def show_shell_mode() -> None:
    """Print the current shell confirmation mode."""
    from core.config import get_config
    mode = get_config().shell_confirmation
    descriptions = {
        "always": "always ask [y/N] before running",
        "never":  "auto-run without asking",
        "smart":  "ask only for destructive commands",
    }
    console.print(
        f"  Shell mode: [bold cyan]{mode}[/bold cyan] — {descriptions.get(mode, '')}"
    )
