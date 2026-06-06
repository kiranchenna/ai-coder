"""
tools/shell_tools.py — Shell command execution with configurable confirmation
=============================================================================
Three confirmation modes (set in ~/.aicoder/config.yaml under shell.confirmation):
  always  — always prompt [y/N] before running (the safe default)
  never   — auto-run without asking
  smart   — auto-approve safe-looking commands, ask for destructive ones

NOTE: "smart" mode uses the best-effort heuristic below. It is a convenience,
NOT a security boundary — a determined or unlucky command can slip past pattern
matching. Use "always" if you need a hard gate before anything runs.
"""

import re
import subprocess
import sys
import threading
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

console = Console()

# ─── Patterns for "smart" mode ────────────────────────────────────────────────
# These match a *segment* of the command (split on shell operators), so a
# dangerous call hidden behind `&&`, `|`, `;` is still caught.

_DESTRUCTIVE_STARTS = (
    "rm ", "rm\t", "del ", "del\t",
    "rmdir", "rd ", "rd\t",
    "format", "mkfs", "fdisk", "dd ",
    "drop ", "drop\t",
    "truncate",
    "shutdown", "reboot", "halt", "poweroff",
    "git clean",
    "git reset --hard",
    "git push", "git checkout --", "git restore",
    "chmod ", "chown ",
    "kill ", "killall", "pkill",
    "sudo ",
)

# Substrings that signal danger anywhere in a command segment.
_DESTRUCTIVE_CONTAINS = (
    " -rf", " -fr", "-rf ", "-fr ",     # recursive-force in any spacing
    "--force", "--hard", "--no-preserve-root",
    " /f", " /q /f",
    "> /dev/", "xargs rm", "xargs -0 rm",
    "rm -r", "shutil.rmtree", "os.remove", "os.unlink",
    "mkfs", "find . -delete", "-delete",
)

# Shell operators that chain or redirect commands — we inspect each segment.
_SEGMENT_SPLIT = re.compile(r"&&|\|\||[;|]")


def _segment_is_destructive(segment: str) -> bool:
    seg = segment.strip().lower()
    if not seg:
        return False
    if any(seg.startswith(p) for p in _DESTRUCTIVE_STARTS):
        return True
    if any(p in seg for p in _DESTRUCTIVE_CONTAINS):
        return True
    # Output redirection that truncates a file: `> file` (but not `>>` append
    # nor fd-dup like `2>&1`).
    if re.search(r"(?<!>)>(?![>&])\s*[^\s&]", seg):
        return True
    return False


def _is_destructive(command: str) -> bool:
    """Best-effort heuristic: does any part of this command look dangerous?

    Splits on shell operators so a destructive call chained behind `&&`/`|`/`;`
    is still flagged. Not exhaustive — see the module docstring.
    """
    for segment in _SEGMENT_SPLIT.split(command):
        if _segment_is_destructive(segment):
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
    except Exception as e:
        console.print(f"[red]⚠ Failed to run command: {e}[/red]")
        return "", str(e), -1

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    # Drain stdout and stderr concurrently. Reading them sequentially can
    # deadlock: a child that fills the (~64KB) stderr pipe buffer blocks while we
    # are still draining stdout, so stdout never reaches EOF.
    def _drain_stdout() -> None:
        for line in process.stdout:  # type: ignore[union-attr]
            stdout_chunks.append(line)
            if stream_output:
                console.print(line, end="", markup=False)

    def _drain_stderr() -> None:
        for line in process.stderr:  # type: ignore[union-attr]
            stderr_chunks.append(line)

    t_out = threading.Thread(target=_drain_stdout, daemon=True)
    t_err = threading.Thread(target=_drain_stderr, daemon=True)
    t_out.start()
    t_err.start()

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        t_out.join()
        t_err.join()
        console.print(f"[yellow]⚠ Command timed out after {timeout}s[/yellow]")
        stderr_chunks.append(f"\nTimeoutExpired after {timeout}s")
        return "".join(stdout_chunks), "".join(stderr_chunks), -1

    t_out.join()
    t_err.join()
    return "".join(stdout_chunks), "".join(stderr_chunks), process.returncode


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
