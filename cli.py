"""
cli.py — Entry point for the aicoder CLI
=========================================
Run directly:   python cli.py [options]
After install:  aicoder [options]

Usage:
    aicoder                     Start in current directory
    aicoder --workspace ./myapp Start in a specific directory
    aicoder --model qwen2.5:7b  Override model for this session
    aicoder --version           Show version
"""

import argparse
import sys
from pathlib import Path


def _version() -> str:
    """The installed package version, read from metadata (falls back gracefully).

    Avoids a hardcoded string that silently drifts from pyproject.toml.
    """
    try:
        from importlib.metadata import version

        return version("ai-coder")
    except Exception:  # PackageNotFoundError, or running from a bare checkout
        return "0+unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aicoder",
        description="AI-powered coding assistant — local, offline, powerful.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aicoder                           # Start in current directory
  aicoder --workspace ./my-project  # Point at a specific project
  aicoder --model qwen2.5-coder:7b  # Use a different Ollama model
  aicoder --shell-mode never        # Auto-approve shell commands this session

In the agent (default):
  Just describe a task in plain English — it reads, edits, runs, and verifies.
  exit / quit     Leave the session

Flags:
  --selftest      Check the model supports tool calling, then exit
        """,
    )

    parser.add_argument(
        "--workspace", "-w",
        type=Path,
        default=None,
        metavar="PATH",
        help="Project directory to work in (default: current directory)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        metavar="MODEL",
        help="Ollama model name to use (overrides config.yaml for this session)",
    )
    parser.add_argument(
        "--shell-mode",
        choices=["always", "never", "smart"],
        default=None,
        dest="shell_mode",
        help="Set shell confirmation mode for this session",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"aicoder {_version()}",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show config file location and current settings, then exit",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Verify the configured model supports tool calling, then exit",
    )

    args = parser.parse_args()

    # ── Resolve workspace ──────────────────────────────────────────────────────
    workspace = (args.workspace or Path.cwd()).resolve()
    if not workspace.exists():
        print(f"[ERROR] Workspace does not exist: {workspace}", file=sys.stderr)
        sys.exit(1)
    if not workspace.is_dir():
        print(f"[ERROR] Not a directory: {workspace}", file=sys.stderr)
        sys.exit(1)

    # ── Load config (and apply overrides) ─────────────────────────────────────
    from core.config import get_config

    cfg = get_config()

    if args.model:
        # Override model name in-memory for this session only
        cfg.raw()["model"]["name"] = args.model

    if args.shell_mode:
        cfg.raw()["shell"]["confirmation"] = args.shell_mode

    # ── --config flag: show settings and exit ─────────────────────────────────
    if args.config:
        from core.config import CONFIG_PATH
        import yaml
        print(f"Config file: {CONFIG_PATH}\n")
        print(yaml.dump(cfg.raw(), default_flow_style=False, sort_keys=False))
        sys.exit(0)

    # ── Verify Ollama is available ─────────────────────────────────────────────
    _check_ollama(cfg.model_base_url, cfg.model_name)

    # ── Self-test: confirm native tool calling, then exit ──────────────────────
    if args.selftest:
        from core.model import selftest
        sys.exit(0 if selftest() else 1)

    # ── Launch the agent ───────────────────────────────────────────────────────
    from agent.loop import run_agent_repl
    run_agent_repl(workspace=workspace)


def _check_ollama(base_url: str, model_name: str) -> None:
    """Warn if Ollama doesn't seem to be running or the model isn't pulled."""
    from core.model import is_model_pulled

    pulled = is_model_pulled(base_url, model_name)
    from rich.console import Console

    if pulled is None:
        Console().print(
            "\n[yellow]⚠ Cannot reach Ollama server.[/yellow]\n"
            f"[dim]Make sure Ollama is running: ollama serve[/dim]\n"
            f"[dim]Expected at: {base_url}[/dim]\n"
        )
    elif not pulled:
        Console().print(
            f"\n[yellow]⚠ Model '[bold]{model_name}[/bold]' may not be pulled yet.[/yellow]\n"
            f"[dim]Run: ollama pull {model_name}[/dim]\n"
        )


if __name__ == "__main__":
    main()
