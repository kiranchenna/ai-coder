"""
cli.py — Entry point for the aicoder CLI
=========================================
Run directly:   python cli.py [options]
After install:  aicoder [options]

Usage:
    aicoder                          Start in current directory
    aicoder --workspace ./myapp      Start in a specific directory
    aicoder --model qwen2.5-coder-7b-instruct  Override model for this session
    aicoder --version                Show version
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
  aicoder                                     # Start in current directory
  aicoder --workspace ./my-project            # Point at a specific project
  aicoder --model qwen2.5-coder-7b-instruct   # Use a different LM Studio model
  aicoder --shell-mode never                  # Auto-approve shell commands this session

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
        help="Model id to use (overrides config.yaml for this session)",
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
    parser.add_argument(
        "--continue", "-c",
        action="store_true",
        dest="continue_session",
        help="Resume the most recent conversation for this workspace instead of starting fresh",
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

    _run_preflight(cfg)

    # ── Self-test: confirm native tool calling, then exit ──────────────────────
    if args.selftest:
        from core.model import selftest
        sys.exit(0 if selftest() else 1)

    # ── Launch the agent ───────────────────────────────────────────────────────
    # A real terminal gets the full-screen chat UI (agent/tui.py); piped/
    # redirected/scripted output (including the whole test suite) falls back
    # to the plain print-and-scroll REPL, which has always supported that.
    if sys.stdout.isatty() and sys.stdin.isatty():
        from agent.tui import run as run_tui
        run_tui(workspace, continue_session=args.continue_session)
    else:
        from agent.loop import run_agent_repl
        run_agent_repl(workspace=workspace, continue_session=args.continue_session)


def _run_preflight(cfg) -> None:
    """Warn before launching if the configured model server doesn't seem
    reachable — see `_check_openai_compatible`."""
    _check_openai_compatible(cfg.model_base_url, cfg.model_name)


def _check_openai_compatible(base_url: str, model_name: str) -> None:
    """Warn if an openai_compatible endpoint (LM Studio, vLLM, a hosted
    API, ...) doesn't seem reachable, or the configured model isn't among
    what it currently reports via /v1/models — LM Studio's own /v1/models
    lists every model downloaded to disk (confirmed live, not just loaded
    ones), so this doubles as a real "is it available" check there, not
    just a reachability ping."""
    from core.model import is_lmstudio_reachable

    from core.console import SafeConsole

    models = is_lmstudio_reachable(base_url)
    if models is None:
        SafeConsole().print(
            "\n[yellow]⚠ Cannot reach the configured model server.[/yellow]\n"
            f"[dim]Expected an OpenAI-compatible server at: {base_url}[/dim]\n"
            "[dim]If this is LM Studio: open it and start the local server "
            "(Developer tab → Start Server).[/dim]\n"
        )
    elif model_name not in models:
        SafeConsole().print(
            f"\n[yellow]⚠ '[bold]{model_name}[/bold]' isn't available on that server.[/yellow]\n"
            f"[dim]If this is LM Studio: `lms get {model_name}` (or pick one you already "
            "have with /model).[/dim]\n"
        )


if __name__ == "__main__":
    main()
