"""
evals/run_eval.py — Lever-ablation eval for Developer Mode
==========================================================
Answers the question the README asserts but the repo could not previously prove:
*do the quality levers actually lift a small model's output, and which ones earn
their latency?*

It runs ONE design phase (default: security) under several lever configurations,
grades each resulting decision with a judge model, and prints a comparison table.
Single-phase keeps the wall-clock bounded and isolates the levers' effect.

    python -m evals.run_eval                          # baseline vs full, whatsapp/security
    python -m evals.run_eval --configs baseline,reflect,best_of,full
    python -m evals.run_eval --fixture invoicing --phase data_model --repeat 2
    python -m evals.run_eval --judge-model qwen2.5-coder:14b   # stronger grader

This needs the configured model server running and the model downloaded. Each config is a full phase run
(best_of=3 + reflect ≈ several model calls), so a 4-config sweep is many minutes
on a 7B — that cost is the point: it tells you what to keep and what to cut.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from devmode.phases import PHASES_BY_ID
from evals.fixtures import DEFAULT_FIXTURE, DEFAULT_PHASE, get_idea
from evals.rubric import coverage_fraction, score_decision

console = Console()

# Every lever this eval toggles, defaulting OFF. A named config layers on top, so
# each run is fully specified and deterministic regardless of the user's config.
_ALL_OFF = {
    "reflect": False,
    "best_of": False,
    "consistency_check": False,
    "build_review": False,
}

# Named configurations: a baseline with everything off, single-lever isolations,
# and the full stack. Add your own here.
LEVER_CONFIGS: dict[str, dict] = {
    "baseline": {},
    "reflect": {"reflect": True},
    "best_of": {"best_of": True},
    "consistency": {"consistency_check": True},
    "full": {"reflect": True, "best_of": True, "consistency_check": True},
}


def resolve_levers(name: str) -> dict:
    """Resolve a named config into the full set of lever booleans (all-off base)."""
    if name not in LEVER_CONFIGS:
        raise SystemExit(f"Unknown config '{name}'. Known: {', '.join(LEVER_CONFIGS)}")
    return {**_ALL_OFF, **LEVER_CONFIGS[name]}


def run_phase_under_config(idea: str, phase_id: str, levers: dict, judge_model: str = "") -> str:
    """Run one design phase in auto mode under the given levers; return its decision text.

    Mutates the in-memory devmode config for this run (the harness owns the process).
    Uses a throwaway workspace so artifacts never collide between configs.
    """
    from core.config import get_config
    from devmode.session import DevSession, _decision_section

    cfg = get_config().raw()
    cfg.setdefault("devmode", {})
    cfg["devmode"].update(levers)
    # The eval deliberately MEASURES every named lever, so it must defeat the
    # production best_of→judge_model gate: when a config turns best_of on without
    # an explicit --judge-model, fall back to the main model as judge (a
    # self-judge) so best-of-N actually runs and can be scored.
    cfg["devmode"]["judge_model"] = judge_model or (
        cfg["model"]["name"] if levers.get("best_of") else "")

    with tempfile.TemporaryDirectory(prefix="aicoder_eval_") as tmp:
        ws = Path(tmp)
        ds = DevSession(ws, idea=idea, auto=True)
        ds.run(only=phase_id)
        spec = PHASES_BY_ID[phase_id]
        path = ds._artifact_path(spec)
        if not path.exists():
            return ""
        return _decision_section(path.read_text(encoding="utf-8", errors="replace"))


def evaluate(idea: str, phase_id: str, config_names: list[str], judge_model: str,
             repeat: int) -> list[dict]:
    """Run every config `repeat` times, grade each decision, and aggregate."""
    spec = PHASES_BY_ID[phase_id]
    rows: list[dict] = []
    for name in config_names:
        levers = resolve_levers(name)
        overalls: list[float] = []
        coverages: list[float] = []
        rationale = ""
        t0 = time.monotonic()
        for r in range(repeat):
            console.print(f"[dim]▶ config [bold]{name}[/bold] — run {r + 1}/{repeat}…[/dim]")
            decision = run_phase_under_config(idea, phase_id, levers, judge_model)
            if not decision.strip():
                console.print(f"[yellow]  config '{name}' produced no decision — skipping run.[/yellow]")
                continue
            score = score_decision(idea, spec, decision, judge_model=judge_model)
            if score is None:
                console.print("[yellow]  judge returned no parseable score — skipping run.[/yellow]")
                continue
            overalls.append(score["overall"])
            cf = coverage_fraction(score)
            if cf is not None:
                coverages.append(cf)
            rationale = score["rationale"] or rationale
            console.print(f"[dim]  → overall {score['overall']:.1f}/10"
                          + (f", coverage {cf * 100:.0f}%" if cf is not None else "") + "[/dim]")
        rows.append({
            "config": name,
            "levers": [k for k, v in levers.items() if v] or ["none"],
            "runs": len(overalls),
            "mean_overall": (sum(overalls) / len(overalls)) if overalls else None,
            "mean_coverage": (sum(coverages) / len(coverages)) if coverages else None,
            "seconds": time.monotonic() - t0,
            "rationale": rationale,
        })
    return rows


def render(rows: list[dict], idea: str, phase_id: str) -> None:
    table = Table(title=f"Lever ablation — phase '{phase_id}'", title_style="bold magenta")
    table.add_column("Config", style="cyan")
    table.add_column("Levers on", style="dim")
    table.add_column("Runs", justify="right")
    table.add_column("Overall /10", justify="right", style="bold")
    table.add_column("Coverage", justify="right")
    table.add_column("Wall-clock", justify="right", style="dim")

    base = next((r["mean_overall"] for r in rows if r["config"] == "baseline"), None)
    for r in rows:
        overall = r["mean_overall"]
        cell = "—" if overall is None else f"{overall:.2f}"
        if overall is not None and base is not None and r["config"] != "baseline":
            delta = overall - base
            cell += f" [{'green' if delta >= 0 else 'red'}]({delta:+.2f})[/]"
        cov = r["mean_coverage"]
        table.add_row(
            r["config"], ", ".join(r["levers"]), str(r["runs"]), cell,
            "—" if cov is None else f"{cov * 100:.0f}%", f"{r['seconds']:.0f}s",
        )
    console.print()
    console.print(table)
    console.print(Panel(
        "[bold]How to read this:[/bold] the [cyan]baseline[/cyan] row is all levers OFF. "
        "A lever earns its place only if its [bold]Overall[/bold] gain is worth its added "
        "[dim]wall-clock[/dim]. A lever near +0.0 is latency you can cut.",
        border_style="dim"))


def main() -> None:
    ap = argparse.ArgumentParser(prog="evals.run_eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE, help="named product idea (see evals/fixtures.py)")
    ap.add_argument("--idea", default=None, help="an explicit product idea (overrides --fixture)")
    ap.add_argument("--phase", default=DEFAULT_PHASE, help=f"phase id to grade (default: {DEFAULT_PHASE})")
    ap.add_argument("--configs", default="baseline,full",
                    help="comma-separated config names: " + ", ".join(LEVER_CONFIGS))
    ap.add_argument("--judge-model", default="", help="stronger model for grading + critic steps")
    ap.add_argument("--repeat", type=int, default=1, help="runs per config (averaged; reduces noise)")
    ap.add_argument("--out", default=None, help="write the raw results JSON to this path")
    args = ap.parse_args()

    if args.phase not in PHASES_BY_ID:
        raise SystemExit(f"Unknown phase '{args.phase}'. Known: {', '.join(PHASES_BY_ID)}")
    idea = get_idea(args.fixture, args.idea)
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]

    console.print(Panel(
        f"[bold]Developer Mode lever ablation[/bold]\n"
        f"Idea: [dim]{idea[:90]}…[/dim]\nPhase: [cyan]{args.phase}[/cyan]   "
        f"Configs: {', '.join(config_names)}   Repeat: {args.repeat}\n"
        f"Judge: [magenta]{args.judge_model or '(main model)'}[/magenta]",
        title="[magenta]🧪 eval[/magenta]", border_style="magenta"))

    rows = evaluate(idea, args.phase, config_names, args.judge_model, args.repeat)
    render(rows, idea, args.phase)

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"idea": idea, "phase": args.phase, "results": rows}, indent=2), encoding="utf-8")
        console.print(f"[dim]Raw results → {args.out}[/dim]")


if __name__ == "__main__":
    main()
