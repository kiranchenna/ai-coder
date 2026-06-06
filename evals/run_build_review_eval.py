"""
evals/run_build_review_eval.py — Does build_review remove planted placeholders?
===============================================================================
Measures the build-time `build_review` lever: it hands the real
``Builder._review_file`` drafts that contain a known placeholder/stub and checks
whether the review pass removed it — plus clean controls to confirm it doesn't
mangle good code.

    python -m evals.run_build_review_eval
    python -m evals.run_build_review_eval --repeat 3

Needs Ollama running. Reports a fix-rate over the planted-issue cases and a
preservation-rate over the clean ones.
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

from evals.build_review_fixtures import CASES

console = Console()


def review_case(case: dict, reviewer=None) -> str:
    """Return the reviewed file content. ``reviewer(entry, draft, system, prompt)``
    is injectable for tests; by default it runs the real Builder._review_file."""
    if reviewer is None:
        from devmode.build import Builder

        with tempfile.TemporaryDirectory(prefix="aicoder_review_") as tmp:
            b = Builder(Path(tmp))
            return b._review_file(case["entry"], case["draft"], case["system"], case["prompt"])
    return reviewer(case["entry"], case["draft"], case["system"], case["prompt"])


def judge_case(case: dict, reviewed: str) -> bool:
    """Whether the review did the right thing for this case.

    - expect "removed": the planted marker is gone AND the file didn't shrink to
      a stub (a real implementation replaced it, not a deletion).
    - expect "preserved": the key symbol still present and the file wasn't gutted.
    """
    marker = case["marker"]
    if case["expect"] == "removed":
        gone = marker.lower() not in reviewed.lower()
        substantial = len(reviewed.strip()) >= len(case["draft"].strip())
        return gone and substantial
    # preserved
    return marker in reviewed and len(reviewed.strip()) >= len(case["draft"].strip()) * 0.8


def evaluate(cases: list[dict], repeat: int, reviewer=None) -> list[dict]:
    results = []
    for case in cases:
        oks = 0
        for r in range(repeat):
            console.print(f"[dim]▶ {case['id']} ({case['expect']}) — run {r + 1}/{repeat}…[/dim]")
            reviewed = review_case(case, reviewer)
            if judge_case(case, reviewed):
                oks += 1
        ok = oks > repeat / 2
        console.print(f"[dim]  → {oks}/{repeat} good → "
                      + ("[green]✓[/green]" if ok else "[red]✗[/red]") + "[/dim]")
        results.append({"id": case["id"], "expect": case["expect"],
                        "ok": ok, "ok_count": oks})
    return results


def compute_metrics(results: list[dict]) -> dict:
    removed = [r for r in results if r["expect"] == "removed"]
    preserved = [r for r in results if r["expect"] == "preserved"]
    fixed = sum(1 for r in removed if r["ok"])
    kept = sum(1 for r in preserved if r["ok"])
    return {
        "fix_rate": fixed / len(removed) if removed else None,
        "preservation_rate": kept / len(preserved) if preserved else None,
        "fixed": fixed, "planted": len(removed),
        "kept": kept, "clean": len(preserved),
    }


def render(results: list[dict], metrics: dict, seconds: float) -> None:
    table = Table(title="build_review — placeholder removal", title_style="bold magenta")
    table.add_column("Case", style="cyan")
    table.add_column("Expectation")
    table.add_column("Result", justify="center")
    for r in results:
        exp = "remove placeholder" if r["expect"] == "removed" else "preserve clean code"
        table.add_row(r["id"], exp, "[green]✓[/green]" if r["ok"] else "[red]✗[/red]")
    console.print()
    console.print(table)

    def pct(x):
        return "—" if x is None else f"{x * 100:.0f}%"

    console.print(Panel(
        f"[bold]Fix rate[/bold] (planted placeholders removed): "
        f"[bold]{pct(metrics['fix_rate'])}[/bold] ({metrics['fixed']}/{metrics['planted']})   "
        f"[bold]Preservation[/bold] (clean code kept intact): "
        f"{pct(metrics['preservation_rate'])} ({metrics['kept']}/{metrics['clean']})\n"
        f"[dim]{seconds:.0f}s   ·   each ✓ removed a TODO/stub/NotImplementedError and replaced "
        "it with a real implementation.[/dim]\n\n"
        "[dim]A high fix rate with full preservation is the case for keeping build_review on in "
        "balanced: it strips the placeholders a 7B leaves behind without harming good drafts.[/dim]",
        border_style="dim"))


def main() -> None:
    ap = argparse.ArgumentParser(prog="evals.run_build_review_eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=1, help="runs per case (majority vote)")
    ap.add_argument("--out", default=None, help="write raw results JSON to this path")
    args = ap.parse_args()

    console.print(Panel(
        f"[bold]build_review placeholder-removal eval[/bold]\n"
        f"{len(CASES)} cases   Repeat: {args.repeat}",
        title="[magenta]🧪 eval[/magenta]", border_style="magenta"))

    t0 = time.monotonic()
    results = evaluate(CASES, args.repeat)
    metrics = compute_metrics(results)
    render(results, metrics, time.monotonic() - t0)

    if args.out:
        Path(args.out).write_text(json.dumps({"results": results, "metrics": metrics}, indent=2),
                                  encoding="utf-8")
        console.print(f"[dim]Raw results → {args.out}[/dim]")


if __name__ == "__main__":
    main()
