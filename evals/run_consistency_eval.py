"""
evals/run_consistency_eval.py — Precision/recall for the consistency_check lever
================================================================================
Measures the lever the single-phase ablation could not: does `consistency_check`
catch a later phase contradicting an earlier one — without crying wolf on clean
designs?

For each labeled fixture it calls ``DevSession._consistency_findings`` (the live
build's code path) and records whether a contradiction was flagged, then reports
a confusion matrix with precision (don't false-alarm) and recall (don't miss).

    python -m evals.run_consistency_eval                       # main model as judge
    python -m evals.run_consistency_eval --judge-model qwen2.5-coder:14b
    python -m evals.run_consistency_eval --repeat 3            # majority-vote per case

Needs the configured model server running. A "contradiction" is the positive class: recall = fraction
of real contradictions caught; precision = fraction of flags that were real.
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
from evals.consistency_fixtures import CASES

console = Console()


def detect_case(case: dict, judge_model: str = "", detector=None) -> bool:
    """Return True if the checker flags a contradiction for this case.

    ``detector(spec, prior, new) -> str`` is injectable for tests; by default it
    runs the real ``DevSession._consistency_findings`` (non-empty == flagged).
    """
    spec = PHASES_BY_ID[case["new_phase"]]
    if detector is None:
        from core.config import get_config
        from devmode.session import DevSession

        cfg = get_config().raw()
        cfg.setdefault("devmode", {})["judge_model"] = judge_model
        with tempfile.TemporaryDirectory(prefix="aicoder_consist_") as tmp:
            ds = DevSession(Path(tmp), "consistency eval")
            findings = ds._consistency_findings(spec, case["prior"], case["new"])
    else:
        findings = detector(spec, case["prior"], case["new"])
    return bool((findings or "").strip())


def compute_metrics(results: list[dict]) -> dict:
    """Confusion matrix + precision/recall/accuracy. Positive class = contradiction."""
    tp = fp = tn = fn = 0
    for r in results:
        positive = r["label"] == "contradiction"
        if r["flagged"] and positive:
            tp += 1
        elif r["flagged"] and not positive:
            fp += 1
        elif not r["flagged"] and positive:
            fn += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "accuracy": (tp + tn) / total if total else None,
    }


def evaluate(cases: list[dict], judge_model: str, repeat: int, detector=None) -> list[dict]:
    """Run every case `repeat` times; a case counts as flagged on a majority of runs."""
    results = []
    for case in cases:
        flags = 0
        for r in range(repeat):
            console.print(f"[dim]▶ {case['id']} ({case['label']}/{case['difficulty']}) "
                          f"— run {r + 1}/{repeat}…[/dim]")
            if detect_case(case, judge_model, detector):
                flags += 1
        flagged = flags > repeat / 2
        correct = flagged == (case["label"] == "contradiction")
        console.print(f"[dim]  → flagged {flags}/{repeat} → "
                      + ("[green]✓ correct[/green]" if correct else "[red]✗ wrong[/red]") + "[/dim]")
        results.append({**{k: case[k] for k in ("id", "label", "difficulty")},
                        "flagged": flagged, "flag_count": flags, "correct": correct})
    return results


def render(results: list[dict], metrics: dict, seconds: float) -> None:
    table = Table(title="consistency_check — contradiction detection", title_style="bold magenta")
    table.add_column("Case", style="cyan")
    table.add_column("Expected")
    table.add_column("Difficulty", style="dim")
    table.add_column("Flagged?", justify="center")
    table.add_column("Verdict", justify="center")
    for r in results:
        verdict = "[green]✓[/green]" if r["correct"] else "[red]✗[/red]"
        exp = "contradiction" if r["label"] == "contradiction" else "clean"
        table.add_row(r["id"], exp, r["difficulty"],
                      "yes" if r["flagged"] else "no", verdict)
    console.print()
    console.print(table)

    def pct(x):
        return "—" if x is None else f"{x * 100:.0f}%"

    console.print(Panel(
        f"[bold]Recall[/bold] (real contradictions caught): [bold]{pct(metrics['recall'])}[/bold]   "
        f"[bold]Precision[/bold] (flags that were real): [bold]{pct(metrics['precision'])}[/bold]   "
        f"[bold]Accuracy[/bold]: {pct(metrics['accuracy'])}\n"
        f"[dim]TP {metrics['tp']}  FP {metrics['fp']}  TN {metrics['tn']}  FN {metrics['fn']}"
        f"   ·   {seconds:.0f}s[/dim]\n\n"
        "[dim]Read it: high recall on [bold]blatant[/bold] cases but misses on [bold]subtle[/bold] "
        "ones is the expected 7B ceiling — it justifies keeping the lever (cheap insurance against "
        "blatant contradictions) while documenting that subtle ones still need 'dev revisit'.[/dim]",
        border_style="dim"))


def main() -> None:
    ap = argparse.ArgumentParser(prog="evals.run_consistency_eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge-model", default="", help="stronger model for the consistency critic")
    ap.add_argument("--repeat", type=int, default=1, help="runs per case (majority vote)")
    ap.add_argument("--out", default=None, help="write raw results JSON to this path")
    args = ap.parse_args()

    console.print(Panel(
        f"[bold]consistency_check detection eval[/bold]\n"
        f"{len(CASES)} cases   Repeat: {args.repeat}   "
        f"Judge: [magenta]{args.judge_model or '(main model)'}[/magenta]",
        title="[magenta]🧪 eval[/magenta]", border_style="magenta"))

    t0 = time.monotonic()
    results = evaluate(CASES, args.judge_model, args.repeat)
    metrics = compute_metrics(results)
    render(results, metrics, time.monotonic() - t0)

    if args.out:
        Path(args.out).write_text(json.dumps({"results": results, "metrics": metrics}, indent=2),
                                  encoding="utf-8")
        console.print(f"[dim]Raw results → {args.out}[/dim]")


if __name__ == "__main__":
    main()
