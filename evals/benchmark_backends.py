"""
evals/benchmark_backends.py — Ollama vs LM Studio raw inference benchmark
============================================================================
Not a Developer Mode quality eval like the rest of this package — this
measures raw backend throughput (time-to-first-token, decode tokens/sec) so
you can decide which local inference server actually performs better on your
machine, instead of trusting a general claim about Ollama vs LM Studio.

Needs the server(s) already running with a model loaded:
  - Ollama:    `ollama serve` (default http://localhost:11434), model pulled.
  - LM Studio: local server started from the app's "Developer" tab (default
               http://localhost:1234/v1), a model loaded.

Usage:
    # both backends, using whatever model ai-coder is configured with for
    # Ollama, and whatever's currently loaded in LM Studio
    python -m evals.benchmark_backends

    python -m evals.benchmark_backends --ollama-model qwen2.5-coder:7b \\
        --lmstudio-model qwen2.5-coder-7b-instruct

    python -m evals.benchmark_backends --only ollama --repeat 3
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import httpx

OLLAMA_URL = "http://localhost:11434"
LMSTUDIO_URL = "http://localhost:1234/v1"

# A short factual answer, a bounded code-gen task, and a longer explanation —
# spans the shapes of prompt a coding assistant actually sends, so the
# average isn't dominated by one response length.
PROMPTS = [
    "What does the Python zip() builtin do? One sentence.",
    "Write a Python function that reverses a singly linked list. "
    "Just the code, no explanation.",
    "Explain the tradeoffs between depth-first and breadth-first search "
    "for finding the shortest path in an unweighted graph.",
]

DEFAULT_MAX_TOKENS = 300


@dataclass
class BenchResult:
    backend: str
    prompt: str
    ttft_s: float
    total_s: float
    output_tokens: int
    tokens_per_sec_wall: float
    tokens_per_sec_native: float | None = None  # server-reported, decode-only (Ollama only)


# ── Ollama ───────────────────────────────────────────────────────────────────

def bench_ollama(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float = 120) -> BenchResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"num_predict": max_tokens},
    }
    start = time.monotonic()
    timed_lines = []
    with httpx.stream("POST", f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                timed_lines.append((line, time.monotonic()))
    return parse_ollama_stream(timed_lines, start, prompt)


def parse_ollama_stream(timed_lines: list[tuple[str, float]], start_time: float, prompt: str) -> BenchResult:
    """Pure — takes already-collected (line, arrival-time) pairs, so this is
    unit-tested against canned fixture data without a real server."""
    first_token_time = None
    for line, ts in timed_lines:
        chunk = json.loads(line)
        if first_token_time is None and chunk.get("message", {}).get("content"):
            first_token_time = ts
        if chunk.get("done"):
            total_s = ts - start_time
            ttft_s = (first_token_time - start_time) if first_token_time else total_s
            output_tokens = chunk.get("eval_count", 0)
            eval_duration_s = chunk.get("eval_duration", 0) / 1e9
            decode_wall_clock = max(total_s - ttft_s, 1e-6)
            return BenchResult(
                backend="ollama",
                prompt=prompt,
                ttft_s=ttft_s,
                total_s=total_s,
                output_tokens=output_tokens,
                tokens_per_sec_wall=output_tokens / decode_wall_clock,
                tokens_per_sec_native=(output_tokens / eval_duration_s) if eval_duration_s else None,
            )
    raise RuntimeError("Ollama stream ended without a final 'done' chunk")


# ── LM Studio (OpenAI-compatible) ───────────────────────────────────────────

def bench_lmstudio(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float = 120) -> BenchResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "stream_options": {"include_usage": True},
    }
    start = time.monotonic()
    timed_lines = []
    with httpx.stream("POST", f"{base_url.rstrip('/')}/chat/completions", json=payload, timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                timed_lines.append((line, time.monotonic()))
    return parse_lmstudio_stream(timed_lines, start, prompt)


def parse_lmstudio_stream(timed_lines: list[tuple[str, float]], start_time: float, prompt: str) -> BenchResult:
    """Pure — same shape as parse_ollama_stream, unit-tested the same way."""
    first_token_time = None
    output_tokens = None
    last_ts = start_time
    for line, ts in timed_lines:
        last_ts = ts
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            continue
        chunk = json.loads(data)
        choices = chunk.get("choices") or []
        if first_token_time is None and choices and choices[0].get("delta", {}).get("content"):
            first_token_time = ts
        usage = chunk.get("usage")
        if usage:
            output_tokens = usage.get("completion_tokens")

    if output_tokens is None:
        raise RuntimeError(
            "LM Studio response never included a usage field — enable "
            "'stream_options.include_usage' support (recent LM Studio versions "
            "have it) or lower --max-tokens and try again."
        )
    total_s = last_ts - start_time
    ttft_s = (first_token_time - start_time) if first_token_time else total_s
    decode_wall_clock = max(total_s - ttft_s, 1e-6)
    return BenchResult(
        backend="lmstudio",
        prompt=prompt,
        ttft_s=ttft_s,
        total_s=total_s,
        output_tokens=output_tokens,
        tokens_per_sec_wall=output_tokens / decode_wall_clock,
    )


# ── Runner ───────────────────────────────────────────────────────────────────

def _default_ollama_model() -> str | None:
    try:
        from core.config import get_config

        cfg = get_config()
        return cfg.model_name if cfg.model_provider == "ollama" else None
    except Exception:  # noqa: BLE001 — benchmark still runs with an explicit --ollama-model
        return None


def _default_lmstudio_model(base_url: str) -> str | None:
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        return models[0]["id"] if models else None
    except Exception:  # noqa: BLE001 — surfaced as a clear skip message by the caller
        return None


def run_backend(name: str, bench_fn, base_url: str, model: str, repeat: int, max_tokens: int) -> list[BenchResult]:
    print(f"\n=== {name} — model: {model} ===")
    results = []
    try:
        # Warm-up: excluded from results — the first request after a model
        # loads/becomes idle pays a one-off cold-start cost that would
        # otherwise skew the very first prompt's numbers.
        print("  (warm-up run, discarded)")
        bench_fn(base_url, model, PROMPTS[0], max_tokens)
    except Exception as e:  # noqa: BLE001 — skip this backend, keep going
        print(f"  ✗ couldn't reach {name} at {base_url}: {e}")
        return results

    for prompt in PROMPTS:
        for i in range(repeat):
            try:
                r = bench_fn(base_url, model, prompt, max_tokens)
            except Exception as e:  # noqa: BLE001 — one bad prompt shouldn't kill the run
                print(f"  ✗ {prompt[:40]!r} failed: {e}")
                continue
            results.append(r)
            rep = f" (rep {i + 1}/{repeat})" if repeat > 1 else ""
            print(f"  {prompt[:40]!r}{rep}: ttft={r.ttft_s:.2f}s  "
                  f"{r.output_tokens} tok in {r.total_s:.2f}s  "
                  f"({r.tokens_per_sec_wall:.1f} tok/s)")
    return results


def print_summary(all_results: dict[str, list[BenchResult]]) -> None:
    print("\n=== Summary (decode tokens/sec, wall-clock, higher is better) ===")
    for name, results in all_results.items():
        if not results:
            print(f"  {name}: no successful runs")
            continue
        avg_wall = sum(r.tokens_per_sec_wall for r in results) / len(results)
        avg_ttft = sum(r.ttft_s for r in results) / len(results)
        native = [r.tokens_per_sec_native for r in results if r.tokens_per_sec_native]
        native_note = f"  (server-reported: {sum(native) / len(native):.1f} tok/s)" if native else ""
        print(f"  {name:10s}  {avg_wall:6.1f} tok/s{native_note}   avg ttft {avg_ttft:.2f}s"
              f"   over {len(results)} run(s)")

    winner = max(
        ((name, sum(r.tokens_per_sec_wall for r in results) / len(results))
         for name, results in all_results.items() if results),
        key=lambda x: x[1],
        default=None,
    )
    if winner:
        print(f"\nFaster on this machine, for these models: {winner[0]} ({winner[1]:.1f} tok/s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["ollama", "lmstudio", "both"], default="both")
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--ollama-model", default=None, help="defaults to ai-coder's configured Ollama model")
    parser.add_argument("--lmstudio-url", default=LMSTUDIO_URL)
    parser.add_argument("--lmstudio-model", default=None, help="defaults to whatever's currently loaded")
    parser.add_argument("--repeat", type=int, default=1, help="repetitions per prompt, averaged in the summary")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    all_results: dict[str, list[BenchResult]] = {}

    if args.only in ("ollama", "both"):
        model = args.ollama_model or _default_ollama_model()
        if not model:
            print("=== ollama ===\n  ✗ no --ollama-model given and none configured (see `aicoder`'s "
                  "config.yaml) — skipping")
        else:
            all_results["ollama"] = run_backend(
                "ollama", bench_ollama, args.ollama_url, model, args.repeat, args.max_tokens
            )

    if args.only in ("lmstudio", "both"):
        model = args.lmstudio_model or _default_lmstudio_model(args.lmstudio_url)
        if not model:
            print("=== lmstudio ===\n  ✗ no --lmstudio-model given and couldn't list loaded models "
                  "(is LM Studio's local server running, with a model loaded?) — skipping")
        else:
            all_results["lmstudio"] = run_backend(
                "lmstudio", bench_lmstudio, args.lmstudio_url, model, args.repeat, args.max_tokens
            )

    if all_results:
        print_summary(all_results)
    else:
        print("\nNo backend produced results — nothing to summarize.")


if __name__ == "__main__":
    main()
