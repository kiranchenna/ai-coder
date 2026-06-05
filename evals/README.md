# Developer Mode eval harness

The missing measurement layer. Developer Mode's quality levers (`reflect`,
`best_of`, `consistency_check`, …) each add latency on the bet that they lift a
small local model's output. This harness lets you **prove or disprove that bet**
— and find which levers are worth their wall-clock — instead of trusting an
assertion.

## What it does

It runs **one design phase** under several lever configurations, grades each
resulting decision with a judge model against the phase's own `must_cover`
checklist (plus an overall 0–10), and prints a comparison table with the score
delta vs. an all-levers-off baseline.

Single-phase by design: it bounds the wall-clock and isolates the levers' effect.

## Usage

Needs Ollama running and the model pulled (same setup as `aicoder`).

```bash
# baseline (all off) vs full stack, on the whatsapp/security fixture
python -m evals.run_eval

# isolate each lever to see its individual contribution
python -m evals.run_eval --configs baseline,reflect,best_of,full

# average 2 runs per config to cut noise; grade a different phase/fixture
python -m evals.run_eval --fixture invoicing --phase data_model --repeat 2

# grade (and run the critic steps) with a stronger model you've pulled
python -m evals.run_eval --judge-model qwen2.5-coder:14b --out results.json
```

Reading the table: a lever earns its place only if its **Overall** gain is worth
its added **wall-clock**. A lever sitting near +0.00 is latency you can cut.

> Cost note: `best_of=3` + `reflect` is several model calls per run, so a
> multi-config sweep is many minutes on a 7B. That cost is the point — it's how
> you decide what to keep.

## Layout

- `fixtures.py` — fixed product ideas (the yardstick; keep them stable).
- `rubric.py` — judge-model scoring; `parse_score` / `score_decision` are pure
  and unit-tested (`tests/test_evals.py`), with the model call injectable.
- `run_eval.py` — the ablation runner + CLI.

## Caveats

- The judge is itself a model; for trustworthy grades use the strongest
  `--judge-model` you can run, and `--repeat` to average out noise.
- `security` (the default phase) needs no web research, so the eval is offline
  and repeatable. Research-backed phases (`architecture`, `api`, …) will hit the
  network and vary with live search results.
