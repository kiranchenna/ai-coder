# Developer Mode eval harness

The missing measurement layer. Developer Mode's quality levers (`reflect`,
`best_of`, `consistency_check`, …) each add latency on the bet that they lift a
small local model's output. This harness lets you **prove or disprove that bet**
— and find which levers are worth their wall-clock — instead of trusting an
assertion.

## Two evals

### 1. Lever ablation — `run_eval` (quality vs. latency)

Runs **one design phase** under several lever configurations, grades each
resulting decision with a judge model against the phase's own `must_cover`
checklist (plus an overall 0–10), and prints a comparison table with the score
delta vs. an all-levers-off baseline. Single-phase by design: it bounds the
wall-clock and isolates each lever's effect.

Finding (security phase, n=2): **`reflect` carries essentially all the gain**
(7.5→9.5/10, 75%→100% coverage, +20% time); **`best_of` doesn't pay** with a
same-strength self-judge (−0.5 at 1.5× time). This is why `balanced` keeps
reflect and drops best-of, and why `best_of` is gated on `judge_model`.

### 2. Consistency detection — `run_consistency_eval` (does the cross-phase check work?)

`consistency_check` only acts *across* phases, so the single-phase ablation is
blind to it. This eval feeds the checker labeled (earlier-decisions,
new-decision) pairs — some genuinely contradictory, some clean — and reports a
confusion matrix with **precision** (don't false-alarm) and **recall** (don't
miss). It drives the live `DevSession._consistency_findings` code path.

Finding (8 cases, main-model judge): **precision 100%, recall 60%** — it caught
**3/3 blatant** contradictions with **zero false alarms**, but **0/2 subtle**
ones. That justifies keeping the lever (cheap, never cries wolf, catches blatant
cross-phase conflicts) while confirming subtle contradictions still need a manual
`dev revisit`.

### 3. Placeholder removal — `run_build_review_eval` (does the build-time review work?)

`build_review` is a per-file build-time lever (draft → self-review → fix), so it
isn't exercised by either design-phase eval. This one hands the live
`Builder._review_file` drafts containing a planted placeholder (`# TODO`,
`NotImplementedError`, `pass # stub`) and checks whether the review removed it
**and replaced it with a real implementation** (a deletion doesn't count) — plus
a clean control that must survive untouched.

Finding (4 cases, main-model judge): **fix rate 100% (3/3), preservation 100%** —
the review stripped every planted placeholder with a working implementation and
left the clean draft intact. Keeps `build_review` on in `balanced`.

## Usage

Needs LM Studio running with the model loaded (same setup as `aicoder`).

```bash
# 1. Lever ablation — baseline vs full stack, on the whatsapp/security fixture
python -m evals.run_eval

# isolate each lever to see its individual contribution
python -m evals.run_eval --configs baseline,reflect,best_of,full

# average 2 runs per config to cut noise; grade a different phase/fixture
python -m evals.run_eval --fixture invoicing --phase data_model --repeat 2

# grade (and run the critic steps) with a stronger model you've pulled
python -m evals.run_eval --judge-model qwen2.5-coder:14b --out results.json

# 2. Consistency detection — precision/recall on labeled contradiction cases
python -m evals.run_consistency_eval
python -m evals.run_consistency_eval --repeat 3 --judge-model qwen2.5-coder:14b

# 3. Build-review — placeholder removal on drafts with planted issues
python -m evals.run_build_review_eval
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
- `run_eval.py` — the lever-ablation runner + CLI.
- `consistency_fixtures.py` — labeled cross-phase contradiction cases.
- `run_consistency_eval.py` — the detection runner; `compute_metrics` /
  `detect_case` are pure and unit-tested, with the detector injectable.
- `build_review_fixtures.py` — drafts with planted placeholders + a clean control.
- `run_build_review_eval.py` — the placeholder-removal runner; `compute_metrics`
  / `judge_case` are pure and unit-tested, with the reviewer injectable.

## Caveats

- The judge is itself a model; for trustworthy grades use the strongest
  `--judge-model` you can run, and `--repeat` to average out noise.
- `security` (the default phase) needs no web research, so the eval is offline
  and repeatable. Research-backed phases (`architecture`, `api`, …) will hit the
  network and vary with live search results.
