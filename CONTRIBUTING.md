# Contributing to AICoder

Thanks for considering a contribution. AICoder is a small, local-first project —
please **open an issue before starting on anything significant** (a new
feature, a behavior change, a new dependency) so we can agree on the approach
before you invest time in it. Small, focused fixes (typos, bugs, docs) can go
straight to a PR.

## Dev setup

```bash
git clone https://github.com/kiranchenna/ai-coder
cd ai-coder
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # installs pytest + ruff
```

You'll also need [LM Studio](https://lmstudio.ai/) running locally with a
model loaded to exercise anything that actually calls the LLM — but the test
suite itself does **not** require LM Studio (model calls are mocked/injected).

## Before opening a PR

```bash
pytest -q          # the full suite must pass
ruff check .       # must be clean — see [tool.ruff] in pyproject.toml
```

There's no CI gate yet, so these two commands are the review bar — please run
them locally and mention the results in your PR description.

## Code style

- Follow the existing conventions in the file/module you're touching over any
  external style guide — this codebase is internally consistent (modern type
  hints, `from __future__ import annotations` where forward refs need it,
  local `from x import y` inside functions as the established lazy-import
  pattern).
- Keep changes minimal and focused — this project prefers small, targeted
  diffs over broad refactors.
- Add tests for new behavior. Look at the existing `tests/test_*.py` for the
  patterns already in use (fixtures, monkeypatching, isolated config via a
  temp `AICODER_HOME`/`CONFIG_PATH`).

## Touching Developer Mode's quality levers?

If you're changing `reflect` / `best_of` / `consistency_check` / `build_review`
or the `devmode.profile` defaults, please back the change with the
[`evals/`](evals/) harness (see `evals/README.md`) rather than intuition — the
current defaults are evidence-based, not assumed, and a change that regresses
measured quality/latency should show that in the eval output.

## Commit messages

Explain *why*, not just *what* — a one-or-two sentence summary is enough for
small changes. See `git log` for the house style.

## Reporting bugs

Open an issue with:
- Your OS, LM Studio version, and the model you're using
- The exact command/prompt and the full error or unexpected output
- Whether `aicoder --selftest` passes

## License

By contributing, you agree your contribution is licensed under this project's
[MIT license](LICENSE).
