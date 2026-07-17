# AICoder — Documentation Index

AICoder v4 — a local, offline **agentic coding assistant** (LM Studio, no cloud, no
API keys). It plans, reads and edits real code, runs commands and tests,
researches the web, and remembers your project — all on your machine.

## Documents

| Document | For | Covers |
|---|---|---|
| [../README.md](../README.md) | everyone | Quickstart, [usage examples](../README.md#usage-examples), install, all commands, configuration reference |
| [features.md](features.md) | users | How the product works — the agentic loop, the tools, in-session commands, planning, **Developer Mode** and its quality levers |
| [architecture.md](architecture.md) | contributors | How it's built — directory structure, tech stack, data flow, config resolution, **LM Studio lifecycle management**, **the eval harness**, data formats, known framework gotchas |
| [dependencies.md](dependencies.md) | contributors | Every third-party library used, why, and its license — verified against what's actually installed |
| [support.md](support.md) | users | Support & FAQ — troubleshooting, common questions, the safety model, known limitations, reporting issues |
| [../evals/README.md](../evals/README.md) | contributors | The Developer Mode measurement harness — what each eval proves and how to run it |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | contributors | Dev setup, the test/lint bar for PRs, code style, and how to back devmode lever changes with evals |
| [../CHANGELOG.md](../CHANGELOG.md) | everyone | Release notes |

## Start here

- **New user?** [README](../README.md) → quickstart and
  [usage examples](../README.md#usage-examples), then [support.md](support.md)
  if anything misbehaves.
- **Want to understand a feature?** [features.md](features.md).
- **Contributing or curious how it works?** [architecture.md](architecture.md),
  [dependencies.md](dependencies.md), [../evals/README.md](../evals/README.md),
  and [../CONTRIBUTING.md](../CONTRIBUTING.md).
- **Need to change code manually and want to know what's safe to touch?**
  [architecture.md#known-gotchas-for-future-changes](architecture.md#known-gotchas-for-future-changes).
