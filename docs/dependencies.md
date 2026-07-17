# Dependencies

Every third-party library `aicoder` ships with, why it's there, and its
license — verified against what's actually installed in a real `.venv`
(`pip show <pkg>`, cross-checked with `importlib.metadata` classifiers where
`pip show` didn't print a `License` line), not just copied from
`pyproject.toml`. Re-verify with the same commands if a dependency is
bumped — installed metadata can drift from what a package's PyPI page
claims.

All licenses below are permissive (MIT/BSD/Apache-2.0-family) except
**pathspec** (MPL 2.0, file-level copyleft — see its row) — there is nothing
in the dependency tree that restricts using, modifying, or redistributing
`aicoder` itself.

## Runtime (`[project.dependencies]`)

| Package | Version installed | License | What it's for |
|---|---|---|---|
| `langchain-core` | 1.4.8 | MIT | The message/tool-call abstractions (`HumanMessage`, `AIMessage`, `BaseTool`, etc.) the whole agent loop is built on — provider-agnostic by design, which is what lets `aicoder` point at any OpenAI-compatible endpoint. |
| `langchain-openai` | 1.3.3 | MIT | `ChatOpenAI` — the actual client that talks to LM Studio's (or any) OpenAI-compatible `/v1/chat/completions` endpoint. See [Known gotchas](architecture.md#known-gotchas-for-future-changes) for a real limitation: it does not extract non-standard `reasoning_content` fields some local models stream. |
| `rich` | 15.0.0 | MIT | Terminal rendering primitives — `Panel`, `Table`, `Markdown`, styled text — used both directly (non-TUI REPL mode) and as the renderable layer Textual's `RichLog` displays. |
| `textual` | 8.2.8 | MIT | The TUI framework (`agent/tui.py`) — the full-screen app, modals, the `ChatInput`/`RichLog` widgets, async event loop. |
| `textual-autocomplete` | 4.0.6 | MIT | Slash-command autocomplete dropdown in the TUI's input box. |
| `pillow` | 12.3.0 | MIT-CMU | Image decoding/resizing so the agent can read image files (screenshots, diagrams) passed as context. |
| `pyyaml` | 6.0.3 | MIT | Parses/writes `~/.aicoder/config.yaml` — the whole config layer (`core/config.py`) is built on it. |
| `ddgs` | 9.14.4 | MIT | DuckDuckGo search backend for the agent's web-search tool — no API key required, matching the project's local-first stance. |
| `httpx` | 0.28.1 | BSD-3-Clause | HTTP client for fetching web pages (paired with `beautifulsoup4` for extraction) and any other outbound HTTP the tools need. |
| `beautifulsoup4` | 4.14.3 | MIT | HTML parsing/text extraction for the web-fetch tool. |
| `packaging` | 26.2 | Apache-2.0 OR BSD-2-Clause | Version-string parsing/comparison — used for install/update version checks. |
| `pathspec` | 1.1.1 | **MPL 2.0** | `.gitignore`-style pattern matching, so file-listing/search tools respect ignore rules the same way `git` does. MPL 2.0 is file-level copyleft (modifications to pathspec's *own* files would need to stay open), not project-wide — it does not affect `aicoder`'s own license, since it's used unmodified as a library dependency. |
| `chromadb` | 1.5.9 | Apache-2.0 | Local embedded vector store backing the project-memory/RAG layer (`rag/`, `memory/`) — runs on-disk, no server, no external service. |
| `pypdf` | 6.12.2 | BSD-3-Clause | PDF text extraction so the agent can read `.pdf` files as context. |
| `python-docx` | 1.2.0 | MIT | `.docx` text extraction, same purpose as `pypdf` for Word documents. |

## Dev extra (`pip install local-aicoder[dev]`)

| Package | Version installed | License | What it's for |
|---|---|---|---|
| `pytest` | 9.0.3 | MIT | Test runner for the whole `tests/` suite. |
| `pytest-asyncio` | 1.4.0 | Apache-2.0 | Runs the TUI's `async def` tests (Textual `Pilot`-driven) — `asyncio_mode = "auto"` in `pyproject.toml`. |
| `ruff` | 0.15.15 | MIT | Linter — the rule set is deliberately just Pyflakes + a pycodestyle subset (`select = ["E4", "E7", "E9", "F"]` in `pyproject.toml`), not a full style-enforcement config. |

## Optional extra (`pip install local-aicoder[mcp]`)

| Package | Version installed | License | What it's for |
|---|---|---|---|
| `mcp` | 1.27.2 | MIT | Model Context Protocol client support, so `aicoder` can connect to external MCP tool servers if configured — optional because most users never need it. |

## External, not a Python dependency

| Tool | License | What it's for |
|---|---|---|
| [LM Studio](https://lmstudio.ai) | Proprietary, free | The actual local LLM runtime — `aicoder` talks to it over its OpenAI-compatible HTTP API and drives its `lms` CLI (`server start`, `load`/`unload`, `ps --json`) for lifecycle management. See [LM Studio lifecycle management](architecture.md#lm-studio-lifecycle-management). Not bundled or installed by `aicoder`; `ensure_lmstudio_running` only automates *using* an existing LM Studio install. |

## How this list is verified

```bash
source .venv/bin/activate
pip show <package>          # Name / Version / License (or License-Expression)
```

A few packages don't print a `License:` line from `pip show` (their metadata
uses classifiers instead) — for those, check
`importlib.metadata.metadata(<package>)`'s `Classifier` entries for a
`License ::` line, as done here for `pathspec` (MPL 2.0) and `chromadb`
(Apache-2.0).
