# AICoder — Features Reference

AICoder v3 is a single **agentic loop**: you describe a task, and the model
calls tools to read and edit code, run commands, research the web, run tests,
and remember decisions. There is no fixed command pipeline — the model decides
which tools to use.

---

## The agent's tools

The model is given these tools (`agent/tools.py`) and calls them as needed. File
paths are sandboxed to the workspace.

### Code navigation
| Tool | Description |
|---|---|
| `list_files(path=".")` | List files/directories as a tree |
| `find_files(name_pattern, path=".")` | Find files by name glob (`*.py`, `*config*`) |
| `find_symbol(name)` | Jump to where a function/class/type is **defined** (symbol index) |
| `search_code(query, path=".")` | Grep file contents (`file:line: text`) |
| `read_file(path, offset, limit)` | Read a file; page large files by line range |

### Editing & execution
| Tool | Description |
|---|---|
| `write_file(path, content)` | Create or overwrite a file (diff + confirm) |
| `edit_file(path, old_string, new_string)` | Replace an exact, unique snippet (diff + confirm) |
| `run_shell(command)` | Run a shell command (confirmation per config) |
| `run_tests()` | Auto-detect and run the test suite, report pass/fail |
| `run_checks()` | Auto-detect and run linters / type checkers (ruff, mypy, eslint, tsc, clippy, go vet) |
| `git_status()` / `git_diff(path)` | Review changes (read-only) |
| `git_commit(message)` | Stage (excluding `.bak`) and commit (confirmation per config) |

`edit_file` tolerates minor whitespace/indentation differences: it matches at
line granularity (so a whole line is replaced, never a fragment) and replaces
using the file's actual text.

### Research & knowledge (RAG)
| Tool | Description |
|---|---|
| `research(query)` | Cache-first web lookup; caches findings and cites sources |
| `fetch_url(url)` | Fetch a page, cache it, return the text |
| `rag_search(query)` | Recall previously researched/ingested knowledge |
| `read_document(path)` | Extract & ingest a PRD/TDD (PDF, docx, md, txt, html) |

### Memory
| Tool | Description |
|---|---|
| `remember(note, category)` | Save a durable project fact (decision/convention/fact/todo) |
| `recall(query="")` | Retrieve saved project facts |

---

## In-session commands

Most of the time you just type a request in plain English. A few literal
commands are handled by the REPL (`agent/loop.py`):

| Command | Description |
|---|---|
| `develop <idea>` | Developer Mode: role-driven SDLC design → build (see below) |
| `dev …` | `dev` (resume) · `dev status` · `dev build` · `dev revisit <phase>` · `dev resolve` |
| `plan <goal>` | Decompose a goal into an ordered, resumable task list and build it |
| `resume` | Continue an in-progress plan |
| `/model [name]` | Show or switch the model for this session |
| `/tools` | List the agent's tools |
| `/diff` | Show the git diff of changes so far |
| `/memory` | Show what's remembered about this project |
| `/knowledge` | RAG: `/knowledge learn <topic\|URL>` researches & caches; bare shows stats; `/knowledge clear` clears this project's docs, `/knowledge clear all` wipes everything |
| `/clear` | Forget the conversation (keeps saved memory) |
| `/help` | Show commands |
| `exit` / `quit` | Leave the session |

---

## The agentic loop

Per user turn (`AgentSession.send`):

1. The model is streamed with the conversation + bound tools (tokens appear
   live in a transient region; tool-call noise is erased, final answers render
   as Markdown).
2. **Native tool calls** are executed; each result is fed back as a tool message.
3. **Fallback:** if the model emitted tool calls as JSON *text* instead of
   natively (common with local models like `qwen2.5-coder`), they are recovered
   (`core/model.extract_text_tool_calls`), filtered to known tools, executed, and
   the results fed back.
4. When the model returns a plain answer (no tool calls), it is shown and the
   turn ends. A step cap (12) bounds runaway loops.

**Context management:** before each turn, if the conversation exceeds a char
budget (~2× the model's context window in chars), older turns are summarized
into a single note while recent turns are kept verbatim — split only at a
user-message boundary so a tool result is never orphaned from its call. This
keeps long sessions and large `plan` builds within the context window.

---

## Planning large tasks

`plan <goal>` (`agent/planner.py`):

- Asks the model for an ordered JSON task list, grounded in any ingested
  document via RAG.
- Persists the plan to `~/.aicoder/memory/<project_id>/plan.json`.
- Executes each task through the agent, saving status after every step, and
  pausing for confirmation between tasks.
- **Resumable** — quit anytime; `resume` continues from the first pending task.

---

## Developer Mode (`devmode/`)

A role-driven SDLC flow for building real applications with full control. The
engine (`devmode/session.py`) is **data-driven**: each phase is a `PhaseSpec`
(`devmode/phases.py`) with a role, a focus, an output artifact, and optional
flags. The same discussion loop runs for every phase except the review-kind one.

- **14 phases** — Product Vision → Market & Competitors → Requirements →
  Architecture → Security/NFR → Data Model → API → App Flow → UI/UX → Testing →
  Deployment → Documentation → Conventions (writes `AICODER.md`) → Design Review.
- **Artifacts** (`docs/dev/*.md` + `state.json`) are the resumable source of
  truth the build reads; conventions go to `AICODER.md`.
- **Brownfield-aware** — for an existing repo every phase is grounded in the
  codebase and the Conventions phase infers your current style.
- **`dev build`** (`devmode/build.py`) — proposes a file plan
  (`docs/dev/build_plan.json`, user-editable), then generates each file grounded
  in the spec + conventions (resumable per file), and **closes the loop**: a
  compile check → tests → agentic-fix loop (≤3 rounds, finds a nested project
  root) gets the code running. Writes `build_manifest.json` (file → phases).
- **`dev revisit <phase>`** / **`dev resolve`** — change a decision, or
  review→fix cross-phase contradictions; both **auto-resync** the code
  (`devmode/resync.py`) via an agentic diff→apply→verify task.
- **`develop --fast <idea>`** — runs the whole design in one pass; each role
  makes its own senior decisions with no back-and-forth (still applies the active
  profile's levers).

### Quality levers (driving a small local model)

The toggleable levers are bundled into one **`devmode.profile`** dial —
`fast` (reflect only), **`balanced`** (default: reflect + consistency_check +
build_review), or `thorough` (everything). An individual lever can still be
overridden in config (`Config.devmode_lever()` resolves profile + override).

| Lever | Where | Config | In `balanced`? |
|---|---|---|---|
| Must-cover checklists | `phases._MUST_COVER` — forces domain-defining decisions | always on | ✓ |
| Reflection (draft→critique→revise) | `session._one_decision` | `reflect` | ✓ |
| Decomposition (list→detail-each→assemble) | `session._summarize_decomposed` (`_DECOMPOSE`) | always on | ✓ |
| Targeted multi-query research | `session._research_queries` / `_research` | always on | ✓ |
| Best-of-N + judge | `session._summarize` / `_judge_best` (`_BEST_OF`) | `best_of` | — (needs `judge_model`) |
| Cross-phase consistency check | `session._report_consistency` (digest-based) | `consistency_check` | ✓ |
| Build self-review | `build._review_file` | `build_review` | ✓ |
| Build verify→fix loop | `build._verify_and_fix` (compile → tests → agentic fix) | always on | ✓ |
| Resolve (fix + resync) | `session.resolve` / `_apply_fix` | always on | ✓ |
| Hybrid judging (stronger critic) | `session._critic_stream` | `judge_model` | opt-in |

**`best_of` is gated on `judge_model`** — best-of-N only fires when a stronger
critic is configured to rank candidates; otherwise it's skipped in favour of a
single reflected pass (a same-strength self-judge added latency without quality).

**Measured effect** (`evals/`, qwen2.5-coder:7b judging itself):

- `reflect` lifts a single security-design phase from **7.5 → 9.5 / 10** (75% →
  100% checklist coverage) for ~20% more time — it carries essentially all the
  gain, which is why it's on in every profile.
- `consistency_check` scores **100% precision / 60% recall** on labeled
  contradictions: it catches every *blatant* cross-phase conflict with zero
  false alarms, but misses *subtle* ones.
- `build_review` removes **100%** of planted placeholders (TODO / stub /
  `NotImplementedError`) while leaving clean drafts intact.

**The honest ceiling:** subtle semantic contradictions a 7B can't reason through
(e.g. a private key stored server-side that the artifact rationalizes as
"encrypted at rest") may still pass — review the output, and use `dev revisit` /
`dev resolve` for the subtle cases. See [`evals/README.md`](../evals/README.md).

---

## Tool capabilities (details)

### File operations (`tools/file_tools.py`)
- **Path safety:** all operations resolved against the workspace root; traversal
  is rejected (`Path escapes workspace`).
- **Encoding fallback:** utf-8 with `replace` (never crashes on binary files).
- **Backup:** writes a `.bak` copy before overwriting (configurable).
- **Diff preview:** Rich syntax-highlighted diffs before applying.
- **Ignore-aware search:** `search_code` skips `.venv`, `node_modules`,
  `__pycache__`, etc.

### Shell execution (`tools/shell_tools.py`)
- **`always`** — ask before every command (default).
- **`smart`** — auto-run safe commands, ask for destructive ones (`rm`, `drop`,
  `-rf`, `--force`, …).
- **`never`** — auto-run everything.
- 120s timeout per command.

### Web integration (`tools/web_tools.py`)
- DuckDuckGo search (no API key) via `ddgs`.
- URL fetch (follows redirects, sends a User-Agent), HTML stripped to readable
  text via BeautifulSoup.

### RAG knowledge base (`rag/store.py`, `rag/ingest.py`)
- **Storage:** ChromaDB at `~/.aicoder/rag/chroma/`.
- **Embeddings:** Ollama `nomic-embed-text` (configurable; `""` = use the chat model).
- **Chunking:** real overlapping chunks (≈1200 chars, 150 overlap).
- **Relevance cutoff:** cosine-distance threshold so unrelated queries return
  nothing instead of the nearest irrelevant chunk.
- **Document loaders:** PDF (pypdf), Word (python-docx, incl. tables),
  Markdown/txt/rst, HTML.
- **Scoping:** web research is **global** (shared cache across projects);
  ingested documents are **per-project** (tagged by workspace, so a PRD from one
  project doesn't surface in another). Recall returns this project's docs + the
  global web cache.

### Test & check detection (`core/project.py`)
- Auto-detects the test command from marker files: pytest, npm/yarn/pnpm test,
  cargo, go, make, maven, gradle. For pytest it prefers the project's `.venv`.
- Auto-detects linters / type checkers: ruff and mypy (only if configured in
  `pyproject.toml`/config), flake8, eslint / tsc (Node), clippy (Rust), go vet.

### Persistent memory (`memory/project.py`)
- **Storage:** `~/.aicoder/memory/<project_id>/project_memory.json`, keyed by
  workspace path.
- Durable categorized facts (decision/convention/fact/todo), idempotent add,
  auto-loaded into the system prompt at session start.

### Hooks (`agent/hooks.py`)
- Optional user shell commands on agent events, configured under `hooks` in
  config: **PreToolUse** (non-zero exit blocks the tool), **PostToolUse**
  (auto-format/notify), **Stop** (turn finished). Matched by a regex on the tool
  name. Commands get a JSON payload on stdin + `AICODER_*` env vars. Opt-in.

### MCP servers (`agent/mcp_client.py`)
- Optional [Model Context Protocol](https://modelcontextprotocol.io/) client.
  Configure stdio servers under `mcp.servers` in config; their tools are
  discovered and exposed to the agent (prefixed `<server>__<tool>`) alongside
  the built-ins. MCP sessions run on a background event loop bridged to the sync
  agent loop. Requires `pip install "ai-coder[mcp]"`; opt-in.

### Project instructions (`AICODER.md`)
- A user-authored `AICODER.md` in the workspace root (and an optional global
  `~/.aicoder/AICODER.md`) is loaded into the system prompt every session, so
  the agent follows your conventions/rules. It takes precedence over defaults.
  Also recognized: `.aicoder.md`, `.aicoderrules`.

### Workspace context (`core/context.py`)
- Scans the tree (depth 3), detects languages, and injects a compact repo
  overview into the system prompt (no file-content dumps — the agent reads on
  demand).
