# AICoder ✨

> A local, offline **agentic coding assistant** — it plans, reads and edits real code, runs commands and tests, researches the web, and remembers your project — all running on your own machine via [Ollama](https://ollama.com/). No API keys. No cloud. Your code never leaves your computer.

---

## Table of contents

- [What is AICoder?](#what-is-aicoder)
- [Key features](#key-features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Choosing a model](#choosing-a-model)
- [Quick start](#quick-start)
- [Command-line usage](#command-line-usage)
- [Talking to the agent](#talking-to-the-agent)
- [In-session commands](#in-session-commands)
- [Multi-step builds (`plan` / `resume`)](#multi-step-builds-plan--resume)
- [Developer Mode](#developer-mode)
- [The tools](#the-tools)
- [Verifying changes (tests, lint, type-check)](#verifying-changes)
- [Git integration](#git-integration)
- [Web research & the knowledge base (RAG)](#web-research--the-knowledge-base-rag)
- [Working from documents (PRDs/specs)](#working-from-documents)
- [Memory & project instructions](#memory--project-instructions)
- [Extending AICoder](#extending-aicoder)
  - [MCP servers](#mcp-servers)
  - [Hooks](#hooks)
- [Configuration reference](#configuration-reference)
- [Safety & confirmation modes](#safety--confirmation-modes)
- [Where your data lives](#where-your-data-lives)
- [Project layout](#project-layout)
- [Architecture](#architecture)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

---

## What is AICoder?

AICoder is an interactive terminal assistant that works on **your actual repository**. You describe a task in plain English (or point it at a document), and instead of just *talking* about code, it **takes real actions** through a set of tools: it reads and edits files, runs shell commands, runs your tests and linters, searches the web for current information, and records what it learns.

The core is an **agentic loop**: you give it a task → the model decides which tools to use → it executes them → reads the results → repeats until the job is done. You stay in control — it shows diffs and asks before risky actions.

It's deliberately **100% local and offline**. That means privacy and zero cost, with one honest tradeoff: it runs small local models (7B-class on a typical laptop), so it's best thought of as a **capable pair-programmer you supervise** rather than a fully autonomous engineer. The bigger the local model your hardware can run, the better the results.

---

## Key features

- 🤖 **Agentic loop** — the model calls tools to get work done, with live token streaming.
- 🏗 **Developer Mode** — a role-driven SDLC: design an app through 14 expert-role phases (captured as editable files), then build it — with quality levers that lift a small local model's output ([details](#developer-mode)).
- 🛠 **Works on any repo** — build new code, modify existing code, add features, fix bugs.
- 🔎 **Code intelligence** — jump to definitions (`find_symbol`), search contents, page through large files.
- ✅ **Verifies its own work** — auto-detects and runs your tests, linters, and type checkers.
- 📄 **Document-driven** — ingest a PRD/TDD (PDF, Word, Markdown) and build from it.
- 🌐 **Stays current** — web research cached into a local vector store (RAG), so it isn't limited to the model's training cutoff.
- 🧠 **Remembers** — durable per-project memory + a user-authored `AICODER.md` instructions file, auto-loaded each session.
- 📋 **Plans big tasks** — decomposes a goal into an ordered, **resumable** task list.
- 🔧 **Git built in** — review and commit changes from the conversation.
- 🔌 **Extensible** — connect **MCP servers** for more tools, and add **hooks** to run your scripts on events.
- 🔒 **You-in-the-loop** — configurable confirmation for file writes and shell commands.

---

## How it works

Each time you send a message:

```
your message
  └─ model.stream(conversation + tools)        ← tokens appear live
       ├─ the model requests tool calls  ──→ AICoder executes them, feeds results back ──┐
       │                                                                                 │
       └─ ... repeats until the model returns a plain answer ←───────────────────────────┘
            └─ answer rendered as Markdown
```

- **Tool calls** are executed (file edits show a diff and, per your settings, ask for confirmation; shell commands are gated by the shell mode).
- A **step cap** bounds runaway loops.
- Some local models emit tool calls as JSON *text* rather than via native tool-calling — AICoder **recovers and runs those too**.
- Long conversations are automatically **compacted** (older turns summarized) to stay within the model's context window.

---

## Requirements

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** installed and running locally
- A pulled chat model (and, for web/document RAG, an embedding model)

---

## Installation

### From PyPI

```bash
pip install ai-coder
```

Optional extras:

```bash
pip install "ai-coder[mcp]"     # MCP server support
```

### From source (development)

```bash
git clone https://github.com/kiranchenna/ai-coder
cd ai-coder
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # dev extras include pytest
```

### Pull the models

```bash
ollama pull qwen2.5-coder:7b   # the agent driver
ollama pull nomic-embed-text   # embeddings (web research + documents)
```

### Verify

```bash
aicoder --selftest    # confirms the configured model can call tools
```

---

## Choosing a model

The default is **`qwen2.5-coder:7b`** — a good balance of code quality and tool-calling reliability. Pick based on your hardware (the model and its context share memory):

| RAM / VRAM | Suggested model | Notes |
|---|---|---|
| 8 GB | `qwen2.5-coder:3b` / `qwen3:4b` | fast, weaker at multi-step work |
| 16 GB | **`qwen2.5-coder:7b`** / `qwen3:8b` | the sweet spot |
| 24 GB+ | `qwen2.5-coder:14b`, `qwen3-coder:30b` | strongest, needs more memory |

Switch models with `aicoder --model <name>`, the in-session `/model <name>` command, or by editing `~/.aicoder/config.yaml`. Run `aicoder --selftest` after switching to confirm the model supports tool calling.

> **Embeddings** (`nomic-embed-text` by default) are only needed for web research and document ingestion. Alternatives: `bge-m3`, `nomic-embed-text-v2-moe`.

---

## Quick start

```bash
cd my-project
aicoder
```

You'll get a prompt. Just describe what you want:

```
my-project> add input validation to the create_user endpoint and run the tests
```

The agent will find the file, read it, make the edit (showing you a diff), run your tests, and fix anything that fails — then summarize what it changed.

Point it at a different directory, or override the model for one session:

```bash
aicoder --workspace ./another-project
aicoder --model qwen2.5-coder:14b
```

---

## Command-line usage

```
aicoder [options]
```

| Flag | Description |
|---|---|
| `--workspace`, `-w PATH` | Project directory to work in (default: current directory) |
| `--model`, `-m MODEL` | Ollama model to use this session (overrides config) |
| `--shell-mode {always,never,smart}` | Shell confirmation mode for this session |
| `--selftest` | Verify the model supports tool calling, then exit |
| `--config` | Show the config file path and current settings, then exit |
| `--version` | Print the version |

If Ollama isn't running or the model isn't pulled, AICoder warns you at startup.

---

## Talking to the agent

Most of the time you just type a request in plain English. Examples:

```
explain how authentication works in this repo
why is test_login failing? find and fix it
add a /health endpoint that returns {"status": "ok"} and a test for it
refactor utils.py to use pathlib instead of os.path
read the spec at docs/PRD.pdf and scaffold the service it describes
what's the latest stable version of httpx, and pin it in requirements.txt
```

The agent navigates the repo itself (it won't ask you where a file is — it searches), shows diffs before applying edits, runs tests/linters to verify, and keeps you informed.

---

## In-session commands

A few literal commands are handled by the REPL; everything else is a task for the agent.

| Command | Description |
|---|---|
| `plan <goal>` | Decompose a goal into an ordered, resumable task list and build it |
| `resume` | Continue an in-progress plan |
| `/model [name]` | Show or switch the model for this session |
| `/tools` | List all available tools (built-in + MCP) |
| `/diff` | Show the git diff of changes so far |
| `/memory` | Show what's remembered about this project |
| `/knowledge [learn <topic\|URL> \| clear \| clear all]` | Manage the RAG knowledge base (see below) |
| `/clear` | Forget the current conversation (keeps saved memory) |
| `/help` | List commands |
| `exit` / `quit` | Leave the session |

---

## Multi-step builds (`plan` / `resume`)

For a large goal, use `plan`:

```
my-project> plan build a FastAPI todo service from docs/PRD.md
```

AICoder decomposes the goal (grounded in any ingested document) into an ordered task list, then executes each task — reading/writing files and verifying as it goes — pausing for your confirmation between tasks.

It's **resumable**: quit anytime, and next session type `resume` to continue from the first unfinished task. Plan state is saved under `~/.aicoder/memory/<project>/plan.json`.

---

## Developer Mode

For building real applications with full control, **Developer Mode** runs a **role-driven SDLC** — it discusses each stage with you (as a different expert role), captures every decision as an editable file, and only then builds. You stay in control of the tech stack, schema, architecture, flows, screens, and the exact code structure.

```
my-project> develop a multi-tenant invoicing SaaS with Postgres and a React UI
```

### The phases

It walks these phases, each a **full back-and-forth discussion** with a role persona — research-enabled phases pull current versions/best-practices from the web:

| # | Phase | Role |
|---|---|---|
| 1 | Product Vision | Product Manager |
| 2 | Market & Competitors | Market Analyst |
| 3 | Requirements | Requirements Analyst |
| 4 | Architecture & Tech Stack | Software Architect |
| 5 | Security & Non-Functional | Security/Platform Engineer |
| 6 | Data Model & DB Schema | Database Architect |
| 7 | API & Interface Contracts | Backend Engineer |
| 8 | Application Flow & Business Logic | Domain Engineer |
| 9 | UI/UX — Screens & Behaviour | Frontend/UX Engineer |
| 10 | Testing Strategy | QA Engineer |
| 11 | Deployment & Infrastructure | DevOps Engineer |
| 12 | Documentation Plan | Technical Writer |
| 13 | Coding Conventions | Tech Lead → writes `AICODER.md` |
| 14 | Design Review | Design Reviewer (critiques all decisions before build) |

In each design phase, type `done` to capture the decision, `skip` to skip, `revise` to restart, or `pause` to stop and resume later. The final **Design Review** doesn't propose a decision — it critiques the others (consistency, gaps, security/scale risks) and points you to `dev revisit <phase>` to fix anything.

### Artifacts you control

Every decision is written to a file you can read, edit, and commit — these are the **source of truth** the build reads:

```
docs/dev/
├── state.json            # phase progress (resumable)
├── 01_requirements.md    # decision + discussion transcript
├── 02_architecture.md
├── … 04_data_model.md, 05_api.md, …
└── build_plan.json       # the file/folder plan — edit it to control structure
AICODER.md                # the coding conventions the build follows
```

### Build, revisit, resync

```
develop <idea>        # start (or resume) the design
develop --fast <idea> # design the whole thing in one pass (roles decide; no back-and-forth)
dev                   # resume the design
dev status            # show phase progress
dev build             # turn the design into code — proposes a file plan you can
                      #   edit (build_plan.json), then generates file-by-file and verifies
dev revisit <phase>   # re-open a decision; if it changes, auto-resync the code to match
dev resolve           # cross-phase review → fix the design contradictions → resync code
```

- **`dev build`** proposes the folder/file structure from the design + your conventions. **Edit `docs/dev/build_plan.json`** (paths, order, naming) and re-run to use your exact structure. It then generates each file — grounded in the spec + `AICODER.md`, shown as a diff, **resumable per file** — and closes the loop: a **compile check → tests → agentic-fix loop** (up to 3 rounds) gets the code actually running, even when the project lives in a subdirectory. It also writes `docs/dev/build_manifest.json` mapping each file to the design phases it implements.
- **`dev revisit <phase>`** lets you change any decision later. If the decision changed and code was built, AICoder **auto-resyncs**: it diffs old→new and runs an agentic task to propagate the change through the code, then verifies.
- **`dev resolve`** reviews every phase together, lists the cross-phase contradictions (e.g. a schema that stores plaintext despite an end-to-end-encryption promise, or an auth mechanism that disagrees with the security phase), and for each one you accept it **rewrites the offending phase's decision and auto-resyncs the code**. It catches blunt contradictions reliably; subtle ones a small local model can't reason through may still need a manual `dev revisit`.

### Greenfield and existing repos

- **Greenfield:** you specify the conventions in the Conventions phase.
- **Existing repo (brownfield):** every phase is grounded in your codebase, and the Conventions phase **infers your current conventions** from the code for you to confirm/adjust — so generated code matches your existing style.

### How it gets quality from a small model

A local 7B model doesn't know which parts of a domain are hard, and it writes a weak first draft. Developer Mode compensates with **engineering, not a bigger model**. The levers are bundled into a single `devmode.profile` dial — **`fast`** (reflect only), **`balanced`** (the default: reflect + consistency + build-review), or **`thorough`** (everything, including best-of-N). You can still override any individual lever in config.

> **Why these defaults?** A lever ablation (see [`evals/`](evals/)) on the security-design phase found that **`reflect` carries essentially all of the quality gain** (+2.0/10, 70%→100% checklist coverage, for ~20% added time), while **`best_of` only pays with a stronger judge** — with a same-strength self-judge it added latency without quality. So `balanced` keeps reflect and drops best-of, and **`best_of` is gated on `judge_model`**: it only fires when you've configured a stronger critic model to rank the candidates. Two more evals back the rest of `balanced`: `consistency_check` measured **100% precision / 60% recall** (caught every *blatant* cross-phase contradiction with zero false alarms, missed the *subtle* ones — so it stays as cheap insurance while subtle conflicts still want a manual `dev revisit`), and `build_review` measured a **100% placeholder-removal rate** with clean drafts left intact. Run them yourself: `python -m evals.run_eval`, `run_consistency_eval`, `run_build_review_eval`.

The levers, each independently toggleable:

- **Must-cover checklists** — each phase carries a senior checklist the model is *forced* to address (e.g. Security must name the actual E2E protocol and per-device keys; Architecture must name the real-time backbone), so it can't skip the defining decisions.
- **Reflection** (`reflect`) — every decision is drafted, then critiqued and revised in a second pass; a small model improves a concrete draft far better than it writes a perfect one first try.
- **Decomposition** — the heavy phases (data model, API, architecture) are designed **one unit at a time** (list → detail each entity/endpoint/component → assemble), which a small model handles far better than one giant answer.
- **Targeted research** — research phases derive 2–3 *specific* web queries (current versions, protocols, pitfalls) instead of one generic search, putting real current facts in context.
- **Best-of-N** (`best_of`) — for the critical phases (requirements, security) it generates several candidate decisions from different angles and a judge keeps the strongest.
- **Cross-phase consistency check** (`consistency_check`) — after each phase, its decision is checked against the earlier ones and contradictions are flagged (and logged to `docs/dev/consistency_notes.md`).
- **Build self-review** (`build_review`) — every generated file is critiqued for bugs, placeholders, and convention misses, then fixed, before it's written.
- **Build verify→fix loop** — after generation, a compile check → tests → agentic-fix loop (≤3 rounds) gets the code actually running, not just plausible-looking.
- **`dev resolve`** — turns those contradictions into fixes: it rewrites the offending phase and auto-resyncs the code.
- **Hybrid judging** (`judge_model`, opt-in) — point the *critic* steps (best-of judging, consistency, review) at a stronger model while generation stays local — the cheapest way to push past what a 7B can reason through.

> Reality check: these levers measurably lift output (on a WhatsApp-clone design test the score rose from ~5.9 to ~8.2 / 10), but a local 7B is still a strong *assistant*, not an autonomous senior engineer — review the generated code, lean on the verify step, and use `dev resolve` / `dev revisit` to correct decisions. Subtle contradictions a 7B can't reason through may still slip past. The design/decision artifacts are valuable on their own, regardless of model strength.

---

## The tools

The model is given these tools and calls them as needed. All file paths are **sandboxed to the workspace**.

### Navigation & search
| Tool | Purpose |
|---|---|
| `list_files(path=".")` | List a directory as a tree |
| `find_files(name_pattern, path=".")` | Find files by name glob (`*.py`, `*config*`) |
| `find_symbol(name)` | Jump to where a function/class/type is **defined** (symbol index) |
| `search_code(query, path=".")` | Grep file contents (`file:line: text`) |
| `read_file(path, offset=1, limit=0)` | Read a file; page large files by line range |

### Editing & execution
| Tool | Purpose |
|---|---|
| `write_file(path, content)` | Create or overwrite a file (diff + confirmation + backup) |
| `edit_file(path, old_string, new_string)` | Replace a snippet — tolerant of minor whitespace/indentation differences |
| `run_shell(command)` | Run a shell command (confirmation per shell mode) |
| `run_tests()` | Auto-detect and run the test suite (pytest, npm, cargo, go, …) |
| `run_checks()` | Auto-detect and run linters / type checkers (ruff, mypy, eslint, tsc, clippy, go vet) |

### Git
| Tool | Purpose |
|---|---|
| `git_status()` / `git_diff(path)` | Review changes (read-only) |
| `git_commit(message)` | Stage (excluding `.bak`) and commit (confirmation per shell mode) |

### Knowledge & web
| Tool | Purpose |
|---|---|
| `research(query)` | Cache-first web lookup that caches findings and cites sources |
| `fetch_url(url)` | Fetch and cache a specific page |
| `rag_search(query)` | Recall from the cached knowledge base |
| `read_document(path)` | Extract & ingest a PRD/TDD (PDF/docx/md/txt/html) |

### Memory
| Tool | Purpose |
|---|---|
| `remember(note, category)` | Save a durable project fact (decision/convention/fact/todo) |
| `recall(query="")` | Retrieve saved project facts |

…plus any tools from configured [MCP servers](#mcp-servers).

---

## Verifying changes

After editing code, the agent verifies it:

- **`run_tests`** auto-detects the test command from marker files: pytest, `npm`/`yarn`/`pnpm test`, `cargo test`, `go test`, `make test`, Maven, Gradle. For pytest it prefers the project's own `.venv`.
- **`run_checks`** auto-detects linters / type checkers: **ruff** and **mypy** (only if configured in your `pyproject.toml`), **flake8**, **eslint**/**tsc** (Node), **clippy** (Rust), **go vet**.

If something fails, the agent reads the output, fixes the cause, and re-runs until clean — or explains what's wrong.

---

## Git integration

- Review the working tree at any time with **`/diff`**, or have the agent call `git_status` / `git_diff`.
- Have the agent commit a coherent set of changes with `git_commit` (it stages everything **except** the agent's `.bak` backups, and respects your shell confirmation mode).
- Shell quoting is cross-platform (POSIX and Windows `cmd.exe`).

---

## Web research & the knowledge base (RAG)

Local models have a training cutoff. AICoder works around that with retrieval:

- The agent can **`research`** a topic on the web (DuckDuckGo) and **cache** the results + top pages in a local **ChromaDB** vector store, then **`rag_search`** to recall them.
- Content is chunked and embedded (via your Ollama embedding model), with a relevance cutoff so unrelated queries return nothing rather than noise.
- **Scoping:** web research is **global** (a shared cache across projects), while ingested **documents are per-project** (a PRD from one project won't surface in another).

Manage it from the REPL:

```
/knowledge                       # stats (total / this-project chunks / path)
/knowledge learn "FastAPI 0.118" # proactively research a topic and cache it
/knowledge learn https://docs... # fetch and cache a specific page
/knowledge clear                 # clear this project's ingested documents
/knowledge clear all             # wipe the entire knowledge base
```

---

## Working from documents

Point the agent at a product document and it ingests the text for grounding:

```
my-project> read the PRD at docs/spec.pdf and summarize what we need to build
```

`read_document` supports **PDF** (pypdf), **Word `.docx`** (python-docx, including tables), **Markdown**, **`.txt`/`.rst`**, and **HTML**. The extracted text is stored (scoped to the project) so the agent — and the planner — can ground their work in what the document actually says.

---

## Memory & project instructions

AICoder remembers across sessions in two ways:

**1. Durable project memory.** The agent saves facts with `remember` (decisions, conventions, TODOs) and they're auto-loaded into context every session, so "continue where we left off" works days later. View it with `/memory`. Stored at `~/.aicoder/memory/<project>/project_memory.json`.

**2. `AICODER.md` — your project instructions.** Drop an `AICODER.md` in your project root with rules the agent should always follow. It's loaded into the agent's context every session and **takes precedence** over its defaults.

```markdown
# AICODER.md
- Use snake_case and full type hints.
- Tests live in tests/ and run with pytest.
- Never edit anything under vendor/.
- Prefer pathlib over os.path.
```

A global `~/.aicoder/AICODER.md` is also loaded (applies to every project), with the per-project file layered on top. (`.aicoder.md` and `.aicoderrules` are also recognized.)

---

## Extending AICoder

### MCP servers

Connect [Model Context Protocol](https://modelcontextprotocol.io/) servers and their tools become available to the agent alongside the built-ins — a database, GitHub, a browser, your own server, anything that speaks the protocol.

```bash
pip install "ai-coder[mcp]"
```

```yaml
# ~/.aicoder/config.yaml
mcp:
  servers:
    filesystem:
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
    sqlite:
      command: uvx
      args: ["mcp-server-sqlite", "--db-path", "./app.db"]
```

Each server's tools appear in `/tools`, prefixed by server name (e.g. `filesystem__read_file`). Opt-in — nothing runs unless you configure servers. (Currently **stdio** transport.)

### Hooks

Run your own shell commands on agent events — guard or block tools, auto-format after edits, or get notified.

```yaml
# ~/.aicoder/config.yaml
hooks:
  PreToolUse:                          # before a tool runs; non-zero exit BLOCKS it
    - matcher: "run_shell"             # regex on the tool name (omit = all tools)
      command: "my-guard.sh"
  PostToolUse:                         # after a tool runs
    - matcher: "write_file|edit_file"
      command: "ruff format ."         # auto-format on every edit
  Stop:                                # when a turn finishes
    - command: "osascript -e 'display notification \"AICoder done\"'"
```

Each command receives a JSON payload on stdin and `AICODER_EVENT` / `AICODER_TOOL` / `AICODER_TOOL_ARGS` env vars. A `PreToolUse` hook that exits non-zero blocks the tool (its output becomes the reason the agent sees). Hooks run arbitrary commands you configure — **only add ones you trust**.

---

## Configuration reference

Auto-created at `~/.aicoder/config.yaml` on first run. Key settings (abridged):

```yaml
model:
  provider: ollama
  name: qwen2.5-coder:7b          # any model you've pulled
  base_url: http://localhost:11434
  temperature: 0.3                # conversational
  temperature_precise: 0.1        # for precise/code output
  context_length: 16384           # num_ctx; also drives history-compaction budget

shell:
  confirmation: always            # always | smart | never

files:
  confirmation: auto              # always (ask) | auto (apply + show diff) | never
  backup: true                    # write a .bak before overwriting

workspace:
  ignore_dirs: [.git, .venv, node_modules, dist, build, ...]
  ignore_extensions: [.pyc, .png, .zip, ...]

search:
  max_results: 5                  # web-search results to consider
  timeout_seconds: 10             # per web request

knowledge:
  embedding_model: "nomic-embed-text"   # "" = use the chat model

mcp:
  servers: {}                     # see "MCP servers"

hooks: {}                         # see "Hooks"

devmode:                          # Developer Mode quality levers (see "How it gets quality")
  profile: balanced               # fast | balanced | thorough — one dial for the levers below
  judge_model: ""                 # optional stronger model for critic steps only ("" = main model)
  # Override an individual lever regardless of profile, e.g.:
  #   best_of: true               # (only fires when judge_model is set — see below)
  #   consistency_check: false
```

- A `.aicoderignore` file (gitignore syntax) in your workspace further excludes files from scanning.

---

## Safety & confirmation modes

You're always in the loop. Two independent gates:

**Shell** (`shell.confirmation`, or `--shell-mode`):
| Mode | Behaviour |
|---|---|
| `always` | Ask before every command *(default — safest)* |
| `smart` | Auto-run safe commands; ask for destructive ones (`rm`, `drop`, `-rf`, `--force`, …) |
| `never` | Auto-run everything |

**Files** (`files.confirmation`):
| Mode | Behaviour |
|---|---|
| `always` | Show the diff and ask before each write |
| `auto` | Show the diff and apply automatically *(default)* |
| `never` | Write immediately, no preview |

Overwritten files are backed up as `*.bak` (when `files.backup: true`). All file operations are sandboxed to the workspace — path traversal is rejected.

---

## Where your data lives

```
~/.aicoder/
├── config.yaml                  # your settings
├── AICODER.md                   # (optional) global project instructions
├── rag/chroma/                  # cached web/document knowledge (vector store)
└── memory/<project_id>/
    ├── project_memory.json      # durable facts the agent remembers
    └── plan.json                # in-progress task plan (resumable)
```

Everything is per-project (keyed by workspace path) and stays on your machine. Code is read from / written to your workspace; nothing is sent anywhere unless you invoke web research.

---

## Project layout

```
ai-coder/
├── cli.py                  # entry point (the `aicoder` command)
├── core/
│   ├── config.py           # configuration (~/.aicoder/config.yaml)
│   ├── model.py            # ChatOllama factory + native tool binding + tool-call recovery
│   ├── context.py          # workspace scanner / repo overview
│   ├── project.py          # test- & lint-command detection
│   └── code_index.py       # symbol index (find_symbol)
├── agent/
│   ├── loop.py             # the agentic loop, REPL, slash commands, history compaction
│   ├── tools.py            # the built-in tools
│   ├── planner.py          # decompose + run resumable task plans
│   ├── prompts.py          # system prompt
│   ├── mcp_client.py       # MCP client (external tool servers)
│   └── hooks.py            # lifecycle hooks
├── devmode/                # Developer Mode: role-driven SDLC design → build
│   ├── phases.py           # the 14 phases + quality-lever config (must-cover/decompose/best-of)
│   ├── session.py          # engine: discuss → summarize → consistency → resolve
│   ├── build.py            # file-plan + per-file generation with self-review
│   └── resync.py           # propagate a changed decision into the code
├── rag/
│   ├── store.py            # ChromaDB vector store with chunking
│   ├── ingest.py           # PDF/docx/md/html document loaders
│   └── research.py         # web research → knowledge-base pipeline
├── memory/
│   └── project.py          # persistent per-project memory
├── tools/
│   ├── file_tools.py       # file read/write/diff/backup/grep, path safety
│   ├── shell_tools.py      # shell execution with confirmation modes
│   └── web_tools.py        # DuckDuckGo search + URL fetch + HTML parsing
└── tests/                  # unit + agent-loop integration tests
```

See [`docs/architecture.md`](docs/architecture.md) and [`docs/features.md`](docs/features.md) for deeper detail.

---

## Architecture

- **Single agentic loop.** One assistant that plans, edits, runs, and verifies any repo — via native tool calling, with a fallback that recovers tool calls a local model emits as text.
- **RAG + memory, not weight training.** Staying current and "learning" is done by retrieving cached web/document knowledge and durable project facts at query time; the model's weights are never modified.
- **Sync core, async edges.** The loop is synchronous and transparent; MCP sessions run on a background event loop bridged into it.
- **Strictly local.** Ollama for inference and embeddings; ChromaDB for the vector store; all data under `~/.aicoder/`.

---

## Limitations

Being honest about the tradeoffs:

- **Local-model intelligence.** A 7B-class local model is a strong *supervised* assistant, not an autonomous senior engineer. Expect to review its diffs; lean on the verify loop. Bigger models help.
- **Context window.** Bounded by your hardware (default 16k tokens). History is compacted to fit, but very large tasks still benefit from `plan`.
- **No image input.** Local code models are text-only.
- **Tool-calling reliability** varies by model — `qwen2.5-coder:7b`+ is recommended; `--selftest` checks it.
- **MCP** is stdio-only for now; **Windows** support is best-effort (the common paths are handled).

---

## Troubleshooting

- **"Cannot reach Ollama" / model warnings** — make sure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull qwen2.5-coder:7b`).
- **`--selftest` says the model can't call tools** — switch to a stronger model (`aicoder --model qwen2.5-coder:7b`).
- **Web research / `read_document` says it couldn't ingest** — pull an embedding model (`ollama pull nomic-embed-text`).
- **MCP servers don't load** — install the extra (`pip install "ai-coder[mcp]"`) and check the server `command`/`args` in your config.
- **Edits get declined / the agent loops** — small models sometimes struggle; rephrase, or switch to a larger model.
- **See your settings** — `aicoder --config`.

---

## Development

```bash
pip install -e ".[dev]"
pytest -q                 # run the test suite

python -m build           # build sdist + wheel (needs `build`)
```

---

## License

MIT — see [LICENSE](LICENSE). Changelog: [CHANGELOG.md](CHANGELOG.md).

Contributions welcome — please open an issue first for significant changes.
