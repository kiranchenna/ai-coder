<p align="center">
  <img src="assets/icon.png" alt="AICoder logo" width="128" height="128">
</p>

<h1 align="center">AICoder ✨</h1>

> A local, offline **agentic coding assistant** — it plans, reads and edits real code, runs commands and tests, researches the web, and remembers your project — all running on your own machine via [LM Studio](https://lmstudio.ai/) by default. No API keys required, nothing sent anywhere unless you invoke web research or explicitly point it at a different server/API (see [Using a different backend](#using-a-different-backend)).

---

## Table of contents

- [What is AICoder?](#what-is-aicoder)
- [Key features](#key-features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Choosing a model](#choosing-a-model)
- [Quick start](#quick-start)
- [Usage examples](#usage-examples)
- [Command-line usage](#command-line-usage)
- [Talking to the agent](#talking-to-the-agent)
- [In-session commands](#in-session-commands)
- [Multi-step builds (`/plan` and `/resume`)](#multi-step-builds-plan-and-resume)
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

- **Python 3.11+**
- **[LM Studio](https://lmstudio.ai/)** installed, with its local server running (Developer tab → Start Server) — `aicoder` warns you at startup if it can't reach it
- A downloaded chat model (and, for web/document RAG, an embedding model)

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

### Download the models

In LM Studio's own model search, grab:

```
lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF   # the agent driver (or an MLX build on Apple Silicon)
nomic-ai/nomic-embed-text-v1.5-GGUF                 # embeddings (web research + documents)
```

Then load the chat model (`lms load <model-id>`, or from LM Studio's UI) — `aicoder`'s `/model` command lists whatever's downloaded and switches between them.

### Verify

```bash
aicoder --selftest    # confirms the configured model can call tools
```

---

## Choosing a model

The default is **`qwen2.5-coder-7b-instruct`** — a good balance of code quality and tool-calling reliability. Pick based on your hardware (the model and its context share memory):

| RAM / VRAM | Suggested models | Notes |
|---|---|---|
| 8 GB | Qwen2.5-Coder-3B-Instruct, Qwen3-4B, Granite-4.0-Micro | fast, weaker at multi-step work |
| 16 GB | **Qwen2.5-Coder-7B-Instruct** (default), Qwen2.5-Coder-14B-Instruct, DeepSeek-Coder-V2-Lite-16B, gpt-oss-20b | the sweet spot |
| 24 GB+ | Qwen3-Coder-30B, Devstral-24B, Codestral, Qwen2.5-Coder-32B-Instruct | strongest, needs more memory |

On Apple Silicon, prefer an **MLX** build over GGUF where available — LM Studio's native format there, and measurably faster (a live benchmark on an M1 Pro showed ~49% higher throughput for the same model/quant vs. GGUF).

Switch models with the in-session `/model` command — type `/model` alone for an interactive picker listing every model already downloaded in LM Studio (current one marked); pick one and it switches (unloading the old one, loading the new). Or `/model <name>` to switch straight to an id you already know (see `lms ls` for exact ids). Either way it's **saved as your default for new sessions**, not just this one. Grabbing a *new* model is a manual step in LM Studio itself — its own CLI download flow proved too unreliable to automate. `aicoder --model <name>` overrides the model for one run only (without changing the saved default), and you can also edit `~/.aicoder/config.yaml` directly. Run `aicoder --selftest` after switching to confirm the model supports tool calling.

> **Embeddings** (`text-embedding-nomic-embed-text-v1.5` by default) are only needed for web research and document ingestion.

### Using a different backend

LM Studio is the default and needs no config changes beyond what's above, but AICoder isn't tied to it — under the hood it always talks the OpenAI chat-completions protocol, so it works with **any server or API that speaks it**:

- **A local runtime sized to your hardware** — a heavy GPU box might run **vLLM** for higher throughput, a lighter machine might run **llama.cpp server**; **text-generation-webui** and **LocalAI** work too.
- **A hosted API, with your own key** — OpenAI, OpenRouter, Groq, Together, or anything else that speaks the same protocol.

```yaml
model:
  provider: openai_compatible
  name: "your-model-id"                    # whatever your server/API expects
  base_url: "http://localhost:8080/v1"     # or a hosted API's endpoint
  api_key: ""                              # blank for local servers that don't check it
```

The rich `/model` picker (list/switch what's downloaded) is LM Studio-specific — detected by `base_url` matching its default (`http://localhost:1234/v1`). Point `base_url` elsewhere and `/model` falls back to showing your current model/endpoint, with `/model <name>` still working to switch the model id on the same endpoint.

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
aicoder --model qwen2.5-coder-14b-instruct
```

Conversations don't survive quitting `aicoder` by default — pick up where you
left off with `aicoder --continue` (or `-c`), which resumes the most recent
conversation for the current workspace instead of starting fresh.

Every session is saved (one file per session, nothing overwritten) so you can
go back and see what actually happened — `/history` lists past sessions for
this workspace (date, first prompt, files touched); `/history <n>` shows one
in full detail: every prompt, every tool call made in answering it, the real
diff for any file that changed, and the final answer.

**On a real terminal, `aicoder` runs a full-screen chat UI** — a scrolling
conversation with a pinned input box at the bottom, a "/" autocomplete
dropdown for slash commands, arrow-key menus (`/model`, confirmations), and a
live "thinking" indicator you can interrupt with Esc — the same overall shape
as Claude Code's interface. It also runs in the alternate screen buffer (the
same mode `vim`/`less`/`htop` use), so when you exit (`/exit`, Ctrl-D,
Ctrl-C), your terminal is restored exactly as it was before, with no session
trace left in your scrollback. If you want to keep a
copy of what happened, run `/export` before exiting.

Piped/redirected/scripted usage (e.g. `echo "..." | aicoder`, or anything run
outside a real terminal) automatically falls back to a plain print-and-scroll
REPL instead — the full-screen UI needs a real terminal to attach to.

**Paste a screenshot with Ctrl+V, just like Claude Code.** Claude's own model
is natively multimodal, but LM Studio's local model ecosystem splits vision
and coding into separate model families — the coding models are text-only.
So this is a two-model handoff: a vision-capable model (nothing configured by
default — download one and set `vision.model`, or use `/vision model`) looks
at the image and describes it, then your regular coding model acts on that
description with its normal tools. Works the same way via `/vision <path>` if
you'd rather point at a file than paste. You can paste more than one image
before sending your message — all of them are described together in one go.
Once the vision model has seen an image, a follow-up question with no path
(`/vision what about the corner?`) asks about that same image again, so you
don't have to re-attach it each time.

Pick a different vision model with `/vision model` — the same arrow-key
picker as `/model`, listing vision-capable models already downloaded in LM
Studio (current one marked), plus "Other…" for anything else — or switch
straight to one with `/vision model <name>`.

Clipboard image paste needs no extra setup on macOS or Windows. **On Linux**,
it needs `wl-paste` (Wayland — from the `wl-clipboard` package) or `xclip`
(X11) on your `PATH`; without either, Ctrl+V shows a one-time warning and
falls back to plain text paste. `/vision <path>` always works regardless of
platform, since it doesn't touch the clipboard at all.

---

## Usage examples

Realistic sessions — what actually happens when you type a request. `→` is a
tool call the agent made on its own; everything else is what you'd see on
screen.

### Fix a bug

```
myproject> the /login endpoint returns 500 when the password field is empty

→ search_code(query=/login)
→ read_file(path=api/auth.py)
→ edit_file(path=api/auth.py, old_string=..., new_string=...)
  ~ MODIFIED api/auth.py
  --- a/api/auth.py
  +++ b/api/auth.py
  @@ -12,6 +12,8 @@
   def login(data: LoginRequest):
  +    if not data.password:
  +        raise HTTPException(400, "password is required")
       user = get_user(data.email)
→ run_tests()
  ✓ 14 passed

Found it — login() never checked for an empty password before hashing it,
which threw and bubbled up as a 500. Added a 400 with a clear message and
confirmed the existing test suite still passes.
⏱ 18.2s
```

### Add a feature, with a test

```
myproject> add a /health endpoint that returns {"status": "ok"}, with a test for it

→ list_files(path=api/)
→ read_file(path=api/routes.py)
→ edit_file(path=api/routes.py, old_string=..., new_string=...)
  ~ MODIFIED api/routes.py
→ write_file(path=tests/test_health.py, content=...)
  + CREATED tests/test_health.py
→ run_tests()
  ✓ 15 passed

Added GET /health returning {"status": "ok"} and a test asserting the
200 + JSON body. All tests pass.
⏱ 12.6s
```

### Refactor and verify

```
myproject> refactor utils.py to use pathlib instead of os.path

→ read_file(path=utils.py)
→ edit_file(path=utils.py, old_string=..., new_string=...)   [repeats per usage]
→ run_tests()
  ✓ 15 passed
→ run_checks()
  ✓ ruff: no issues

Replaced 4 os.path calls with their pathlib.Path equivalents
(os.path.join → Path(...) / ..., os.path.exists → Path(...).exists()).
Tests and lint both pass.
⏱ 21.4s
```

### Understand a codebase

```
myproject> explain how authentication works in this repo

→ find_files(name_pattern=*auth*)
→ read_file(path=api/auth.py)
→ search_code(query=verify_token)

Authentication is JWT-based: POST /login issues a token signed with
JWT_SECRET (api/auth.py:8), and verify_token() (api/auth.py:34) is used
as a FastAPI dependency on every protected route. Tokens expire after 24h
(settings.TOKEN_TTL). There's no refresh-token flow yet.
⏱ 8.1s
```

### Look something up and act on it

```
myproject> what's the latest stable version of httpx, and pin it in requirements.txt

→ research(query=httpx latest stable version)
→ read_file(path=requirements.txt)
→ edit_file(path=requirements.txt, old_string=httpx, new_string=httpx==0.28.1)
  ~ MODIFIED requirements.txt

httpx 0.28.1 is the latest stable release (per the cached search). Pinned
it in requirements.txt.
⏱ 9.8s
```

### Review and commit

```
myproject> /diff
[shows the accumulated diff for this session]

myproject> commit these changes with a clear message

→ git_status()
→ git_diff()
→ git_commit(message="Add /health endpoint and fix empty-password 500 on /login")
  ✓ committed a1b2c3d

Committed both changes together — same session, related code.
⏱ 4.2s
```

### Debug from a screenshot

```
[Ctrl+V — pastes a screenshot of a broken layout]
myproject> what's wrong here, and can you fix it?

→ [vision model describes the image: "a product card with the price
   overlapping the title, image is 3x larger than the surrounding cards"]
→ search_code(query=ProductCard)
→ read_file(path=components/ProductCard.tsx)
→ edit_file(path=components/ProductCard.tsx, old_string=..., new_string=...)
  ~ MODIFIED components/ProductCard.tsx

The image had no max-width, pushing the price out of the card. Constrained
it to the card width and added the missing margin under the title.
⏱ 26.7s
```

### A quick one-liner for the day-to-day

Most requests don't need a walkthrough — just describe the task:

```
why is test_login failing? find and fix it
add input validation to the create_user endpoint
what does this regex on line 42 of parser.py actually match?
bump the Node version in the Dockerfile to 22 and update the CI workflow
```

### Bigger jobs

A single message is enough for most day-to-day work, but some jobs are too
big for one turn:

- **[`/plan <goal>`](#multi-step-builds-plan-and-resume)** — decomposes a goal
  into an ordered, resumable task list and builds it, e.g.
  `/plan add JWT refresh tokens with rotation and revocation`.
- **[`/develop <idea>`](#developer-mode)** — designs a whole application
  phase-by-phase (product, architecture, schema, API, UI, …) with you in
  control of every decision, *then* builds it, e.g.
  `/develop a multi-tenant invoicing SaaS with Postgres and a React UI`.
- **[Working from a document](#working-from-documents)** — ground either of
  the above in a spec you already have:
  `read the PRD at docs/spec.pdf and scaffold the service it describes`.
- **[`/knowledge learn <topic>`](#web-research--the-knowledge-base-rag)** —
  cache current info before a task that needs it, e.g.
  `/knowledge learn "FastAPI 0.118 lifespan events"` before asking the agent
  to migrate an app off the old `@app.on_event` startup hooks.

---

## Command-line usage

```
aicoder [options]
```

| Flag | Description |
|---|---|
| `--workspace`, `-w PATH` | Project directory to work in (default: current directory) |
| `--model`, `-m MODEL` | Model id to use this session (overrides config) |
| `--shell-mode {always,never,smart}` | Shell confirmation mode for this session |
| `--selftest` | Verify the model supports tool calling, then exit |
| `--config` | Show the config file path and current settings, then exit |
| `--version` | Print the version |

If LM Studio isn't reachable or the model isn't loaded, AICoder warns you at startup.

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

A few literal commands are handled by the REPL; everything else is a task for the agent. If you're coming from Claude Code, most of its slash commands have a direct equivalent here — `/init`, `/status`, `/context`, `/compact`, `/permissions`, `/model`, `/mcp`, `/review`, `/bug` all work the same way. (Some don't apply to a local, single-agent, no-accounts tool — `/login`, `/cost`, `/agents`, `/ide` — so they're not here.)

| Command | Description |
|---|---|
| `/develop [--fast] <idea>` | Developer Mode: role-driven SDLC design → build (`--fast` = no back-and-forth) |
| `/dev [status\|build\|revisit <phase>\|resolve]` | Resume Developer Mode, or run a sub-step |
| `/plan <goal>` | Decompose a goal into an ordered, resumable task list and build it |
| `/resume` | Continue an in-progress plan |
| `/init` | Analyze the codebase and write/update `AICODER.md` — takes effect immediately in this session |
| `/model [name]` | With LM Studio: pick a model interactively (lists downloaded models, current marked), or switch straight to `<name>` — either way, saved as your default. Pointed at a different server: shows your current model/endpoint; `/model <name>` still switches |
| `/status` | Show the workspace, model, provider, and Developer Mode profile |
| `/context` | Show conversation size vs. the auto-compaction budget |
| `/compact` | Summarize older turns now — the same compaction that runs automatically, on demand |
| `/permissions [shell\|files <mode>]` | View or change the shell/file confirmation modes without restarting |
| `/review` | Ask the agent to review the current git diff for bugs and cleanup opportunities |
| `/tools` | List all available tools (built-in + MCP) |
| `/mcp` | List connected MCP servers and their tools |
| `/hooks` | List configured lifecycle hooks |
| `/diff` | Show the git diff of changes so far |
| `/memory` | Show what's remembered about this project |
| `/knowledge [learn <topic\|URL> \| clear \| clear all]` | Manage the RAG knowledge base (see below) |
| `/export [file]` | Save this conversation to a markdown file (default: a timestamped name) |
| `/doctor` | Diagnose the model/tool-calling setup without restarting (same check as `--selftest`) |
| `/bug` | Where and what to include when reporting a problem |
| `/clear` | Forget the current conversation (keeps saved memory) |
| `/help` | List commands |
| `/exit` | Leave the session |

---

## Multi-step builds (`/plan` and `/resume`)

For a large goal, use `/plan`. AICoder decomposes it into an ordered task list, then executes each task — reading/writing files and verifying as it goes — pausing for your confirmation between tasks:

```
my-project> /plan add JWT refresh tokens with rotation and revocation

Decomposed into 5 tasks:
  1. Add a refresh_tokens table (id, user_id, token_hash, expires_at, revoked_at)
  2. Issue a refresh token alongside the access token on login
  3. Add POST /auth/refresh — validate + rotate the refresh token
  4. Add POST /auth/revoke — mark a refresh token revoked
  5. Add tests for issuance, rotation, revocation, and reuse-detection

Starting task 1/5: Add a refresh_tokens table
→ read_file(path=models.py)
→ edit_file(path=models.py, old_string=..., new_string=...)
  ~ MODIFIED models.py
→ run_shell(command=alembic revision --autogenerate -m "add refresh_tokens")
→ run_tests()
  ✓ 15 passed
✓ Task 1/5 done. Continue to task 2/5? [Y/n]
```

It's **resumable**: quit anytime, and next session type `/resume` to continue from the first unfinished task:

```
my-project> /resume

Resuming plan: add JWT refresh tokens with rotation and revocation (1/5 done)
Starting task 2/5: Issue a refresh token alongside the access token on login
→ read_file(path=api/auth.py)
→ edit_file(path=api/auth.py, old_string=..., new_string=...)
  ~ MODIFIED api/auth.py
→ run_tests()
  ✓ 16 passed
✓ Task 2/5 done. Continue to task 3/5? [Y/n]
```

Plan state is saved under `~/.aicoder/memory/<project>/plan.json`, so this works even after closing the terminal or switching machines (same `~/.aicoder` directory).

---

## Developer Mode

For building real applications with full control, **Developer Mode** runs a **role-driven SDLC** — it discusses each stage with you (as a different expert role), captures every decision as an editable file, and only then builds. You stay in control of the tech stack, schema, architecture, flows, screens, and the exact code structure.

```
my-project> /develop a multi-tenant invoicing SaaS with Postgres and a React UI

━━ Phase 1/14: Product Vision (Product Manager) ━━
As your Product Manager, let's nail down the vision first. A few questions:
 1. Who's the primary user — the tenant's own accountant, or their customers
    paying invoices?
 2. Is "multi-tenant" full data isolation per company, or a shared workspace?
 3. What's the core loop that makes this worth paying for vs. spreadsheets?

my-project> tenant = a small business owner; full data isolation per company;
core loop is create invoice → send → get paid → auto-reconcile

Draft decision:
  Primary user: small business owners managing their own invoicing.
  Isolation: full per-tenant data isolation (row-level, tenant_id on every table).
  Core loop: create → send → track payment → auto-reconcile against bank feed.
  ...

my-project> done
✓ Phase 1/14 captured → docs/dev/01_product_vision.md

━━ Phase 2/14: Market & Competitors (Market Analyst) ━━
→ research(query=invoicing SaaS competitors small business 2026)
Based on current players (FreshBooks, Wave, Invoice Ninja)... your
differentiator is the auto-reconcile loop — most competitors require manual
matching. Agree, or want to weigh a different angle?

my-project> agree, that's the wedge
✓ Phase 2/14 captured → docs/dev/02_market.md

... (12 more phases — architecture, schema, API, UI, security, testing, ...)
```

Once every phase is captured (or design review flags something you fix via `/dev revisit`), `/dev build` turns it into code:

```
my-project> /dev build

Proposed file plan (42 files) — edit docs/dev/build_plan.json to change paths/
order/naming, or press Enter to build as-is:
  backend/app/models/{tenant,invoice,payment}.py
  backend/app/api/routes/{invoices,auth,webhooks}.py
  frontend/src/pages/{InvoiceList,InvoiceDetail}.tsx
  ...

Building... [1/42] backend/app/models/tenant.py
  ✓ self-review: no issues
[2/42] backend/app/models/invoice.py
  ✓ self-review: fixed 1 issue (missing tenant_id foreign key)
  ...
[42/42] frontend/src/pages/InvoiceList.tsx  ✓

Verifying: compile check → tests → fix loop
  ✓ backend compiles, 28/28 tests pass
  ✓ frontend builds, 6/6 tests pass

Build complete. docs/dev/build_manifest.json maps each file to the phases
that shaped it.
```

If `--fast` is passed instead (`/develop --fast <idea>`), every phase is decided in one pass with no back-and-forth — good for a quick throwaway scaffold, at the cost of your input into each decision.

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

In each design phase, type `done` to capture the decision, `skip` to skip, `revise` to restart, or `pause` to stop and resume later. The final **Design Review** doesn't propose a decision — it critiques the others (consistency, gaps, security/scale risks) and points you to `/dev revisit <phase>` to fix anything.

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
/develop <idea>        # start (or resume) the design
/develop --fast <idea> # design the whole thing in one pass (roles decide; no back-and-forth)
/dev                   # resume the design
/dev status            # show phase progress
/dev build             # turn the design into code — proposes a file plan you can
                       #   edit (build_plan.json), then generates file-by-file and verifies
/dev revisit <phase>   # re-open a decision; if it changes, auto-resync the code to match
/dev resolve           # cross-phase review → fix the design contradictions → resync code
```

- **`/dev build`** proposes the folder/file structure from the design + your conventions. **Edit `docs/dev/build_plan.json`** (paths, order, naming) and re-run to use your exact structure. It then generates each file — grounded in the spec + `AICODER.md`, shown as a diff, **resumable per file** — and closes the loop: a **compile check → tests → agentic-fix loop** (up to 3 rounds) gets the code actually running, even when the project lives in a subdirectory. It also writes `docs/dev/build_manifest.json` mapping each file to the design phases it implements.
- **`/dev revisit <phase>`** lets you change any decision later. If the decision changed and code was built, AICoder **auto-resyncs**: it diffs old→new and runs an agentic task to propagate the change through the code, then verifies.
- **`/dev resolve`** reviews every phase together, lists the cross-phase contradictions (e.g. a schema that stores plaintext despite an end-to-end-encryption promise, or an auth mechanism that disagrees with the security phase), and for each one you accept it **rewrites the offending phase's decision and auto-resyncs the code**. It catches blunt contradictions reliably; subtle ones a small local model can't reason through may still need a manual `/dev revisit`.

### Greenfield and existing repos

- **Greenfield:** you specify the conventions in the Conventions phase.
- **Existing repo (brownfield):** every phase is grounded in your codebase, and the Conventions phase **infers your current conventions** from the code for you to confirm/adjust — so generated code matches your existing style.

### How it gets quality from a small model

A local 7B model doesn't know which parts of a domain are hard, and it writes a weak first draft. Developer Mode compensates with **engineering, not a bigger model**. The levers are bundled into a single `devmode.profile` dial — **`fast`** (reflect only), **`balanced`** (the default: reflect + consistency + build-review), or **`thorough`** (everything, including best-of-N). You can still override any individual lever in config.

> **Why these defaults?** A lever ablation (see [`evals/`](evals/)) on the security-design phase found that **`reflect` carries essentially all of the quality gain** (+2.0/10, 70%→100% checklist coverage, for ~20% added time), while **`best_of` only pays with a stronger judge** — with a same-strength self-judge it added latency without quality. So `balanced` keeps reflect and drops best-of, and **`best_of` is gated on `judge_model`**: it only fires when you've configured a stronger critic model to rank the candidates. Two more evals back the rest of `balanced`: `consistency_check` measured **100% precision / 60% recall** (caught every *blatant* cross-phase contradiction with zero false alarms, missed the *subtle* ones — so it stays as cheap insurance while subtle conflicts still want a manual `/dev revisit`), and `build_review` measured a **100% placeholder-removal rate** with clean drafts left intact. Run them yourself: `python -m evals.run_eval`, `run_consistency_eval`, `run_build_review_eval`.

The levers, each independently toggleable:

- **Must-cover checklists** — each phase carries a senior checklist the model is *forced* to address (e.g. Security must name the actual E2E protocol and per-device keys; Architecture must name the real-time backbone), so it can't skip the defining decisions.
- **Reflection** (`reflect`) — every decision is drafted, then critiqued and revised in a second pass; a small model improves a concrete draft far better than it writes a perfect one first try.
- **Decomposition** — the heavy phases (data model, API, architecture) are designed **one unit at a time** (list → detail each entity/endpoint/component → assemble), which a small model handles far better than one giant answer.
- **Targeted research** — research phases derive 2–3 *specific* web queries (current versions, protocols, pitfalls) instead of one generic search, putting real current facts in context.
- **Best-of-N** (`best_of`) — for the critical phases (requirements, security) it generates several candidate decisions from different angles and a judge keeps the strongest.
- **Cross-phase consistency check** (`consistency_check`) — after each phase, its decision is checked against the earlier ones and contradictions are flagged (and logged to `docs/dev/consistency_notes.md`).
- **Build self-review** (`build_review`) — every generated file is critiqued for bugs, placeholders, and convention misses, then fixed, before it's written.
- **Build verify→fix loop** — after generation, a compile check → tests → agentic-fix loop (≤3 rounds) gets the code actually running, not just plausible-looking.
- **`/dev resolve`** — turns those contradictions into fixes: it rewrites the offending phase and auto-resyncs the code.
- **Hybrid judging** (`judge_model`, opt-in) — point the *critic* steps (best-of judging, consistency, review) at a stronger model while generation stays local — the cheapest way to push past what a 7B can reason through.

> Reality check: these levers measurably lift output — the `evals/` harness shows `reflect` taking a security-design phase from 7.5 to 9.5/10, `consistency_check` catching every blatant cross-phase contradiction (100% precision / 60% recall), and `build_review` removing 100% of planted placeholders. But a local 7B is still a strong *assistant*, not an autonomous senior engineer — review the generated code, lean on the verify step, and use `/dev resolve` / `/dev revisit` to correct decisions. Subtle contradictions a 7B can't reason through may still slip past. The design/decision artifacts are valuable on their own, regardless of model strength.

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
- Content is chunked and embedded (via your configured embedding model, served by LM Studio), with a relevance cutoff so unrelated queries return nothing rather than noise.
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
  provider: openai_compatible     # always openai_compatible; kept as an explicit field
  name: qwen2.5-coder-7b-instruct # any model downloaded in LM Studio (or another server's model id)
  base_url: http://localhost:1234/v1   # LM Studio's default; point elsewhere for a different server
  api_key: ""                     # blank for local servers with no auth
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
  embedding_model: "text-embedding-nomic-embed-text-v1.5"   # "" = use the chat model

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
│   ├── model.py            # OpenAI-compatible chat model factory + native tool binding + tool-call recovery
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
├── evals/                  # Developer Mode quality-lever measurement harness
└── tests/                  # unit + agent-loop integration tests
```

See [`docs/features.md`](docs/features.md) (how it works), [`docs/architecture.md`](docs/architecture.md) (how it's built), [`docs/support.md`](docs/support.md) (FAQ & troubleshooting), and [`evals/README.md`](evals/README.md) (the quality-lever measurements) for deeper detail.

---

## Architecture

- **Single agentic loop.** One assistant that plans, edits, runs, and verifies any repo — via native tool calling, with a fallback that recovers tool calls a local model emits as text.
- **RAG + memory, not weight training.** Staying current and "learning" is done by retrieving cached web/document knowledge and durable project facts at query time; the model's weights are never modified.
- **Sync core, async edges.** The loop is synchronous and transparent; MCP sessions run on a background event loop bridged into it.
- **Strictly local.** LM Studio for inference and embeddings; ChromaDB for the vector store; all data under `~/.aicoder/`.

---

## Limitations

Being honest about the tradeoffs:

- **Local-model intelligence.** A 7B-class local model is a strong *supervised* assistant, not an autonomous senior engineer. Expect to review its diffs; lean on the verify loop. Bigger models help.
- **Context window.** Bounded by your hardware (default 16k tokens). History is compacted to fit, but very large tasks still benefit from `plan`.
- **No image input.** Local code models are text-only.
- **Tool-calling reliability** varies by model — `qwen2.5-coder-7b-instruct`+ is recommended; `--selftest` checks it.
- **MCP** is stdio-only for now; **Windows** support is best-effort (the common paths are handled).

---

## Troubleshooting

- **"Couldn't reach LM Studio"** — make sure LM Studio's local server is running (Developer tab → Start Server), and that a model is loaded (`lms load <model-id>`, or from the UI).
- **`--selftest` says the model can't call tools** — switch to a stronger model (`aicoder --model qwen2.5-coder-7b-instruct`).
- **Web research / `read_document` says it couldn't ingest** — download an embedding model in LM Studio (e.g. `nomic-ai/nomic-embed-text-v1.5-GGUF`) and set `knowledge.embedding_model` if it's not the default.
- **MCP servers don't load** — install the extra (`pip install "ai-coder[mcp]"`) and check the server `command`/`args` in your config.
- **"langchain-openai isn't installed"** — `pip install langchain-openai` (this is a core dependency, so it should already be present — a missing install usually means a broken environment).
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

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md); please open an issue first for significant changes.
