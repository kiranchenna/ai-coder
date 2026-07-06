# AICoder — Architecture

**Version:** 3.0.0 | **Language:** Python 3.11+ | **Entry point:** `cli.py` → `aicoder` CLI command

AICoder v3 is a local, offline **agentic** coding assistant. It uses Ollama
(local LLM, no cloud, no API keys) to drive a single tool-calling loop that
works on your real repository: reading and editing code, running commands and
tests, researching the web, and remembering decisions across sessions.

---

## Directory structure

```
ai-coder/
├── cli.py                      # Entry point (args, config, launch the agent)
├── aicoder_cli/__init__.py     # Installable package wrapper (sets up sys.path)
│
├── core/                       # Shared core
│   ├── config.py               # Configuration (~/.aicoder/config.yaml)
│   ├── context.py              # Workspace scanner / repo overview
│   ├── code_index.py           # ctags-style symbol index (find_symbol)
│   ├── model.py                # ChatOllama factory + tool-call recovery + selftest
│   ├── model_catalog.py        # Curated /model recommendations by tier (fast/balanced/powerful)
│   ├── ollama_install.py       # Detect + offer to install Ollama itself (cli.py's pre-flight check)
│   └── project.py              # Test- & lint-command detection
│
├── agent/                      # The agentic core
│   ├── loop.py                 # Tool-calling loop, plain-REPL fallback, slash commands
│   ├── tui.py                  # Full-screen chat UI (Textual) — the default on a real terminal
│   ├── tools.py                # The 19 agent tools
│   ├── planner.py              # Decompose + run resumable task plans
│   ├── hooks.py                # Lifecycle hooks (Pre/PostToolUse, Stop)
│   ├── mcp_client.py           # MCP stdio client (optional)
│   └── prompts.py              # System prompt
│
├── devmode/                    # Developer Mode: role-driven SDLC design → build
│   ├── phases.py               # PhaseSpec data: 14 phases + must-cover/decompose/best-of
│   ├── session.py              # Engine: discuss → summarize → consistency → resolve
│   ├── build.py                # File-plan + per-file generation with self-review
│   └── resync.py               # Propagate a changed decision into the code
│
├── rag/                        # Retrieval-augmented knowledge
│   ├── store.py                # ChromaDB vector store with chunking + TTL
│   ├── research.py             # Web research → cached knowledge
│   └── ingest.py               # PDF/docx/md/html document loaders
│
├── memory/                     # Persistent per-project memory
│   └── project.py              # Durable facts (decisions/conventions/TODOs)
│
├── tools/                      # Helpers
│   ├── file_tools.py           # File read/write/diff/backup, path safety, grep
│   ├── shell_tools.py          # Shell execution with 3 confirmation modes
│   └── web_tools.py            # DuckDuckGo search + URL fetch + HTML parsing
│
├── evals/                      # Developer Mode measurement harness (see evals/README.md)
│   ├── run_eval.py             # Lever ablation: quality vs latency per phase
│   ├── run_consistency_eval.py # consistency_check detection precision/recall
│   ├── run_build_review_eval.py# build_review placeholder-removal rate
│   └── *_fixtures.py · rubric.py# fixed yardsticks + judge-model scoring
│
└── tests/                      # pytest unit tests
    ├── test_agent.py           # Agent logic (parsers, chunking, detection, memory)
    ├── test_devmode.py         # Developer Mode engine, build, resync, resolve, levers
    ├── test_loop.py            # Agent-loop integration
    ├── test_hooks.py · test_mcp.py · test_config.py · test_file_tools.py
```

Runtime data lives under `~/.aicoder/` (config, RAG store, per-project memory),
not in the repo.

---

## Tech stack

| Category | Library | Floor |
|---|---|---|
| LLM | langchain-ollama | 1.0+ |
| LLM core | langchain-core | 1.0+ |
| Terminal rendering | rich | 13.0+ |
| Full-screen chat UI | textual | 8.0+ |
| "/" autocomplete | textual-autocomplete | 4.0+ |
| Clipboard image paste | pillow | 10.0+ |
| Web search | ddgs (DuckDuckGo) | 9.0+ |
| HTTP | httpx | 0.28+ |
| HTML parsing | beautifulsoup4 | 4.12+ |
| Config | pyyaml | 6.0+ |
| Vector DB | chromadb | 1.0+ |
| PDF parsing | pypdf | 4.0+ |
| Word parsing | python-docx | 1.1+ |
| File patterns | pathspec | 0.12+ |
| Testing | pytest | 8.0+ |
| External | Ollama (local LLM server) | — |

---

## Data flow

```
cli.py
  ├── real terminal      →  agent.tui.run() [agent/tui.py] — full-screen chat UI
  └── piped/scripted     →  run_agent_repl() [agent/loop.py] — plain print-and-scroll fallback
        │  (both funnel into the same business logic below)
        ├── "/command"     →  _handle_command — all structured commands, incl.
        │                      /develop, /dev, /plan (→ agent/planner.py → task
        │                      list → AgentSession per task), /resume, /model, …
        └── plain English  →  AgentSession.send():
                                 model.stream(history + tools)  (live tokens)
                                   ├── native tool_calls → execute → feed back
                                   ├── else text tool-calls → recover → execute → feed back
                                   └── else → final answer
```

Tools touch the workspace (read/write code, run shell/tests) and `~/.aicoder/`
(RAG store, project memory, plan state).

---

## Configuration

- File: `~/.aicoder/config.yaml` (auto-created on first run)
- Default model: `qwen2.5-coder:7b` via Ollama at `http://localhost:11434`
- `model.provider` also accepts `openai_compatible` — any local server (llama.cpp
  server, vLLM, LM Studio, ...) or hosted API that speaks the OpenAI
  chat-completions protocol, sized to the user's own hardware/preference. See
  `core/model.py`'s `get_chat_model()`/`_build_openai_compatible()` for the
  branch, and the README's "Using a different backend" for setup.

```yaml
model:
  provider: ollama                # ollama | openai_compatible
  name: qwen2.5-coder:7b
  base_url: http://localhost:11434
  api_key: ""                     # openai_compatible only
  temperature: 0.3
  temperature_precise: 0.1
  context_length: 16384

shell:
  confirmation: always       # always | smart | never

files:
  confirmation: auto         # always | auto | never
  backup: true

knowledge:
  embedding_model: "nomic-embed-text"

vision:
  model: qwen2.5vl:7b            # used only when an image is attached (Ctrl+V
                                  # paste in the TUI, or /vision <path>) — the
                                  # two-model handoff; "" disables it

devmode:                       # Developer Mode quality levers (one dial)
  profile: balanced            # fast | balanced | thorough
  judge_model: ""              # optional stronger model for critic steps ("" = main)
  # individual levers can be overridden, e.g.  best_of: true  /  reflect: false
```

The `devmode` levers trade extra model calls for quality on a small local model.
They're bundled into a single **`profile`** dial (`Config.devmode_lever()` in
`core/config.py` resolves it; an explicit per-lever bool still overrides):

| profile | reflect | consistency_check | build_review | best_of |
|---|---|---|---|---|
| `fast` | ✓ | — | — | — |
| `balanced` (default) | ✓ | ✓ | ✓ | — |
| `thorough` | ✓ | ✓ | ✓ | ✓ (needs `judge_model`) |

`best_of` is **gated on `judge_model`**: best-of-N only fires when a stronger
critic model is configured to rank candidates (a same-strength self-judge added
latency without quality in the ablation — see `evals/`). The defaults are
evidence-based; see "Measuring quality" below and the README's "How it gets
quality from a small model".

---

## Key architectural decisions

1. **Single agentic loop** — one assistant that plans, edits, runs, and verifies,
   working on any repo. Replaces the old fixed 7-phase pipeline.
2. **Native tool calling, with text recovery** — tools are bound for native tool
   calling; when a local model emits calls as JSON text instead, they are parsed
   from the content and executed (`core/model.py`).
3. **RAG + memory, not weight training** — staying current and "learning" is done
   by retrieving cached web/document knowledge and durable project facts at query
   time; the model's weights are never modified.
4. **You-in-the-loop** — file writes and shell commands are gated by configurable
   confirmation modes; overwritten files are backed up.
5. **Sandboxed file ops** — all paths resolved against the workspace root;
   traversal is rejected.
6. **Resumable plans** — `/plan <goal>` saves task state after each step so a build
   resumes after a quit.
7. **Local by default, not local by force** — Ollama (no cloud, no API keys) is
   the default and needs no setup beyond it. `model.provider: openai_compatible`
   is an explicit opt-in for a different local runtime sized to the user's own
   hardware (vLLM, llama.cpp server, LM Studio, ...) or a hosted API with the
   user's own key — never on unless configured. RAG/embeddings stay on Ollama
   regardless of `model.provider`. All data still lives under `~/.aicoder/`.
8. **Evidence-based quality levers** — the Developer Mode levers are validated by
   a local eval harness (`evals/`), not assumed. The default `balanced` profile
   carries only the levers that measurably earned their latency; `best_of` is
   gated behind a stronger `judge_model` because the ablation showed it doesn't
   pay with a same-strength self-judge.
9. **Optional extras stay optional** — `langchain-openai` (for `openai_compatible`)
   and `mcp` are lazy-imported only when their feature is actually configured, so
   a plain-Ollama install never pulls them in; a missing package gives a clean,
   actionable error rather than a traceback.
10. **Full-screen, trace-free terminal session** — a real terminal gets
    `agent/tui.py`, a Textual chat UI in the alternate screen buffer (the same
    mechanism vim/htop/Claude Code use), restoring the terminal exactly as
    found on exit. Piped/redirected output falls back to `run_agent_repl`'s
    plain print-and-scroll REPL, unaffected. Rather than reimplement every
    slash-command handler and confirmation for the new UI, `agent/tui.py`
    swaps in adapters (a `console.print`-compatible `RichLog` writer, and a
    runtime monkeypatch of `rich.prompt.Confirm.ask`/`Prompt.ask` — the same
    technique this project's own tests already use) so all of that existing,
    already-tested business logic runs unchanged inside it.
11. **Vision as a two-model handoff, not a unified capability** — unlike a
    single multimodal model (e.g. Claude's), Ollama's local ecosystem splits
    vision and coding into separate model families, and the curated coding
    catalog is text-only. So an attached image (Ctrl+V paste, or
    `/vision <path>`) is described in text by a separate vision model
    (`vision.model`, built fresh, never persisted as the default driver, never
    bound to the coding tools) before the regular coding model ever sees the
    turn — the rest of the agentic loop is unaware anything visual was
    involved at all. Because it's a genuinely separate model family (not a
    "supports vision" flag on the coding catalog), `/vision model` gets its
    own picker sourced from a distinct `VISION_MODELS` catalog — but reuses
    `/model`'s picker machinery outright (`_build_model_menu`/
    `_run_model_picker`, extracted from `_handle_model_command` once a second
    caller needed it) rather than duplicating the grouping/highlighting/
    "Other…" logic a second time.
12. **Conversations don't persist unless asked** — `aicoder --continue`
    resumes the last saved conversation for the workspace; the default (no
    flag) is always a fresh session, matching how the whole test suite (and
    every other flag) already behaves. Persistence itself is unconditional
    and best-effort on every `send()` (a `finally` block, so an interrupted
    or failed turn still saves progress) — the opt-in is in *reading* it
    back, not in whether it's written.

---

## Measuring quality (`evals/`)

Developer Mode's value rests on the claim that engineering (reflection,
decomposition, checklists, review) lifts a small local model's output. That
claim is **measured**, not asserted, by three reproducible evals that drive the
live code paths (`python -m evals.<name>`):

| Eval | Lever measured | Method | Result (7B self-judge) |
|---|---|---|---|
| `run_eval` | `reflect`, `best_of` | grade one design phase vs an all-off baseline | reflect +2.0/10 (carries the gain); best_of −0.5 without a stronger judge |
| `run_consistency_eval` | `consistency_check` | precision/recall on labeled contradiction cases | 100% precision / 60% recall (catches blatant, misses subtle) |
| `run_build_review_eval` | `build_review` | placeholder-removal on drafts with planted issues | 100% fix rate, 100% preservation |

Scoring logic (`parse_score`, `compute_metrics`, `judge_case`) is pure and
unit-tested with the model call injectable, so the suite runs without Ollama.
See [`evals/README.md`](../evals/README.md). Caveat: numbers are small-n with a
same-model judge — deltas are decisive, absolute figures want a stronger
`--judge-model` and higher `--repeat`.

---

## LLM integration

Uses LangChain message types (`HumanMessage`, `AIMessage`, `SystemMessage`,
`ToolMessage`). `core/model.get_chat_model()` builds a `ChatOllama` (conversational
`temperature=0.3`, precise `0.1`) and binds the agent's tools. `core/model.selftest()`
checks tool calling (native or text-recovered) for the configured model.

---

## Data formats

**Project memory** (`~/.aicoder/memory/<project_id>/project_memory.json`):
```json
[
  {"id": "ab12cd34", "text": "Auth uses argon2", "category": "decision",
   "created_at": "2026-06-02T11:00:00"}
]
```

**Task plan** (`~/.aicoder/memory/<project_id>/plan.json`):
```json
{
  "goal": "build a todo API",
  "tasks": [
    {"id": 1, "title": "Create the model", "description": "...", "status": "done"},
    {"id": 2, "title": "Add endpoints", "description": "...", "status": "pending"}
  ]
}
```

**Conversation transcript** (`~/.aicoder/memory/<project_id>/conversation.json`,
for `aicoder --continue`) — everything after the system prompt, via
LangChain's `messages_to_dict`:
```json
[
  {"type": "human", "data": {"content": "fix the login bug", ...}},
  {"type": "ai", "data": {"content": "Found it — a stale session check.", "tool_calls": [], ...}}
]
```

**Developer Mode artifacts** (in the workspace under `docs/dev/`):
```
docs/dev/
├── state.json            # phase progress + cached consistency digests (resumable)
├── NN_<phase>.md         # each decision + its discussion transcript
├── consistency_notes.md  # cross-phase contradictions flagged during design
├── build_plan.json       # the file plan (user-editable: paths, order, naming)
└── build_manifest.json   # built file → the design phases it implements (provenance)
AICODER.md                # the coding conventions the build follows
```

**RAG store** — ChromaDB collection `aicoder_rag` at `~/.aicoder/rag/chroma/`;
content is chunked, embedded, and tagged with `source`, `title`, `fetched_at`,
`ttl_hours`, and `project`. Web research is stored globally (`project=""`) and
shared across projects; ingested documents are tagged with the project id, and
search returns global entries + the current project's own.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .

aicoder --selftest            # check tool calling
aicoder                       # start the agent

# models
ollama pull qwen2.5-coder:7b   # the agent driver
ollama pull nomic-embed-text   # embeddings (web research + documents)
```
