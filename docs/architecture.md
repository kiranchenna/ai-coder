# AICoder — Architecture

**Version:** 4.0.1 | **Language:** Python 3.11+ | **Entry point:** `cli.py` → `aicoder` CLI command | **PyPI package:** `local-aicoder` (command stays `aicoder` — see [Key architectural decisions #13](#key-architectural-decisions))

AICoder is a local, offline **agentic** coding assistant. It uses LM Studio
(local LLM, no cloud, no API keys) to drive a single tool-calling loop that
works on your real repository: reading and editing code, running commands and
tests, researching the web, and remembering decisions across sessions.

For a validated list of every dependency (purpose, license, why it was
chosen), see [dependencies.md](dependencies.md). This document covers the
system itself — structure, data flow, and the non-obvious decisions/gotchas
someone needs before making a manual change.

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
│   ├── model.py                # OpenAI-compatible chat model factory + LM Studio discovery + tool-call recovery + selftest
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

Quick reference — see **[dependencies.md](dependencies.md)** for the full,
license-verified breakdown (purpose, why each one was chosen, alternatives
considered where relevant).

| Category | Library | Floor |
|---|---|---|
| LLM | langchain-openai | 1.0+ |
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
| Testing | pytest, pytest-asyncio | 8.0+ / 1.0+ |
| Linting | ruff (dev extra) | 0.15+ |
| MCP client (optional) | mcp (`mcp` extra) | 1.0+ |
| External | LM Studio (local LLM server, via its `lms` CLI for lifecycle control — see [LM Studio lifecycle management](#lm-studio-lifecycle-management)) | — |

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
- Default model: `qwen2.5-coder-7b-instruct` via LM Studio at `http://localhost:1234/v1`
- `model.provider` is always `openai_compatible` under the hood — kept as an
  explicit field since the chat-model factory is still provider-typed. Any
  local server (llama.cpp server, vLLM, LM Studio, ...) or hosted API that
  speaks the OpenAI chat-completions protocol works by pointing `base_url`
  elsewhere; the rich `/model` picker (list/switch downloaded models) is
  LM-Studio-specific, detected by `base_url` matching its default. See
  `core/model.py`'s `get_chat_model()`/`_build_openai_compatible()`, and the
  README's "Using a different backend" for setup.
- `model.context_length` (default 131072 / 128k) drives both the LM Studio
  load parameters (`--context-length`) and the client-side history-
  compaction budget (`max(8_000, context_length * 2)` chars — see
  `AgentSession._history_budget`). Change it in-session with
  **`/context-length <n>`**, not by hand-editing the file while `aicoder` is
  running — the command also reloads the live model at the new value; see
  [LM Studio lifecycle management](#lm-studio-lifecycle-management).

```yaml
model:
  provider: openai_compatible     # always openai_compatible; kept as an explicit field
  name: qwen2.5-coder-7b-instruct
  base_url: http://localhost:1234/v1   # LM Studio's default
  api_key: ""                     # blank for local servers with no auth
  temperature: 0.3
  temperature_precise: 0.1
  context_length: 131072  # 128k — change with /context-length, not by hand

shell:
  confirmation: always       # always | smart | never

files:
  confirmation: auto         # always | auto | never
  backup: true

knowledge:
  embedding_model: "text-embedding-nomic-embed-text-v1.5"

vision:
  model: ""                      # "" disables vision by default — download a
                                  # vision model in LM Studio and set this (or
                                  # use /vision model) to enable the two-model
                                  # handoff (Ctrl+V paste in the TUI, or
                                  # /vision <path>)

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
7. **Local by default, not local by force** — LM Studio (no cloud, no API keys) is
   the default and needs no setup beyond it. Pointing `base_url` at a different
   local runtime (vLLM, llama.cpp server, ...) or a hosted API with the user's
   own key works the same way, since the underlying protocol is always OpenAI
   chat-completions — but only the rich `/model`/`/vision model` pickers are
   LM-Studio-specific. All data still lives under `~/.aicoder/`.
8. **Evidence-based quality levers** — the Developer Mode levers are validated by
   a local eval harness (`evals/`), not assumed. The default `balanced` profile
   carries only the levers that measurably earned their latency; `best_of` is
   gated behind a stronger `judge_model` because the ablation showed it doesn't
   pay with a same-strength self-judge.
9. **Optional extras stay optional** — `mcp` is lazy-imported only when a server
   is actually configured; a missing package gives a clean, actionable error
   rather than a traceback.
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
    single multimodal model (e.g. Claude's), LM Studio's local model ecosystem
    splits vision and coding into separate model families; the typical coding
    models are text-only. So an attached image (Ctrl+V paste, or
    `/vision <path>`) is described in text by a separate vision model
    (`vision.model`, built fresh, never persisted as the default driver, never
    bound to the coding tools, disabled by default until one is downloaded and
    configured) before the regular coding model ever sees the turn — the rest
    of the agentic loop is unaware anything visual was involved at all.
    Because it's a genuinely separate model family, `/vision model` gets its
    own picker (filtered to vision-capable downloaded models via `lms ls`'s
    `vision` flag) but reuses `/model`'s picker machinery outright
    (`_build_model_menu`/`_run_model_picker`, extracted from
    `_handle_model_command` once a second caller needed it) rather than
    duplicating the grouping/highlighting/"Other…" logic a second time.
12. **Every session is saved, but resuming is opt-in** — one JSON file per
    session (`sessions/<session_id>.json`), never overwritten across
    sessions; `aicoder --continue` resumes the most recent one for the
    workspace, but the default (no flag) is always a fresh session, matching
    how the whole test suite (and every other flag) already behaves.
    Persistence itself is unconditional and best-effort on every `send()`
    (a `finally` block, so an interrupted or failed turn still saves
    progress) — the opt-in is in *reading* it back, not in whether it's
    written. The same file backs two different needs without duplicating
    data: `raw_messages` for `--continue`'s exact restore, `turns` (with
    real per-action diffs, not just a status string) for `/history`'s
    human-readable browsing.
13. **PyPI package name ≠ command name.** The distribution is `local-aicoder`
    (`pip install local-aicoder`) but the command stays `aicoder`
    (`[project.scripts]` in `pyproject.toml`) — these are independent in
    Python packaging. Forced by PyPI's typosquat guard: `ai-coder` normalizes
    (hyphens/case ignored) to the same name as an unrelated pre-existing
    package, so the *distribution* name had to change; the actual command
    users type didn't need to, and keeping it unchanged avoided breaking
    every existing doc/muscle-memory reference for a purely cosmetic PyPI
    constraint.
14. **LM Studio is managed, not just talked to.** Beyond sending chat
    requests, `aicoder` actively manages the LM Studio server's lifecycle
    (auto-start, correct context length + idle-unload TTL on every load,
    auto-reload before a turn if idle-unloaded, explicit unload on a clean
    exit) — see [LM Studio lifecycle management](#lm-studio-lifecycle-management)
    below. This exists because LM Studio's own defaults silently diverge from
    `config.yaml` in ways that only surface as a cryptic runtime crash — not
    a hypothetical, each mechanism below was added after reproducing the
    failure live.
15. **The startup banner is full-width, content-height, two columns side by
    side** (`agent/loop.py`'s `_startup_banner`) — logo/identity on the left
    (~1/3, protected by a hard minimum width so the block-letter logo can
    never wrap), tip + tool highlights on the right. Getting this right
    surfaced a real Textual bug worth knowing before touching it again — see
    [Known gotchas](#known-gotchas-for-future-changes).
16. **Prompt history, like a shell.** The TUI's input (`agent/tui.py`'s
    `ChatInput`) tracks every submission; Up/Down recall it, matching Claude
    Code's own input box. Per-session only (not persisted to disk) — kept
    deliberately simple.
17. **`--continue` replays the conversation, not just its context.**
    `AgentSession.load_transcript()` alone restores `session.messages` (so
    the *model* has full context) but doesn't touch the visible chat log —
    confirmed live, that made `--continue` look exactly like starting fresh
    even though the context genuinely was there. Both front-ends now replay
    the actual prior turns via `_render_session_detail` (the same renderer
    `/history <n>` uses) right after a successful resume.
18. **`/develop`'s back-and-forth reads like a conversation, not a stream of
    popups.** Every `Prompt.ask`/`Confirm.ask` in `devmode/session.py` and
    `devmode/build.py` goes through `_ask`/`_confirm`, which — only in the
    TUI, only for devmode — route through the main chat input instead of a
    modal (`agent/tui.py`'s `ask_inline`/`ask_inline_confirm`), since a
    popup showing nothing but a bare phase id ("requirements") as its
    question was confirmed live to be genuinely confusing. Deliberately
    scoped to devmode: shell/file-write confirmations and the `/model`
    picker keep their modals unchanged.
19. **An empty model turn is retried, not accepted as "done".**
    `AgentSession._run_steps()` treats a turn with no tool call *and* no
    text as recoverable, not final — some local "thinking" models can spend
    an entire turn on internal reasoning LangChain's `ChatOpenAI` can't
    capture (it doesn't extract non-standard `reasoning_content` deltas —
    see its own docstring) and then simply stop. It nudges the model to
    commit to an answer or a tool call, up to `MAX_EMPTY_RESPONSE_RETRIES`
    (2) times, before giving up with an explanation instead of a silent
    `(no further response)`.

---

## LM Studio lifecycle management

Four independent mechanisms in `core/model.py`/`agent/loop.py`, each added
after reproducing a specific failure live against a real LM Studio instance
— not preemptive engineering:

| Mechanism | Where | Why |
|---|---|---|
| **Auto-start on launch** | `cli.py`'s `_check_openai_compatible` → `core/model.py`'s `ensure_lmstudio_running` | If the server isn't reachable at startup, runs `lms server start` and loads the configured model — narrating progress via an `on_status` callback — instead of just telling the user to open LM Studio themselves. Surfaces the *real* reason on failure (e.g. `lms` not on PATH → "install LM Studio") rather than a generic "cannot reach" message. |
| **Context length always matches config** | `switch_lmstudio_model` | Every load passes an explicit `--context-length` from `Config.model_context_length`. Without this, LM Studio loads at its own default (often 4096–8192) regardless of `config.yaml` — confirmed live, twice, with two different models: `aicoder` builds a prompt sized for the configured context, LM Studio silently truncates to its own smaller window, and the request fails with "tokens to keep... greater than the context length" on the first real turn. If a model's already loaded but at the *wrong* context length (loaded by hand, or by an older `aicoder`), it's unloaded and reloaded rather than left mismatched. |
| **Idle-unload TTL, not aggressive unload-on-exit** | `switch_lmstudio_model`'s `--ttl LMSTUDIO_IDLE_UNLOAD_SECONDS` (600 = 10 min) | Every load also sets LM Studio's own native TTL, so a model unloads on its own after being genuinely unused — this is LM Studio's own idle-tracking, survives `aicoder` crashing, and (deliberately) won't unload a model something *else* (another `aicoder` session, LM Studio's own chat UI) is still actively using, since real requests reset the clock. A model loaded without a TTL at all doesn't count as "already correctly loaded" either — it gets reloaded with one. |
| **Explicit unload on a clean exit** | `agent/loop.py`'s `_try_lmstudio_unload`, wired into both `run_agent_repl`'s exit path and the TUI's `on_unmount` | A deliberate `/exit` (or Ctrl+C/Ctrl+D) is a much stronger "I'm done" signal than idle — unloads immediately rather than waiting out the TTL. Silent on failure (unlike the load path's warning): the app is already exiting, and the TTL is still there as a fallback either way. |
| **Auto-reload before a turn, if idle-unloaded** | `AgentSession.send()` → `_ensure_model_loaded()` | Checked (read-only, `is_lmstudio_model_loaded`) at the start of every turn. Matters beyond convenience: LM Studio *already* auto-loads a downloaded-but-unloaded model on the first request that touches it — but at its own default context length and no TTL, silently reintroducing the exact context-length crash above, one idle-timeout later. `_ensure_model_loaded` reloads it *properly* (context length + TTL) before that implicit auto-load gets the chance to. |

All four are gated on `is_lmstudio_endpoint(base_url)` — a custom/remote
OpenAI-compatible server is never touched by any `lms` CLI shellout.

---

## Known gotchas for future changes

Non-obvious traps hit while building the above — worth reading before
touching these areas again.

- **`RichLog.write()` measures-then-shrinks unless given an explicit
  `width=`.** Without it, `write()` measures the renderable's own declared
  width, then shrinks it to `scrollable_content_region.width` at the moment
  the (possibly deferred) write actually replays — confirmed live, that
  locks onto whichever resize event `RichLog` receives *first*
  (`RichLog.on_resize` latches `_size_known` permanently on the first
  non-zero width it sees), which can be an early, narrower intermediate
  layout pass, not the final one. A fixed-pixel-width `Table` (the startup
  banner's two-column layout) doesn't reflow gracefully when shrunk — it
  mangles content mid-cell. Always pass `width=` explicitly to `write()`
  when rendering something layout-sensitive (see `agent/tui.py`'s
  `on_mount`).
- **Rich's `Table` `ratio=` columns ignore `min_width` once `expand=True`.**
  Confirmed live: a 1:2 ratio split at console width 80 measured out to
  27:53, nowhere near a requested 41-column floor. Use explicit computed
  integer `width=` per column instead if a minimum must be guaranteed (e.g.
  protecting the startup banner's logo from wrapping).
- **A Rich `Table` column's default `overflow` is `"ellipsis"` — silent
  truncation from the end, not wrapping.** For anything where the *end* of
  the text is the meaningful part (a workspace path — the last component is
  the actual project name), this hides exactly the wrong part. Set
  `overflow="fold"` to wrap instead.
- **`reasoning_content` from local "thinking" models is invisible to
  LangChain's `ChatOpenAI`**, by that class's own documented design — it
  doesn't extract non-standard provider fields. A model can stream an
  entire turn of reasoning and `aicoder` sees only empty `content`, empty
  `tool_calls`, sometimes even empty `response_metadata`. There is currently
  no way to surface that reasoning in the UI; `_run_steps`'s retry-with-nudge
  (decision #19 above) manages the *symptom*, not the cause. Enabling
  `enable_thinking: false`/`/no_think` did not work against a live LM
  Studio + Qwen3.5 setup — this may be model/build-specific.
- **LM Studio's own default `/v1/models` reachability check isn't the same
  as "the configured model is actually loaded at the right settings".** A
  server can be reachable with the model downloaded but loaded (or about to
  be lazily auto-loaded) at the wrong context length/TTL — see the whole
  [LM Studio lifecycle management](#lm-studio-lifecycle-management) table
  above, which exists entirely because of this gap.
- **Test isolation: real `AgentSession`/`AICoderApp` instances use the real,
  unmocked config**, which points at LM Studio's actual default endpoint.
  `tests/test_loop.py` and `tests/test_tui.py` both carry an autouse
  `_assume_model_already_loaded` fixture patching
  `core.model.is_lmstudio_model_loaded` — without it, every test calling
  `send()` would shell out to a real `lms ps` subprocess. If you add a new
  test file that constructs a real session, check whether it needs the same
  fixture (verify by running the suite with `lms` stripped from `PATH`).

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
unit-tested with the model call injectable, so the suite runs without a live
model server. See [`evals/README.md`](../evals/README.md). Caveat: numbers are small-n with a
same-model judge — deltas are decisive, absolute figures want a stronger
`--judge-model` and higher `--repeat`.

---

## LLM integration

Uses LangChain message types (`HumanMessage`, `AIMessage`, `SystemMessage`,
`ToolMessage`). `core/model.get_chat_model()` builds a `ChatOpenAI` pointed at
the configured `base_url` (conversational `temperature=0.3`, precise `0.1`)
and binds the agent's tools. `core/model.selftest()` checks tool calling
(native or text-recovered) for the configured model.

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

**Session log** (`~/.aicoder/memory/<project_id>/sessions/<session_id>.json`
— one file per session, `session_id` a filesystem-safe timestamp, never
overwritten across sessions): `turns` is the human-analyzable log `/history`
reads (real diffs, not just "wrote N chars"); `raw_messages` is everything
after the system prompt via LangChain's `messages_to_dict`, what
`aicoder --continue` restores:
```json
{
  "session_id": "2026-07-06T14-30-00-123456",
  "workspace": "/path/to/project",
  "started_at": "2026-07-06T14:30:00.123456+00:00",
  "turns": [
    {
      "prompt": "fix the login bug",
      "actions": [
        {"tool": "read_file", "args": {"path": "login.py"}, "result": "...", "diffs": []},
        {"tool": "write_file", "args": {"path": "login.py", "content": "..."},
         "result": "Updated login.py (512 chars).",
         "diffs": [{"path": "login.py", "diff": "--- a/login.py\n+++ b/login.py\n..."}]}
      ],
      "answer": "Found it — a stale session check.",
      "completed": true
    }
  ],
  "raw_messages": [
    {"type": "human", "data": {"content": "fix the login bug", ...}},
    {"type": "ai", "data": {"content": "Found it — a stale session check.", "tool_calls": [], ...}}
  ]
}
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

# models — download in LM Studio's own model search
# lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF   # the agent driver
# nomic-ai/nomic-embed-text-v1.5-GGUF                 # embeddings (web research + documents)
```
