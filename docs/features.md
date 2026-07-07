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
| `/develop <idea>` | Developer Mode: role-driven SDLC design → build (see below) |
| `/dev …` | `/dev` (resume) · `/dev status` · `/dev build` · `/dev revisit <phase>` · `/dev resolve` |
| `/plan <goal>` | Decompose a goal into an ordered, resumable task list and build it |
| `/resume` | Continue an in-progress plan |
| `/init` | Have the agent explore the codebase and write/update `AICODER.md` (mirrors Claude Code's `/init`); reloads the system prompt so it takes effect immediately, no restart needed |
| `/model [name]` | **Ollama provider (default):** `/model` alone opens an interactive picker: your pulled models (current marked) plus curated, not-yet-pulled recommendations grouped by fast/balanced/powerful tier — pick one to pull (with confirmation) and switch. `/model <name>` switches straight to a name you know. Either way the choice is persisted to `config.yaml` as the new default (mirrors Claude Code's `/model`). **Other providers:** `/model` shows the current provider/model/endpoint instead (no discovery API to pick from); `/model <name>` still switches the model id |
| `/status` | Workspace, model, provider, and Developer Mode profile — the startup banner's content, on demand |
| `/context` | Conversation size vs. the auto-compaction budget, as a percentage |
| `/compact` | Force the same summarize-older-turns compaction `AgentSession` runs automatically, right now |
| `/permissions` | Bare: show the shell/file confirmation modes. `/permissions shell\|files <mode>`: change and persist one, without restarting |
| `/review` | Ask the agent to review the current `git_diff` for correctness bugs and cleanup opportunities |
| `/tools` | List the agent's tools |
| `/mcp` | List configured MCP servers, whether each connected, and their discovered tools |
| `/hooks` | List configured `PreToolUse`/`PostToolUse`/`Stop` hooks |
| `/diff` | Show the git diff of changes so far |
| `/memory` | Show what's remembered about this project |
| `/knowledge` | RAG: `/knowledge learn <topic\|URL>` researches & caches; bare shows stats; `/knowledge clear` clears this project's docs, `/knowledge clear all` wipes everything |
| `/export [file]` | Write the conversation transcript to a markdown file (default: a timestamped name) in the workspace |
| `/doctor` | Run the same tool-calling diagnostic as `aicoder --selftest`, without restarting |
| `/bug` | The issues URL and what to include in a report |
| `/clear` | Forget the conversation (keeps saved memory) |
| `/help` | Show commands |
| `/exit` / `/quit` / `/q` | Leave the session |

Most of these mirror a Claude Code command of the same name/purpose, so the
muscle memory carries over. A few of Claude Code's commands don't map onto
this app's shape (`/login`/`/logout` — no accounts; `/cost` — no token costs on
a free local model; `/agents` — single-agent design, not a custom-subagent
system; `/ide` — no IDE integration; `/vim` — the input prompt doesn't have a
full editor mode) and were deliberately left out rather than faked.

### Full-screen chat UI

On a real terminal, `cli.py` launches `agent/tui.py` — a Textual application
giving AICoder the same overall shape as Claude Code's interface: a scrolling
chat log with a pinned input box at the bottom, arrow-key menus, and a live
"thinking" indicator, all inside the alternate screen buffer (the same mode
`vim`/`less`/`htop` use), so exiting (`/exit`, Ctrl-D, Ctrl-C, or any error)
restores the terminal to exactly what was there before — no session trace
left in your scrollback. Since scrollback is gone once you exit, use
`/export` beforehand if you want to keep a copy of the conversation.

Piped/redirected/scripted output (including the whole test suite) instead
falls back to `run_agent_repl` — the original print-and-scroll REPL, unchanged
— since a full-screen UI needs a real terminal to attach to.

**Design: existing business logic, unchanged.** Rather than rewrite every
slash-command handler and confirmation prompt for the new UI, `agent/tui.py`
makes the *existing* code work inside it unchanged:
- `RichLogConsole` is a drop-in replacement for `rich.console.Console` (the
  actual call surface used everywhere: bare content, plus the odd `end=`/
  `markup=`) that writes into the chat log instead of stdout — swapped in for
  the `console` singleton every module already has.
- `Confirm.ask`/`Prompt.ask` (both from `rich.prompt`) are monkeypatched for
  the lifetime of the session — the same technique this project's own test
  suite already uses, just applied at runtime — to route through a Textual
  modal instead. Every existing confirmation across the app (shell/file
  writes, the Ollama install offer, the devmode discuss loop, ...) becomes
  TUI-native with zero changes to any of those call sites.
- The `/model` picker additionally gets a genuine arrow-key `OptionList` (via
  `ask_choice`/`is_tui_active`), on top of the generic bridge above — grouped
  under section headers (Installed / Recommended — tier, skippable by arrow
  keys, not selectable) matching the plain-REPL panel's own grouping, opening
  with the current model already highlighted and checkmarked (✓) rather than
  always starting at the top — the same picker style Claude Code uses. An
  "Other… (type any Ollama model name)" entry at the end covers anything
  beyond the curated recommendations (Ollama has no API to browse its full
  library, so this list is a hand-picked subset, not everything available) —
  picking it prompts for a model tag, then pulls it (with confirmation) if
  not already installed, or switches straight to it if it is. Also reachable
  directly without the picker at all: `/model <name>` always works, and
  warns rather than silently failing if that name isn't pulled yet.
- Blocking business logic (`AgentSession.send`) runs in a Textual worker
  thread so the UI stays responsive; a real interrupt mechanism
  (`AgentSession.request_interrupt`, checked between streamed chunks and
  between tool-call steps) backs the status bar's "esc to interrupt" —
  best-effort, since a chunk already arriving over the network still has to
  land first, and how finely Ollama batches tokens per chunk isn't something
  AICoder controls.

**"/" autocomplete.** Typing `/` opens a dropdown of every slash command
(name + one-line description, from `SLASH_COMMANDS` in `agent/loop.py`),
narrowing as you keep typing; arrow keys navigate, Tab/Enter accepts (leaving
the cursor ready to type an argument), Escape dismisses. Once the input
already matches a command exactly, the dropdown gets out of the way so a
single Enter runs it — otherwise `AutoComplete`'s own Enter handling would
"complete" an already-complete command instead of submitting it, silently
requiring two Enters. Built on the `textual-autocomplete` package.

**Vision: paste a screenshot with Ctrl+V.** Claude's model is natively
multimodal; Ollama's local ecosystem splits vision and coding into separate
model families, so this is a two-model handoff rather than one model seeing
the image directly:
- `ChatInput.action_paste` (a small `Input` subclass) checks the *real* OS
  clipboard for an image via Pillow's `ImageGrab.grabclipboard()` — confirmed
  independently (via `osascript` setting the clipboard, then a real pty
  sending a raw Ctrl+V byte to the actual global install) that this bypasses
  the terminal entirely. That's necessary because `Input`'s default Ctrl+V
  binding only pastes Textual's own `app.clipboard` (text copied *within* the
  app, e.g. via OSC 52), which isn't the same clipboard a screenshot lands
  on — an image has no text form to forward over stdin in the first place.
  Falls back to the normal text-paste behavior when there's no image.
- The image is saved to a scratch temp file and queued (`AICoderApp.
  pending_images`); a stray slash command never consumes a pending
  attachment, only a real message submission does.
- `AgentSession.describe_images`/`send_with_images` do the handoff: a
  vision-capable model (`vision.model` in config, default `qwen2.5vl:7b`,
  pulled on first use with confirmation) looks at the image and describes it
  in text (via a multimodal `HumanMessage`, base64-encoded image content) —
  built fresh through `get_chat_model(model=...)`, never bound to the
  session's tools and never persisted as the default driver, unlike `/model`.
  That description is folded into a normal text turn for the regular coding
  model, so the rest of the agentic loop (tool calling, editing) is
  completely unchanged from a plain turn.
- `/vision <path>` is the file-path equivalent, usable in both front-ends
  (deliberately *not* the sandboxed `resolve()` used by `read_file`/
  `write_file` — screenshots typically live outside the workspace, e.g. the
  Desktop, and this only reads bytes to show a model, not write/execute
  anything).
- `/vision model` picks the vision-capable model interactively — the exact
  same picker `/model` uses (`_build_model_menu`/`_run_model_picker`, shared
  rather than duplicated: grouped sections, current model pre-highlighted,
  "Other…" escape hatch), just sourced from a separate `VISION_MODELS`
  catalog (`core/model_catalog.py`) and persisting to `vision.model` instead
  of `model.name`. `/vision model <name>` switches straight to `<name>`, same
  shape as `/model <name>`. The two catalogs stay genuinely separate lists
  (not one filtered by "supports vision") since Ollama's vision and coding
  models are different model families, not a flag on one family.

**Model pickers filter obviously-wrong-category installed models.** Ollama's
`/api/tags` lists every locally-pulled model with no way to tell an
embedding-only model or a vision model from a coding model apart. `/model`
excludes the configured `knowledge.embedding_model` from its "Installed"
section (it's never usable for chat); `/vision model` goes further and only
lists installed models matching a known `VISION_MODELS` family (or whatever's
already configured as `vision.model`) — a coding model you happen to have
pulled won't show up there just because it's installed. "Other…" still
covers anything not listed in either picker.

---

## Session persistence, resume, and history

Every session is saved as its own file — nothing gets overwritten across
different sessions — at
`~/.aicoder/memory/<project_id>/sessions/<session_id>.json`
(`session_id` is a filesystem-safe timestamp, also the sort key). Two things
live in each file, serving two different purposes:

- **`raw_messages`** — everything after the system prompt, via LangChain's
  own `messages_to_dict`, which round-trips tool calls and multimodal
  content correctly. This is what `aicoder --continue` (or `-c`) restores:
  it finds the most recently saved *other* session for the current workspace
  and appends its messages after a freshly rebuilt system prompt (never
  persisted/restored itself, since the repo may have changed since).
- **`turns`** — a human-analyzable log, one entry per user message:
  `{prompt, actions, answer, completed}`, where each action is
  `{tool, args, result, diffs}` — `diffs` are **real unified diffs**, not
  just "wrote N chars": `AgentSession._exec()` drains a small module-level
  recorder in `agent/tools.py` (`_pending_diffs`, appended to by
  `_apply_write` only after an actual write — never on a declined or
  no-op write) right after each tool call, so a diff is correlated with
  the exact action that produced it, without changing what any tool
  actually returns to the model.

Both are saved in a `finally` block on every `send()` call — a turn that
gets interrupted or errors still saves its progress (with `completed: false`
and whatever partial `actions` happened), and a save failure (e.g. disk
full) never masks whatever the turn itself raised.

**`/history`** browses this: bare, it lists every saved session for the
current workspace (date, first prompt, turn count, files touched, with the
current session marked); `/history <n>` shows one in full detail — every
prompt, every tool call, the real diff for anything that changed
(syntax-highlighted, same as `/diff`), and the final answer.

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

`/plan <goal>` (`agent/planner.py`):

- Asks the model for an ordered JSON task list, grounded in any ingested
  document via RAG.
- Persists the plan to `~/.aicoder/memory/<project_id>/plan.json`.
- Executes each task through the agent, saving status after every step, and
  pausing for confirmation between tasks.
- **Resumable** — quit anytime; `/resume` continues from the first pending task.

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
- **`/dev build`** (`devmode/build.py`) — proposes a file plan
  (`docs/dev/build_plan.json`, user-editable), then generates each file grounded
  in the spec + conventions (resumable per file), and **closes the loop**: a
  compile check → tests → agentic-fix loop (≤3 rounds, finds a nested project
  root) gets the code running. Writes `build_manifest.json` (file → phases).
- **`/dev revisit <phase>`** / **`/dev resolve`** — change a decision, or
  review→fix cross-phase contradictions; both **auto-resync** the code
  (`devmode/resync.py`) via an agentic diff→apply→verify task.
- **`/develop --fast <idea>`** — runs the whole design in one pass; each role
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
"encrypted at rest") may still pass — review the output, and use `/dev revisit` /
`/dev resolve` for the subtle cases. See [`evals/README.md`](../evals/README.md).

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

### Model providers (`core/model.py`)
- Ollama is the default provider and needs no configuration beyond the
  Requirements above. Set `model.provider: openai_compatible` in config.yaml
  to instead point at any local server (llama.cpp server, vLLM, LM Studio,
  text-generation-webui, LocalAI) or hosted API (OpenAI, OpenRouter, Groq,
  Together, ...) that speaks the OpenAI chat-completions protocol — pick
  whatever fits your hardware or preference. Set `model.base_url`,
  `model.name`, and optionally `model.api_key` (blank for local servers that
  don't check it). Requires `pip install "ai-coder[openai]"`; a missing
  package raises a clean, actionable error rather than a traceback.
  Ollama-specific behavior (the startup install-offer/reachability checks,
  the rich `/model` picker) only runs for the `ollama` provider. RAG's
  embedding model always goes through Ollama regardless of this setting.

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
