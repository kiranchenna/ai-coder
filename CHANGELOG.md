# Changelog

## Unreleased
- **Vision: paste a screenshot with Ctrl+V, just like Claude Code.** Claude's
  model is natively multimodal; Ollama's local ecosystem splits vision and
  coding into separate model families (the curated coding catalog is
  text-only), so this is a two-model handoff rather than one model seeing the
  image directly.
  - `ChatInput.action_paste` checks the *real* OS clipboard for an image via
    Pillow's `ImageGrab.grabclipboard()` — confirmed independently that this
    bypasses the terminal entirely (via `osascript` setting the clipboard,
    then a real pty sending a raw Ctrl+V byte to the actual global install).
    `Input`'s default Ctrl+V binding only pastes Textual's own `app.clipboard`
    (text copied *within* the app), not the real system clipboard a
    screenshot lands on. Falls back to normal text paste when there's no
    image. A stray slash command never consumes a pending attachment.
  - `AgentSession.describe_images`/`send_with_images`: a vision-capable model
    (new `vision.model` config, default `qwen2.5vl:7b`, pulled with
    confirmation on first use) describes the image in text via a multimodal
    `HumanMessage` (base64-encoded image content) — built fresh, never bound
    to the session's tools, never persisted as the default driver, unlike
    `/model`. That description is folded into a normal text turn for the
    regular coding model, so the rest of the agentic loop (tool calling,
    editing) is completely unchanged from a plain turn.
  - New `/vision <path>` command — the file-path equivalent, usable in both
    front-ends. Deliberately not the sandboxed `resolve()` used by
    `read_file`/`write_file`, since screenshots typically live outside the
    workspace and this only reads bytes to show a model, not write/execute
    anything.
  - New `/vision model` picker, giving the same flexibility `/model` has but
    for the vision model: installed models + a curated `VISION_MODELS`
    catalog (`core/model_catalog.py`, a genuinely separate list from the
    coding catalog — Ollama's vision and coding models are different model
    families, not one family with a "supports vision" flag), grouped
    sections, current model pre-highlighted, "Other…" escape hatch, and
    `/vision model <name>` to switch straight to one — all via `/model`'s
    existing picker machinery, generalized (`_build_model_menu`/
    `_run_model_picker`, extracted from `_handle_model_command`) rather than
    duplicated. Persists to `vision.model`, never touches the coding model.
  - New dependency: `pillow>=10.0`.
  - Caught and fixed two real bugs along the way: (1) a structural test
    isolation gap — `_isolate_config` (three copies, one per test file)
    redirected the config file path but never reset `get_config()`'s cached
    module-level singleton, so an earlier test's in-memory mutation could leak
    into a later "isolated" test regardless of the path redirect; discovered
    because an earlier ad-hoc verification script (not a pytest test) had
    actually corrupted the real `~/.aicoder/config.yaml`'s model name, and the
    leaky singleton then propagated that into the suite. Fixed by also
    resetting the cached singleton to `None` in all three copies. (2) A
    `_pull_via_ollama` refactor (extracted from `_pull_arbitrary_model`) to
    share the confirm+pull+report logic with the vision model's own pull flow.
  - 22 new tests across `test_loop.py`/`test_tui.py`, verified live end to end
    on the real global install (real clipboard image via `osascript`, real
    Ctrl+V byte via a real pty) — stopping short of an actual vision-model
    pull (a multi-GB download) for the live check; the two-model handoff
    logic itself is covered by a scripted fake vision model in the test suite.
- **Full-screen chat UI, matching Claude Code's interface.** On a real
  terminal, `aicoder` now launches a Textual application (`agent/tui.py`)
  instead of the plain print-and-scroll REPL: a scrolling chat log with a
  pinned input box at the bottom, arrow-key menus, and a live "thinking"
  indicator with an elapsed-time counter and esc-to-interrupt — the same
  overall shape as Claude Code's UI. Piped/redirected/scripted output
  (including the whole test suite) is unaffected — it still gets the original
  `run_agent_repl` REPL, since a full-screen UI needs a real terminal.
  - Every existing slash-command handler, confirmation prompt, and tool works
    *unchanged* inside the new UI: a `console.print`-compatible adapter
    (`RichLogConsole`) is swapped in for each module's `console` singleton,
    and `rich.prompt.Confirm.ask`/`Prompt.ask` are monkeypatched for the
    session's lifetime (the same technique this project's tests already use)
    to route through a Textual modal — verified against real, unmodified call
    sites (e.g. the file-write confirmation in `agent/tools.py`), not just
    synthetic ones.
  - `/model`'s picker gets a genuine arrow-key `OptionList` on top of the
    generic bridge above — grouped under section headers (Installed /
    Recommended — tier, skippable by arrow keys, not selectable) matching the
    plain-REPL panel's own grouping, opening with the current model already
    highlighted and checkmarked (✓) instead of always starting at the top —
    the same picker style Claude Code uses (`ChoiceModal`'s `groups`/
    `initial_index` params, reusable by future pickers). An "Other… (type any
    Ollama model name)" entry covers anything beyond the curated
    recommendations (Ollama has no API to browse its full library) —
    `_handle_custom_model_entry` prompts for a tag, then pulls it (with
    confirmation) if not already installed. `_pull_arbitrary_model` uses
    `subprocess.run` with an argv list rather than `run_command`'s
    `shell=True`, since the tag here is raw user input and shell-interpolating
    it would be an injection risk (`_confirm_and_pull`'s existing shell path
    stays as-is — it only ever sees our own hardcoded catalog tags).
  - Typing `/` opens an autocomplete dropdown of every slash command (name +
    description, from the new `SLASH_COMMANDS` list), narrowing as you type,
    arrow keys + Tab/Enter to accept, Escape to dismiss — built on the
    `textual-autocomplete` package, positioned above the input (it's docked at
    the bottom with the Footer directly beneath it, so "below" has no room —
    overrides the library's default downward placement). The dropdown gets out
    of the way once the input already matches a command exactly, so a single
    Enter runs it (otherwise the library's own Enter handling "completes" an
    already-complete command instead of submitting it).
  - `AgentSession` gained a real interrupt mechanism
    (`request_interrupt`/`_TurnInterrupted`), checked between streamed chunks
    and between tool-call steps — best-effort, since a chunk already arriving
    over the network still has to land first.
  - New dependencies: `textual>=8.0`, `textual-autocomplete>=4.0`. New dev
    dependency: `pytest-asyncio` (for Textual's own headless `run_test()`
    harness). 44 new tests, verified live in both a headless harness and a
    real pty (byte-level checks of the alt-screen escape codes and the custom
    cyan/gold theme actually rendering, plus a real, unmodified pipx global
    install). Also fixed: a flaky-test race where the status-bar interval
    timer could fire mid-teardown against an already-unmounted widget (now
    stopped explicitly on unmount), and `_patch_prompts`'s restore
    unconditionally deleting `Confirm.ask`/`Prompt.ask` instead of putting
    back whatever was there before — broke any test that monkeypatches
    Confirm/Prompt itself and *also* exercises the TUI in the same test.
    Also fixed a real flaky-test race along the way: the status-bar interval
    timer could fire mid-teardown and query an already-unmounted widget — now
    stopped explicitly on unmount.
- **Full-screen terminal session, like Claude Code's.** On a real terminal,
  the REPL now runs inside Rich's `Console.screen()` — the same "alternate
  screen buffer" mechanism `vim`/`less`/`htop` use: swaps to a separate, blank
  screen for the session and restores the terminal to exactly what was there
  before on exit (normal exit, `/exit`, Ctrl-D/Ctrl-C, or any error) — no trace
  of the session left in your scrollback. A no-op when output isn't a real
  terminal (piped/redirected/scripted usage, tests, CI — verified both ways:
  the escape codes appear on a forced-terminal console and are absent on a
  non-terminal one). Cursor stays visible (`hide_cursor=False`), since input is
  still normal `Prompt.ask()`, not a custom raw-mode editor. Since scrollback
  is gone after exit, `/export` is now the way to keep a copy of a
  conversation. 2 new tests.
- **BREAKING: `develop`, `dev`, `plan`, `resume`, and `exit`/`quit` now require
  a leading `/`** (`/develop`, `/dev`, `/plan`, `/resume`, `/exit`/`/quit`/`/q`),
  for one consistent command surface instead of five bare-word special cases
  alongside every other command already being `/`-prefixed. Bare `exit`/`quit`
  no longer quits — Ctrl-D/Ctrl-C still do, or use `/exit`. Typing the bare
  word now (e.g. "plan a todo app") is just sent to the agent as a regular
  message, same as any other unrecognized text. `_handle_command` now returns
  a bool signaling whether the REPL should exit, instead of `run_agent_repl`
  bare-word-matching `_EXIT_WORDS` itself (removed). Updated every usage
  example across the README and docs accordingly. 17 new tests.
- **10 new slash commands, mirroring Claude Code's, for anyone coming from
  there.** `/init` (explore the codebase and write/update AICODER.md — reloads
  the system prompt so it applies immediately, no restart), `/status`
  (workspace/model/provider/profile on demand), `/context` (conversation size
  vs. the compaction budget), `/compact` (force the same auto-compaction on
  demand), `/permissions` (view/change shell & file confirmation modes without
  restarting — reuses `Config.set_shell_confirmation`/`set_file_confirmation`),
  `/review` (ask the agent to review the current git diff), `/mcp` (list
  configured MCP servers, connection status, discovered tools — added
  `MCPManager.status()`), `/hooks` (list configured lifecycle hooks), `/export
  [file]` (save the conversation transcript to markdown), `/doctor` (the
  `--selftest` diagnostic, callable without restarting). Deliberately did NOT
  add commands that don't map onto this app's shape: `/login`/`/logout` (no
  accounts), `/cost` (no token costs on a free local model), `/agents`
  (single-agent design), `/ide` (no IDE integration), `/vim` (no full editor
  input mode). 26 new tests (`tests/test_slash_commands.py`).
  - **Found and fixed a real bug along the way**: the text-tool-call-recovery
    fallback's injected placeholder (`"(Requested tools: X)"`) is stored as an
    `AIMessage`, so it appears in history as the model's own prior turn — a
    small model (observed with `qwen2.5-coder:7b`) tends to imitate the
    *surface form* of its own previous turn, and would copy that terse,
    request-shaped placeholder verbatim as a fake "tool call" on the next turn
    instead of emitting a real one. Reworded it as a past-tense statement of
    fact ("I already called `X` — its result is below") instead, which
    eliminated that specific failure in testing.
  - **Honest finding, not fully solved**: `/init` on a fresh multi-file
    codebase didn't reliably complete explore→write in one shot on this 7B
    model even after the fix above and three rounds of prompt refinement — it
    sometimes narrates its next step in prose ("I'll call read_file...")
    instead of calling it. The command's plumbing (prompt, tool execution,
    instruction-reload) is verified correct; the remaining gap is this
    specific local model's multi-turn follow-through, the same honest
    small-model caveat documented elsewhere in this project. Larger models are
    expected to be more reliable here, consistent with the project's existing
    "bigger models help" guidance. If it stalls, just tell the agent to
    continue — it's a normal conversational session, not an isolated mode.
- **Official logo + a branded terminal startup banner.** Added `assets/icon.png`
  (the app icon/logo), shown at the top of the README. The CLI's startup
  banner now carries a small brand mark (`⟨❯_⟩ AICoder`) in the logo's cyan/amber
  palette instead of plain magenta text — terminals can't render the actual
  bitmap logo inline (no universal image protocol), so this is the in-terminal
  equivalent, echoing the icon's bracket and `>_` cursor motifs.
- **Support for non-Ollama backends via `model.provider: openai_compatible`.**
  Ollama remains the default and needs no changes, but you can now point
  AICoder at any server or API that speaks the OpenAI chat-completions
  protocol — a local runtime sized to your hardware (llama.cpp server, vLLM,
  LM Studio, text-generation-webui, LocalAI, ...) or a hosted API (OpenAI,
  OpenRouter, Groq, Together, ...) with your own key. Configure via
  `model.provider`, `model.base_url`, `model.name`, and the new
  `model.api_key` (blank for local servers that don't check it) in
  `config.yaml`. Needs the optional `langchain-openai` package
  (`pip install "ai-coder[openai]"`) — omitted from the default install so
  plain-Ollama users never pull it in; a missing package gives a clean,
  actionable error instead of a traceback. The Ollama-specific startup
  checks (install-offer, reachability) and the rich `/model` picker
  (list/recommend/pull) only make sense for Ollama's own APIs, so they're
  skipped for other providers in favour of a simpler "here's your current
  model/endpoint, edit config.yaml or use `/model <name>` to change it"
  message. Verified live end-to-end against a real OpenAI-compatible
  endpoint (Ollama's own `/v1` shim) — including a genuine finding: native
  tool-calling came back empty through that specific compat path, but the
  model still emitted a valid tool call as text, and the app's *existing*
  text-recovery fallback (built earlier for weaker local models) caught it
  correctly — the safety net already in the codebase applies uniformly
  across providers. 12 new tests (`tests/test_model_provider.py` +
  additions to `test_config.py`, `test_cli.py`, `test_loop.py`).
- **Offers to install Ollama itself if it's missing.** On startup, `cli.py` now
  checks whether the `ollama` binary is on PATH at all (distinct from
  "installed but not running," which was already handled) via
  `core/ollama_install.py`. If it's missing, it shows Ollama's own official
  install command in full and asks before running anything — declining, or a
  non-interactive/no-stdin context, falls back cleanly to printing
  ollama.com/download instead of crashing. Uses the same script for macOS and
  Linux and the official PowerShell one-liner for Windows (verified against
  ollama/ollama's own README, not guessed). 9 new tests
  (`tests/test_ollama_install.py`, `tests/test_cli.py`).

## 3.1.0 - 2026-07-04
- **Repo hygiene ahead of going public.** Added `CONTRIBUTING.md` (dev setup,
  the pytest/ruff bar for PRs, code style, and a pointer to `evals/` for
  anyone changing Developer Mode's quality-lever defaults). Removed the
  tracked `output/task_tracker/` and `specs/task_tracker*` demo artifacts left
  over from earlier testing (placeholder values only — no real credentials;
  `output/`/`specs/` are gitignored, so they won't be re-tracked).
- **`/model` is now an interactive picker, like Claude Code's.** Typed alone,
  it lists every model you've pulled (via Ollama's `/api/tags`), marks the
  active one, and lets you pick by number; `/model <name>` still switches
  directly. Either way the choice is **persisted to `config.yaml` as your new
  default** (previously `/model <name>` only changed it for that session).
  Warns if the chosen model isn't actually pulled yet, and falls back
  gracefully with a same-turn `/model <name>` suggestion if Ollama can't be
  reached to list models. Added `core.model.list_ollama_models` /
  `is_model_pulled`, which `cli.py`'s startup pulled-model check now reuses
  too (one code path instead of two).
- **`/model` now recommends models you haven't pulled yet, by preference.**
  Added `core/model_catalog.py`: ~11 hand-verified Ollama tags (confirmed
  against ollama.com/library, not guessed) grouped into **fast & light
  (~8GB)**, **balanced (~16GB)**, and **powerful (24GB+)** tiers, each with a
  one-line reason. The `/model` picker now shows these alongside your
  installed models (deduped — an already-pulled catalog entry isn't listed
  twice); picking a not-yet-installed one prompts to confirm the download
  size, runs `ollama pull`, then switches and persists as the new default on
  success. A brand-new user with zero models pulled now sees the full
  curated list instead of a dead end. README's "Choosing a model" table and
  `config.py`'s model-name comment (which cited an unverified
  `qwen2.5-coder:4b` and the superseded `deepseek-coder:6.7b`) now point at
  the same verified catalog.
- **Tooling/hygiene pass** (from a full-codebase audit — feature completeness,
  dependency currency, and best-practices review):
  - **`devmode.profile` is now visible in-session** — the active profile shows
    in the startup banner, `/help`, and `dev status`, so a user editing
    `config.yaml` can confirm which levers are actually active.
  - **Python floor raised to 3.11** (`requires-python`, classifiers, README,
    docs). 3.10 enters its final security-only phase in Oct 2026; nothing in
    the codebase used 3.10-only syntax, so this is a clean bump.
  - **Ruff is now explicitly configured** (`[tool.ruff]` in `pyproject.toml`) —
    the rule set it was already running (Pyflakes + a pycodestyle subset) is
    declared rather than implicit, with a documented per-file exemption for the
    tests' deliberate point-of-use import style. `ruff` is now a declared `dev`
    extra instead of an ad hoc local install.
  - **Full-tree lint cleanup**: fixed a genuinely misplaced import block in
    `agent/tools.py`, renamed 6 ambiguous `l` variables to `line`/`lang`, and
    removed 3 dead `tempfile` imports in tests. `ruff check .` is clean across
    the whole repository, not just recently-touched files.
- **Developer Mode eval harness + evidence-based lever defaults.** Added
  [`evals/`](evals/): a lever-ablation harness that runs one design phase under
  different quality-lever configs, grades each decision with a judge model
  against the phase's `must_cover` checklist, and tabulates the score delta vs.
  wall-clock. The first ablation (security phase) found **`reflect` carries
  essentially all of the quality gain** while **`best_of` only pays with a
  stronger judge**. Acting on that:
  - Quality levers are now bundled into a single **`devmode.profile`** dial —
    `fast` (reflect only), **`balanced`** (new default: reflect + consistency +
    build-review), or `thorough` (everything). Individual levers can still be
    overridden in config.
  - **`best_of` is gated on `judge_model`** — best-of-N only fires when a
    stronger critic model is configured to rank the candidates; otherwise it is
    skipped (with a note) in favour of a single reflected pass. This drops a
    default that cost 3× generation latency for no measured quality gain.
  - A second eval (`run_consistency_eval`) measures the `consistency_check`
    lever as contradiction-detection precision/recall on labeled cases — it
    scored 100% precision / 60% recall (every blatant cross-phase contradiction
    caught with zero false alarms; subtle ones missed), confirming it belongs in
    `balanced` as cheap insurance.
  - A third eval (`run_build_review_eval`) measures the `build_review` lever by
    handing the live review pass drafts with planted placeholders — 100% removal
    (each replaced with a real implementation) with clean drafts left intact. All
    four quality levers are now backed by a reproducible number.
  - Docs/code drift fixed: the default embedding model name and the
    context-length fallback now match across README and config, and the
    end-of-design panel points to `dev build` / `dev resolve` (the build
    hand-off shipped).
- **Review fixes (correctness, safety, context)** — a pass of fixes from a
  codebase review:
  - **Planner no longer marks unfinished tasks done.** `AgentSession.send` now
    records whether the turn reached a genuine answer (`last_turn_complete`); the
    planner leaves a task **pending** when the step cap was hit instead of
    silently marking it complete.
  - **Developer Mode grounds later phases on digests, not raw artifacts.**
    Chaining ~14 full artifacts overflowed `num_ctx` and Ollama truncated the
    *earliest* (most foundational) decisions; phases are now grounded on the
    compact, already-cached per-phase digests within a bounded budget.
  - **`dev build` feeds the cross-file symbol index** of already-generated files
    to each new file, so imports reference real names instead of being guessed
    (fewer compile/test fix-loop rounds). `pytest` exit code 5 ("no tests
    collected") is no longer treated as a build failure.
  - **Safer shell heuristic.** `smart` mode now inspects each segment of a
    chained command (`&&`/`|`/`;`) and catches more destructive forms
    (`find -delete`, truncating `>` redirects, `xargs rm`, `git push --force`,
    …). It is still best-effort, not a security boundary — documented as such.
  - **Tighter text-tool-call recovery.** A tool call recovered from message text
    must now *dominate* the message, so a large illustrative JSON example in an
    explanation can't trigger a real `write_file`/`run_shell`.
  - **RAG:** default embedding model is the canonical `nomic-embed-text`; a
    failed semantic search now warns once instead of silently returning nothing.
  - **`fetch_url`** honours the caller's character budget (the agent tool reads
    up to 20k instead of being silently clipped to 8k).
  - **MCP shutdown** drains and joins its background loop so server subprocesses
    are torn down instead of leaked.
  - **`--version`** reads the installed package version instead of a hardcoded
    string. Generated demo output (`output/`, `specs/`) is no longer tracked in
    git. New tests cover the shell heuristic, planner completion, the recovery
    gate, the judge parser, and digest grounding.
- **Developer Mode build loop** — `dev build` now closes the loop: after
  generating files it runs a **compile check → tests → agentic-fix** loop (up to
  3 rounds) so it produces code that actually runs, not just plausible code. It
  finds a **nested project root** (tests in a subdirectory are no longer missed),
  writes a **`build_manifest.json`** mapping each file to the design phases it
  implements, and a Python syntax pass catches cross-file breakage before tests.
- **Developer Mode — fast mode & hybrid judging** — `develop --fast <idea>` runs
  the whole design in one pass (each role makes the senior decisions itself, no
  back-and-forth). `devmode.judge_model` lets an optional stronger model handle
  just the high-leverage critic steps (best-of judging, consistency, review)
  while generation stays on the main local model.
- **Developer Mode quality** — squeeze better output from a small local model:
  per-phase "must-cover" checklists force domain-defining depth (e.g. Security
  must name the actual E2E protocol / per-device keys, Architecture the real-time
  backbone), a draft→critique→revise reflection pass refines each decision
  (config `devmode.reflect`), the heavy phases (data model / API / architecture)
  are designed one unit at a time, research phases derive 2-3 targeted queries
  (current versions / protocols) instead of one generic search, the critical
  phases (requirements, security) generate several candidate decisions and a
  judge keeps the strongest (config `devmode.best_of`), and the prompts push
  depth and forbid dropping requested features. `dev build` now self-reviews
  each generated file (draft → critique for bugs/placeholders/convention misses →
  fix) before writing it (config `devmode.build_review`). After each phase a
  cross-phase consistency check digests the new decision and flags contradictions
  with earlier ones (auth/tech/datastore mismatches, dropped scope, server-side
  secrets vs an E2E promise), logged to `docs/dev/consistency_notes.md` (config
  `devmode.consistency_check`). `dev resolve` makes those findings actionable: it
  runs a holistic cross-phase review, and for each contradiction you accept it
  rewrites the offending phase's decision (with anti-truncation and echo guards)
  and auto-resyncs the code — moving the design toward self-correcting.
- **Developer Mode roles** — added Product Manager (vision/MVP), Market Analyst
  (competitors), Technical Writer (docs plan), and a Design Reviewer that
  critiques all decisions for consistency/gaps/risks before the build (14 roles).
- **Developer Mode** — `develop <idea>` runs a role-driven SDLC
  design: Requirements → Architecture → Security/NFR → Data model → API →
  App flow → UI/UX → Testing → Deployment → Conventions, each a full discussion
  that captures an editable artifact in `docs/dev/` (conventions → `AICODER.md`).
  Resumable (`dev` / `dev status` / `dev revisit <phase>`). `dev build` turns the
  design into code: proposes a file plan (you control it via build_plan.json),
  generates file-by-file grounded in the spec + conventions, and verifies.
  `dev revisit <phase>` re-opens a decision and, if it changed, auto-resyncs the
  code (diffs old→new and runs an agentic apply+verify task). Brownfield-aware:
  for an existing repo it grounds every phase in the codebase and infers the
  current coding conventions.
- **Windows robustness** — cross-platform shell quoting for git commands
  (cmd.exe-safe), forward-slash path output from search/find/index, and
  `gradlew.bat` detection.
- **Hooks** — run user shell commands on agent events (PreToolUse can block a
  tool, PostToolUse to auto-format/notify, Stop on turn end). Opt-in via config.
- **`find_symbol` + large-file paging** — a fast definitions index to jump to
  where things are defined, and `read_file(offset, limit)` to page big files.
- **MCP client support** — connect Model Context Protocol servers via
  `mcp.servers` in config; their tools are exposed to the agent alongside the
  built-ins. Opt-in (`pip install "ai-coder[mcp]"`).
- **`AICODER.md`** — a user-authored project-instructions file (plus optional
  global `~/.aicoder/AICODER.md`) loaded into the agent's context every session.

## 3.0.0

A complete rewrite into a single **agentic** coding assistant. The legacy
7-phase planning pipeline is gone; everything now runs through one tool-calling
loop on a local Ollama model.

### Highlights
- **Agentic loop** with native tool calling (and recovery of tool calls that
  small local models emit as text), token-by-token streaming, and a step cap.
- **18 tools**: file navigation (`list_files`, `find_files`, `search_code`,
  `read_file`), editing (`write_file`, `edit_file` with whitespace-tolerant
  matching), `run_shell`, verification (`run_tests`, `run_checks` for
  linters/type checkers), web research + RAG (`research`, `fetch_url`,
  `rag_search`, `read_document`), git (`git_status`, `git_diff`, `git_commit`),
  and memory (`remember`, `recall`).
- **`plan <goal>` / `resume`** — decompose a goal (grounded in ingested
  documents) into an ordered, resumable task list and build it step by step.
- **RAG** — ChromaDB + Ollama embeddings with real chunking; web research is
  global, ingested documents (PDF/docx/md) are scoped per project.
- **Persistent project memory** auto-loaded each session; **history compaction**
  keeps long sessions within the context window.
- **In-session commands**: `/help`, `/model`, `/tools`, `/diff`, `/memory`,
  `/knowledge` (`learn` / stats / `clear` / `clear all`), `/clear`.

### Requirements
- Python 3.10+ and a running [Ollama](https://ollama.com/). Recommended models:
  `qwen2.5-coder:7b` (driver) and `nomic-embed-text-v2-moe` (embeddings).
