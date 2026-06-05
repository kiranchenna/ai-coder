# Changelog

## Unreleased
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
