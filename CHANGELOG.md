# Changelog

## Unreleased
- **Developer Mode quality** — squeeze better output from a small local model:
  per-phase "must-cover" checklists force domain-defining depth (e.g. Security
  must name the actual E2E protocol / per-device keys, Architecture the real-time
  backbone), a draft→critique→revise reflection pass refines each decision
  (config `devmode.reflect`), the heavy phases (data model / API / architecture)
  are designed one unit at a time, research phases derive 2-3 targeted queries
  (current versions / protocols) instead of one generic search, and the prompts
  push depth and forbid dropping requested features. `dev build` now self-reviews
  each generated file (draft → critique for bugs/placeholders/convention misses →
  fix) before writing it (config `devmode.build_review`).
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
