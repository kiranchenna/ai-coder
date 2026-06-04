# Changelog

## Unreleased
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
