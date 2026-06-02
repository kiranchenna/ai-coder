# AICoder — Architecture

**Version:** 3.0.0 | **Language:** Python 3.10+ | **Entry point:** `cli.py` → `aicoder` CLI command

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
│   ├── model.py                # ChatOllama factory + tool-call recovery + selftest
│   └── project.py              # Test-command detection
│
├── agent/                      # The agentic core
│   ├── loop.py                 # Tool-calling loop, REPL, slash commands
│   ├── tools.py                # The 14 agent tools
│   ├── planner.py              # Decompose + run resumable task plans
│   └── prompts.py              # System prompt
│
├── rag/                        # Retrieval-augmented knowledge
│   ├── store.py                # ChromaDB vector store with chunking + TTL
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
└── tests/                      # pytest unit tests
    ├── test_agent.py           # Agent logic (parsers, chunking, detection, memory)
    ├── test_config.py
    └── test_file_tools.py
```

Runtime data lives under `~/.aicoder/` (config, RAG store, per-project memory),
not in the repo.

---

## Tech stack

| Category | Library | Floor |
|---|---|---|
| LLM | langchain-ollama | 1.0+ |
| LLM core | langchain-core | 1.0+ |
| Terminal UI | rich | 13.0+ |
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
  └── run_agent_repl() [agent/loop.py]
        ├── "plan <goal>"  →  agent/planner.py  →  task list → AgentSession per task
        ├── "/command"     →  _handle_command (model/tools/memory/clear)
        └── plain English  →  AgentSession.send():
                                 model.invoke(history + tools)
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

```yaml
model:
  provider: ollama
  name: qwen2.5-coder:7b
  base_url: http://localhost:11434
  temperature: 0.3
  temperature_precise: 0.1
  context_length: 16384

shell:
  confirmation: always       # always | smart | never

files:
  confirmation: auto         # always | auto | never
  backup: true

knowledge:
  embedding_model: "nomic-embed-text-v2-moe"
```

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
6. **Resumable plans** — `plan <goal>` saves task state after each step so a build
   resumes after a quit.
7. **Strictly local** — no cloud, no API keys; all data under `~/.aicoder/`.

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
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text-v2-moe
```
