# AICoder — Architecture

**Version:** 2.0.0 | **Language:** Python 3.10+ | **Entry point:** `cli.py` → `aicoder` CLI command

AICoder is a local AI-powered coding assistant CLI. It uses Ollama (local LLM, no cloud, no API keys) to guide users through a 7-phase project planning and code generation pipeline, plus a 24-command REPL for day-to-day coding work.

---

## Directory Structure

```
ai-coder/
├── cli.py                      # Main entry point
├── main.py                     # Legacy 4-phase wizard (original prototype)
├── aicoder_cli/__init__.py     # Installable package wrapper, sets up sys.path
│
├── core/                       # Core application logic (8 modules)
│   ├── config.py               # Configuration (~/.aicoder/config.yaml)
│   ├── context.py              # Workspace scanner & AI context builder
│   ├── knowledge.py            # Vector RAG (ChromaDB + Ollama embeddings)
│   ├── memory.py               # Persistent conversation memory + auto-summarization
│   ├── pipeline.py             # 7-phase pipeline orchestrator
│   ├── repl.py                 # Main interactive REPL loop
│   ├── roles.py                # 18 AI persona system prompts
│   └── streaming.py            # Streaming LLM output, <think> tag handling
│
├── phases/                     # 7-phase planning pipeline
│   ├── base.py                 # Abstract BasePhase
│   ├── idea.py                 # Phase 1: Idea Refinement (Product Manager)
│   ├── competitors.py          # Phase 2: Competitor Analysis (Market Analyst)
│   ├── architecture.py         # Phase 3: Architecture Design (Software Architect)
│   ├── models.py               # Phase 4: Data Models (Database Architect)
│   ├── api_design.py           # Phase 5: API Design (Backend Engineer)
│   ├── frontend.py             # Phase 6: Frontend Planning (Frontend Engineer)
│   └── codegen.py              # Phase 7: Code Generation (auto-selected Developer)
│
├── tools/                      # External integrations
│   ├── file_tools.py           # File read/write/diff/backup, path safety
│   ├── shell_tools.py          # Shell execution with 3 confirmation modes
│   ├── web_tools.py            # DuckDuckGo search + URL fetch + HTML parsing
│   └── tech_tools.py           # Package version lookup (PyPI/npm/GitHub)
│
├── commands/                   # Slash command system
│   ├── registry.py             # Command registration & dispatch
│   └── handlers.py             # 24 command implementations
│
├── tests/                      # pytest unit tests (43 tests)
│   ├── test_config.py
│   ├── test_file_tools.py
│   └── test_tech_tools.py
│
├── specs/                      # Pipeline output specs (user-generated, gitignored)
├── process/                    # Phase state per project (gitignored)
└── output/                     # Generated code output (gitignored)
```

---

## Tech Stack

| Category | Library | Version |
|---|---|---|
| LLM | langchain-ollama | 0.2.0+ |
| LLM Core | langchain-core | 0.3.0+ |
| Terminal UI | rich | 13.0.0+ |
| Web Search | ddgs (DuckDuckGo) | 9.0.0+ |
| HTTP | httpx | 0.27.0+ |
| HTML parsing | beautifulsoup4 | 4.12.0+ |
| Config | pyyaml | 6.0+ |
| Vector DB | chromadb | 0.5.0+ |
| Version parsing | packaging | 24.0+ |
| File patterns | pathspec | 0.12.0+ |
| Testing | pytest | 8.0+ |
| External req | Ollama (local LLM server) | — |

---

## Data Flow

```
cli.py
  └── run_repl() [core/repl.py]
        ├── /command  →  commands/registry.py  →  commands/handlers.py
        └── natural language  →  core/streaming.py  →  Ollama LLM

/project "idea"
  └── core/pipeline.py
        ├── Phase 1-6: web research (cached ChromaDB) + AI discussion + write to process/<project>/
        └── Phase 7: codegen file-by-file → workspace, tracked in codegen_state.json
```

---

## Configuration

- File: `~/.aicoder/config.yaml` (auto-created on first run)
- Default model: `qwen3.5:2b` via Ollama at `http://localhost:11434`

```yaml
model:
  provider: ollama
  name: qwen3.5:2b
  base_url: http://localhost:11434
  temperature: 0.3
  temperature_precise: 0.1
  context_length: 16384

shell:
  confirmation: always       # always | never | smart

files:
  confirmation: auto         # auto | always | never
  backup: true

workspace:
  auto_scan: true
  ignore_dirs: [.git, venv, node_modules, ...]
  max_file_size_kb: 200
  max_context_files: 30

knowledge:
  embedding_model: "nomic-embed-text-v2-moe"
```

---

## Key Architectural Decisions

1. **Local-only** — No cloud, no API keys. All user data stored in `~/.aicoder/`.
2. **Role-based AI personas** — Each pipeline phase switches to a tailored system prompt (`core/roles.py`).
3. **Vector RAG** — ChromaDB + Ollama embeddings for semantic search over cached web results. Reuses research across phases.
4. **Resumable pipeline** — State saved after each phase in `process/<project>/state.json`. Users can exit and resume.
5. **File-by-file codegen** — Phase 7 writes one file at a time, progress tracked in `codegen_state.json`.
6. **Workspace context injection** — `core/context.py` auto-scans project and injects a summary into every LLM prompt.
7. **Auto-summarization** — Memory auto-compresses older messages when history exceeds 20K chars.
8. **Path safety** — All file ops resolved against workspace root; escape checks prevent directory traversal.

---

## LLM Integration Pattern

Uses LangChain message types: `HumanMessage`, `AIMessage`, `SystemMessage`.

Two lazy-loaded `ChatOllama` instances per session:
- `temperature=0.3` — conversational default
- `temperature=0.1` — precise mode for specs and code

`<think>` tags in model output are intercepted and hidden from the terminal display.

---

## Data Formats

**Pipeline state** (`process/<project>/state.json`):
```json
{
  "project_name": "my-app",
  "idea": "...",
  "current_phase": 3,
  "phases": {
    "1": {"status": "done", "name": "Idea Refinement"},
    "3": {"status": "in_progress", "name": "Architecture Design"}
  }
}
```

**Codegen state** (`process/<project>/codegen_state.json`):
```json
{
  "generated_files": ["src/main.py", "src/models.py"],
  "total_files": 12,
  "complete": false
}
```

**Session memory** (`~/.aicoder/memory/<project_id>/session.json`):
```json
{
  "project_path": "/path/to/project",
  "saved_at": "2025-04-11T...",
  "history": [
    {"role": "human", "content": "..."},
    {"role": "ai", "content": "..."}
  ]
}
```

**File block format** (used by codegen parser):
```
===FILE: path/to/file.ext===
<complete file content>
===END===

===SUMMARY===
What changed: description
===END===
```

---

## Installation

```bash
# Dev install (editable)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e .
aicoder

# Standalone binary (no Python needed for end users)
pip install pyinstaller
pyinstaller --name aicoder --onefile cli.py
# dist/aicoder  (or dist/aicoder.exe on Windows)
```

Requires Ollama running locally: `ollama pull qwen3.5:2b`
