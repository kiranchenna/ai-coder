# AICoder ✨

> An AI-powered coding assistant CLI — local, offline, and professional.
> Runs entirely on your machine via Ollama. No API keys. No cloud.

---

## What It Does

AICoder is an interactive terminal assistant that understands your project, researches the web for up-to-date information, and takes real actions — planning products, designing architectures, generating full codebases, fixing bugs, and running commands — all from a natural language REPL.

**Key capabilities:**

- 🚀 **Professional project pipeline** — 7-phase planning: Idea → Competitors → Architecture → Models → API → Frontend → Code
- 💬 **Chat with your codebase** — ask questions, get explanations, request changes
- 🔧 **Fix bugs & improve code** — AI reads your files, diffs changes, writes them back
- 🔍 **Web research with RAG** — DuckDuckGo search + vector knowledge cache (ChromaDB)
- 🧠 **Role-based AI** — switches between PM, Architect, DBA, React/Python/Rust devs, etc.
- 📦 **Version & dependency checks** — PyPI, npm, GitHub latest releases
- 🖥 **Shell execution** — configurable confirmation modes
- 💾 **Persistent memory** — conversation history saved per project, auto-summarized

---

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** — installed and running locally

```bash
# Pull the default model
ollama pull qwen3.5:2b

# Recommended: dedicated embedding model (faster RAG)
ollama pull nomic-embed-text-v2-moe
```

---

## Installation

### Option 1: Developer Install (Source Code)
If you are cloning the repo to modify the code:

```bash
# 1. Clone
git clone <repository-url>
cd ai-coder

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 3. Install globally as a CLI command
pip install -e .
```

### Option 2: Standalone Executable (No source code / No Python required)
You can distribute AICoder as a completely standalone, closed-source `.exe` (or binary on Mac/Linux). Users just download the file and run it.

**To build the standalone executable (for the project creator):**
```bash
# 1. Install PyInstaller
pip install pyinstaller

# 2. Compile AICoder into a single executable
pyinstaller --name aicoder --onefile cli.py

# Your standalone tool will be generated at: dist/aicoder.exe
```
You can now share `dist/aicoder.exe` with anyone. They do not need Python installed to use it.

---

## Quick Start

```bash
# Start in any project folder
cd my-project
aicoder

# Plan a new product from scratch
/project "developer portfolio with GitHub sync and blog"

# Resume after closing — auto-detected
/project

# Or point at a different workspace
aicoder --workspace ./another-project
```

---

## The Project Planning Pipeline (`/project`)

AICoder's flagship feature: a **7-phase professional pipeline** that mirrors how real teams plan and build software. At each phase, the AI takes on a specific expert role and researches current best practices from the web before discussing with you.

```
/project "your idea"
```

### Phases

| # | Phase | AI Role | Output |
|---|---|---|---|
| 1 | **Idea Refinement** | Product Manager | Product brief + finalised feature list |
| 2 | **Competitor Analysis** | Market Analyst | Comparison table + gap analysis + differentiators |
| 3 | **Architecture Design** | Software Architect | ADR + tech stack table + ASCII diagram + cost estimate |
| 4 | **Data Models** | Database Architect | Entity schemas + ERD + index strategy |
| 5 | **API Design** | Backend Engineer | Endpoint table + auth strategy + error format |
| 6 | **Frontend Planning** | Frontend Engineer + UX | Page matrix + routing + user flows |
| 7 | **Code Generation** | Developer (auto-chosen) | Complete codebase, file-by-file |

### Phase behaviour

- **Skippable** — prompted between each phase; skip any that don't apply
- **Interactive** — full back-and-forth conversation with the AI expert until you're satisfied
- **Resumable** — exit anytime; `/project` auto-detects and continues where you left off
- **File-by-file codegen** — Phase 7 writes each file individually and tracks progress, so it can resume if interrupted

### Resumability

```bash
# Just type /project — it finds your in-progress session automatically
/project

# Explicitly resume
/project resume

# Resume a specific project
/project resume my-task-manager

# List all in-progress projects
/project list
```

### Where data is saved

```
your-workspace/
├── process/<name>/            ← per-phase discussion + summaries
│   ├── state.json             ← pipeline progress (resumable)
│   ├── 01_idea_refinement.md
│   ├── 02_competitor_analysis.md
│   ├── 03_architecture_design.md
│   ├── 04_data_models.md
│   ├── 05_api_design.md
│   ├── 06_frontend_planning.md
│   ├── 07_codegen.md
│   └── codegen_state.json    ← tracks generated files (resumable)
├── specs/<name>.md            ← combined running spec (all phases)
└── output/<name>/             ← generated code
```

> You can commit `process/` and `specs/` to your repo — they're permanent planning documents.

---

## All Slash Commands

Type `/help` inside the REPL or use any of these:

### 🚀 Project Pipeline
| Command | Description |
|---|---|
| `/project "idea"` | Start the 7-phase professional planning pipeline |
| `/project` | Auto-resume in-progress session (or show help) |
| `/project resume [name]` | Explicitly resume a paused pipeline |
| `/project list` | List all in-progress projects in this workspace |

### 🔧 Code Operations
| Command | Description |
|---|---|
| `/fix [file] [desc]` | AI fixes bugs — shows diff before writing |
| `/improve [file] [what]` | Improve code quality, types, error handling |
| `/review [file]` | Code review with 🔴/🟡/🟢 severity ratings |
| `/explain [file] [what]` | Explain a file or concept |

### 🏗 Project Scaffolding
| Command | Description |
|---|---|
| `/new <name> [desc]` | Quick-scaffold a new project directory |
| `/test [analyse\|cmd]` | Run tests; `analyse` asks AI to explain failures |

### 🔍 Research & Info
| Command | Description |
|---|---|
| `/research <topic>` | Web-search + AI synthesis |
| `/stack <app type>` | Recommend tech stack with current best practices |
| `/docs <library>` | Fetch and summarize official documentation |
| `/versions <pkg> [pkg2]` | Latest version from PyPI or npm |
| `/checkdeps [file]` | Scan `requirements.txt` for outdated packages |
| `/github <owner/repo>` | Latest GitHub release; `search <q>` for repo search |

### 🗂 Context & Files
| Command | Description |
|---|---|
| `/load <file> [file2]` | Pin specific files into AI conversation context |
| `/context [--refresh]` | Show what the AI knows about your workspace |
| `/git [diff\|log\|context]` | View git status/diff/log; `context` injects into AI |
| `/knowledge` | Show vector RAG cache stats |
| `/knowledge search <q>` | Search the cached knowledge base |
| `/knowledge learn <q>` | Learn a new topic or fetch a specific URL into cache |
| `/knowledge clear` | Clear the RAG cache |

### 🛠 Shell & Settings
| Command | Description |
|---|---|
| `/run <command>` | Run a shell command in the workspace |
| `/shell-mode [always\|never\|smart]` | Change shell confirmation mode |
| `/diff-mode [always\|auto\|never]` | Change how AI file changes are reviewed |
| `/memory [clear]` | View or wipe persistent session memory |
| `/config` | Show current configuration |
| `/help` | List all commands |

> You can also **type anything in plain English** — the AI understands your project and can write or fix code directly.

---

## AI Roles

The pipeline switches between expert roles automatically. Each has a tailored system prompt:

| Role | Used in |
|---|---|
| Product Manager | Phase 1 — Idea refinement |
| Market Analyst | Phase 2 — Competitor research |
| Software Architect | Phase 3 — Architecture design |
| Database Architect (DBA) | Phase 4 — Data models |
| Backend Engineer | Phase 5 — API design |
| Frontend Engineer + UX Designer | Phase 6 — Frontend planning |
| Python / React / Vue / Rust / Java / Android / iOS dev | Phase 7 — Auto-selected from tech stack |

---

## RAG Knowledge Base

Web research is automatically cached in a local **ChromaDB vector store** (`~/.aicoder/knowledge/chroma/`). This means:

- Searches done in Phase 2 are instantly available in Phase 5
- Re-running `/project` reuses cached research (no redundant web requests)
- Embeddings use your existing Ollama model — no extra download needed
- TTLs: search results (6h), docs pages (48h)

```bash
# Inside aicoder
/knowledge                  # stats
/knowledge learn "https://vuejs.org/guide" # learn from specific URL
/knowledge learn "React 19" # web search and learn about topic
/knowledge search "fastapi" # semantic search across cached content
/knowledge clear            # wipe cache
```

---

## Shell Confirmation Modes

Toggle with `/shell-mode <mode>`:

| Mode | Behaviour |
|---|---|
| `always` | Asks `[y/N]` before every command *(default — safest)* |
| `smart` | Auto-approves safe commands; asks for destructive ones (`rm`, `git reset --hard`) |
| `never` | Auto-runs everything without asking |

---

## Diff Review Modes

Toggle with `/diff-mode <mode>`:

| Mode | Behaviour |
|---|---|
| `auto` | Shows coloured diff then applies automatically *(default)* |
| `always` | Shows diff and asks `[y/N]` before each file |
| `never` | Writes immediately, no preview |

Files overwritten by AI are backed up as `filename.ext.bak` automatically.

---

## Configuration

Auto-created at `~/.aicoder/config.yaml` on first run:

```yaml
model:
  provider: ollama
  name: qwen3.5:2b               # any model you have pulled
  base_url: http://localhost:11434
  temperature: 0.3
  temperature_precise: 0.1

shell:
  confirmation: always           # always | never | smart

files:
  confirmation: auto             # always | auto | never
  backup: true

workspace:
  auto_scan: true
  ignore_dirs: [.git, venv, __pycache__, node_modules, dist, build]
  max_file_size_kb: 200
  max_context_files: 30

knowledge:
  # Dedicated embedding model for the vector RAG knowledge base.
  # nomic-embed-text-v2-moe: MoE architecture, multilingual, 523 MB — best quality.
  # Set to "" to fall back to the main chat model (no extra pull needed).
  embedding_model: "nomic-embed-text-v2-moe"

memory:
  enabled: true
  max_history: 50
```

**Good model choices** (`ollama pull <model>`):
- `qwen3.5:2b` — fastest, lightest, great for most tasks *(default)*
- `qwen3.5:4b` — larger, slightly stronger general reasoning
- `qwen2.5-coder:7b` — stronger coding focus
- `qwen2.5-coder:14b` — best quality, needs more RAM

---

## Project Structure

```
ai-coder/
├── cli.py                      # Entry point (aicoder command)
├── aicoder_cli/                # Installable package wrapper
├── core/
│   ├── config.py               # Config management (~/.aicoder/config.yaml)
│   ├── context.py              # Workspace scanner & project summarizer
│   ├── knowledge.py            # Vector RAG (ChromaDB + Ollama embeddings)
│   ├── memory.py               # Persistent memory + auto-summarization
│   ├── pipeline.py             # 7-phase pipeline orchestrator
│   ├── roles.py                # AI role definitions (18 personas)
│   ├── repl.py                 # Main REPL loop
│   └── streaming.py            # Live LLM token streaming
├── phases/
│   ├── base.py                 # BasePhase (discussion, research, file I/O)
│   ├── idea.py                 # Phase 1 — Idea Refinement
│   ├── competitors.py          # Phase 2 — Competitor Analysis
│   ├── architecture.py         # Phase 3 — Architecture Design
│   ├── models.py               # Phase 4 — Data Models
│   ├── api_design.py           # Phase 5 — API Design
│   ├── frontend.py             # Phase 6 — Frontend Planning
│   └── codegen.py              # Phase 7 — Code Generation
├── tools/
│   ├── file_tools.py           # Read / write / diff / backup files
│   ├── shell_tools.py          # Shell execution with confirmation
│   ├── web_tools.py            # DuckDuckGo search + URL fetching
│   └── tech_tools.py           # PyPI / npm / GitHub version lookup
├── commands/
│   ├── registry.py             # Slash command registry
│   └── handlers.py             # All 24 command implementations
├── tests/                      # 43 unit tests
├── specs/                      # Pipeline spec documents
├── process/                    # Pipeline phase outputs (per project)
├── output/                     # Generated code (per project)
├── .aicoderignore              # (optional) per-project ignore rules
└── pyproject.toml
```

---

## .aicoderignore

Exclude files/folders from AI context scans using gitignore syntax:

```gitignore
# .aicoderignore
dist/
build/
*.log
.env
secrets/
large_dataset/
```

---

## Memory & Sessions

- **Per-project memory** stored in `~/.aicoder/memory/` — each directory has its own history
- **Auto-saved** after every conversation turn
- **Auto-resumed** when you open `aicoder` in the same directory
- **Auto-summarized** — when history grows > 20K characters, older messages are compressed automatically to stay within the model's context window
- Clear with `/memory clear`

---

## Running Tests

```bash
# Run the full test suite
pytest tests/ -v

# Or from inside aicoder
/test
/test analyse    # AI explains any failures
```

---

## License

MIT

## Contributing

Pull requests welcome. Please open an issue first for significant changes.
