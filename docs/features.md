# AICoder — Features Reference

---

## 24 Slash Commands

All commands are registered in `commands/registry.py` and implemented in `commands/handlers.py`.

### Project Pipeline

| Command | Description |
|---|---|
| `/project "idea"` | Start the 7-phase planning pipeline with an idea |
| `/project` | Resume the most recent in-progress pipeline |
| `/project resume [name]` | Resume a specific named pipeline |
| `/project list` | List all in-progress projects |

### Code Operations

| Command | Description |
|---|---|
| `/fix [file] [description]` | AI fixes bugs; shows diff before applying |
| `/improve [file] [what]` | Improve code quality, types, error handling |
| `/review [file]` | Code review with severity ratings |
| `/explain [file] [what]` | Explain code or concepts |

### Project Scaffolding

| Command | Description |
|---|---|
| `/build` | Launch the original 4-phase wizard (`main.py`) |
| `/new <name> [desc]` | Quick-scaffold a new project directory |
| `/test [analyse\|cmd]` | Run tests or have AI analyze failures |

### Research & Info

| Command | Description |
|---|---|
| `/research <topic>` | Web search + AI synthesis |
| `/stack <app type>` | Recommend tech stack with best practices |
| `/docs <library>` | Fetch and summarize official documentation |
| `/versions <pkg> [pkg2]` | Latest version from PyPI or npm |
| `/checkdeps [file]` | Scan requirements.txt for outdated packages |
| `/github <owner/repo>` | Latest GitHub release info |

### Context & Files

| Command | Description |
|---|---|
| `/load <file> [file2]` | Pin files into the AI context for this session |
| `/context [--refresh]` | Show current workspace context summary |
| `/git [diff\|log\|context]` | Git diff, log, or git-aware context |
| `/knowledge [stats\|search <q>\|learn <q>\|clear]` | Manage the RAG knowledge base |

### Shell & Settings

| Command | Description |
|---|---|
| `/run <command>` | Execute a shell command (with confirmation) |
| `/shell-mode [always\|never\|smart]` | Set shell confirmation level |
| `/diff-mode [always\|auto\|never]` | Set file diff review mode |
| `/memory [clear]` | View or wipe session memory |
| `/config` | Show current configuration |
| `/help` | List all available commands |

---

## 7-Phase Planning Pipeline

Triggered by `/project "your idea"`. Each phase supports:
- Interactive back-and-forth discussion with the AI (type `done` to advance)
- Web research automatically fetched and cached in ChromaDB
- Resumable at any point — state saved to `process/<project>/state.json`
- All phase outputs combined into `specs/<project>.md`

| # | Phase | File | AI Role | Output |
|---|---|---|---|---|
| 1 | Idea Refinement | `phases/idea.py` | Product Manager | Product brief + finalized feature list |
| 2 | Competitor Analysis | `phases/competitors.py` | Market Analyst | Competitor comparison table + gap analysis |
| 3 | Architecture Design | `phases/architecture.py` | Software Architect | ADR + tech stack recommendation + cost estimate |
| 4 | Data Models | `phases/models.py` | Database Architect | Entity schemas + ERD diagram |
| 5 | API Design | `phases/api_design.py` | Backend Engineer | Endpoint definitions + auth strategy |
| 6 | Frontend Planning | `phases/frontend.py` | Frontend Engineer + UX | Page matrix + user flows |
| 7 | Code Generation | `phases/codegen.py` | Auto-selected Developer | Complete codebase, file-by-file |

Phase 7 auto-selects the developer role based on the chosen tech stack:
`python_developer`, `react_developer`, `vue_developer`, `rust_developer`, `java_developer`, `android_developer`, `ios_developer`, `fullstack_developer`

---

## 18 AI Roles (`core/roles.py`)

Each role has a unique system prompt with communication style, relevant frameworks, and output format expectations.

| Role Key | Persona | Focus |
|---|---|---|
| `product_manager` | Product Manager | Clarifying questions, user pain points |
| `market_analyst` | Market Analyst | Competitive intelligence, market gaps |
| `scrum_master` | Scrum Master | Actionable stories, sprint planning |
| `software_architect` | Software Architect | Scalable design, trade-off analysis |
| `database_architect` | Database Architect | Entity design, ERD, indexing strategy |
| `backend_engineer` | Backend Engineer | API design, auth, error handling |
| `python_developer` | Python Developer | FastAPI, Django, SQLAlchemy |
| `react_developer` | React Developer | Next.js, React 19, SPA patterns |
| `vue_developer` | Vue Developer | Vue 3, Nuxt, composition API |
| `rust_developer` | Rust Developer | Actix-web, async patterns |
| `java_developer` | Java Developer | Spring Boot, enterprise patterns |
| `android_developer` | Android Developer | Jetpack Compose, Material 3 |
| `ios_developer` | iOS Developer | Swift, SwiftUI, async/await |
| `fullstack_developer` | Fullstack Developer | Full stack patterns |
| `ux_designer` | UX Designer | User flows, wireframes, accessibility |

---

## Tool Capabilities

### File Operations (`tools/file_tools.py`)
- **Path safety:** All operations resolved against workspace root; escape checks prevent traversal
- **Encoding fallback:** utf-8 with `replace` error handling (never crashes on binary/non-UTF8 files)
- **Backup system:** Creates `.bak` copy before overwriting any file
- **Diff preview:** Rich syntax-highlighted, color-coded diffs shown before applying changes
- **File block parser:** Parses `===FILE:===` / `===END===` format from LLM output

### Shell Execution (`tools/shell_tools.py`)
- **`always`** — Asks `[y/N]` confirmation before every command (default)
- **`never`** — Auto-runs all commands without asking
- **`smart`** — Auto-runs safe commands; asks for destructive ones
- Destructive patterns detected: `rm`, `del`, `drop`, `truncate`, `-rf`, `--force`, `| rm`, etc.
- Timeout: 120 seconds per command

### Web Integration (`tools/web_tools.py`)
- DuckDuckGo search — no API key, up to 5 results, 10s timeout
- URL fetching — follows redirects, includes User-Agent header
- HTML parsing — strips script/style/nav/footer tags, extracts main content, truncates to 8000 chars
- Documentation fetching — tries multiple search results if first fails

### Package Version Lookup (`tools/tech_tools.py`)
- PyPI latest version
- npm latest version
- GitHub latest release tag

### RAG Knowledge Base (`core/knowledge.py`)
- **Storage:** ChromaDB at `~/.aicoder/knowledge/chroma/`
- **Embeddings:** Ollama `nomic-embed-text-v2-moe`
- **TTL:** Web search results: 6 hours | Documentation pages: 48 hours
- **Usage:** Automatically injects semantically relevant cached knowledge into LLM prompts
- **Management:** `/knowledge stats`, `/knowledge search <q>`, `/knowledge learn <q>`, `/knowledge clear`

### Memory System (`core/memory.py`)
- **Storage:** `~/.aicoder/memory/<project_id>/session.json`
- **Auto-load:** Previous conversation loaded when re-entering the same workspace
- **Auto-summarize:** Older messages compressed when history exceeds 20K chars
- **Limit:** Keeps last 50 messages (configurable)
- **Management:** `/memory` to view, `/memory clear` to wipe

### Workspace Context (`core/context.py`)
- Scans workspace directory tree (up to depth 3)
- Detects programming languages by file extension counts
- Reads key files: `package.json`, `pyproject.toml`, `README.md`, etc.
- Respects `.aicoderignore` (gitignore-style pattern file)
- Injects compact summary into every LLM prompt automatically
