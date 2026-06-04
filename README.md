# AICoder ✨

> A local, offline **agentic** coding assistant — plans, edits real code, runs commands, and verifies its own work.
> Runs entirely on your machine via Ollama. No API keys. No cloud.

---

## What It Does

AICoder is an interactive terminal agent that works on **your actual repository**. Describe a task in plain English (or point it at a PRD), and it uses tools to read and edit code, search the web for current information, run tests, and remember decisions across sessions.

**Key capabilities**

- 🤖 **Agentic loop** — the model calls tools (read/write/edit files, run shell, run tests) to get work done, not just chat.
- 🛠 **Works on any repo** — build new code, modify existing code, add features, fix bugs.
- 📄 **Document-driven** — ingest a PRD/TDD (PDF, Word, Markdown) and build from it.
- 🔍 **Stays current** — web research (DuckDuckGo) cached into a local vector store (RAG), so it isn't limited to the model's training cutoff.
- ✅ **Verifies its work** — auto-detects and runs your test suite (pytest, npm, cargo, go, …), reads failures, and fixes.
- 🧠 **Remembers** — durable per-project memory (decisions, conventions, TODOs) auto-loaded each session.
- 📋 **Plans big tasks** — decomposes a goal into an ordered, **resumable** task list and executes it step by step.
- 🔒 **You-in-the-loop** — file writes and shell commands are gated by configurable confirmation modes.

---

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** — installed and running locally

```bash
# Recommended models
ollama pull qwen2.5-coder:7b        # the agent driver (strong code + tool use)
ollama pull nomic-embed-text-v2-moe # embeddings for RAG
```

> On a 16 GB machine, `qwen2.5-coder:7b` is the sweet spot. A 2–4 B model (e.g. `qwen3.5:2b`) runs faster but is much weaker at multi-step work.

---

## Installation

### From PyPI

```bash
pip install ai-coder

aicoder --selftest               # confirm the model supports tool calling
aicoder                          # start the agent in the current directory
```

### From source (development)

```bash
git clone https://github.com/kiranchenna/ai-coder
cd ai-coder
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

---

## Usage

```bash
aicoder                          # work in the current directory
aicoder --workspace ./my-app     # point at another project
aicoder --model qwen2.5-coder:7b # override the model for this session
aicoder --selftest               # check tool calling, then exit
```

Inside the agent, just describe what you want:

```
> add input validation to the create_user endpoint and run the tests
> read the spec at docs/PRD.pdf and summarize what we need to build
> why is test_auth failing? find and fix it
```

### Multi-step builds

```
> plan build a FastAPI todo service from docs/PRD.md
```

`plan <goal>` decomposes the goal (grounded in any ingested document) into an ordered task list and executes each task with verification. It's **resumable** — quit anytime and type `resume` to continue.

### In-session commands

| Command | Description |
|---|---|
| `plan <goal>` | Decompose a goal into tasks and build it |
| `resume` | Continue an in-progress plan |
| `/model [name]` | Show or switch the model for this session |
| `/tools` | List the agent's tools |
| `/memory` | Show what's remembered about this project |
| `/clear` | Forget the conversation (keeps saved memory) |
| `/help` | Show commands |
| `exit` | Quit |

---

## Project instructions (`AICODER.md`)

Drop an **`AICODER.md`** file in your project root with rules the agent should
always follow — conventions, do's and don'ts, architecture notes. It's loaded
into the agent's context every session and takes precedence over its defaults.

```markdown
# AICODER.md
- Use snake_case and full type hints.
- Tests live in tests/ and run with pytest.
- Never edit anything under vendor/.
- Prefer pathlib over os.path.
```

A global `~/.aicoder/AICODER.md` is also loaded (applies to every project), with
the per-project file layered on top.

---

## The agent's tools

| Tool | Purpose |
|---|---|
| `list_files`, `find_files`, `find_symbol`, `search_code` | Explore the repo, jump to definitions, grep |
| `read_file`, `write_file`, `edit_file` | Read (paged for large files) and modify code (diff + confirmation; whitespace-tolerant edits) |
| `run_shell` | Run commands (with confirmation) |
| `run_tests`, `run_checks` | Auto-detect and run the test suite / linters + type checkers |
| `research`, `fetch_url`, `rag_search` | Web research + recall from the cached knowledge base |
| `read_document` | Ingest a PRD/TDD (PDF/docx/md) into the knowledge base |
| `git_status`, `git_diff`, `git_commit` | Review and commit changes |
| `remember`, `recall` | Save and retrieve durable project facts |

…plus any tools from configured **MCP servers** (see below).

> Robustness: some local models emit tool calls as JSON text rather than via native tool calling — the agent loop recovers and executes those too.

---

## MCP servers (optional)

Connect [MCP](https://modelcontextprotocol.io/) servers and their tools become
available to the agent alongside the built-ins — a database, GitHub, a browser,
your own server, anything that speaks the protocol.

```bash
pip install "ai-coder[mcp]"
```

```yaml
# ~/.aicoder/config.yaml
mcp:
  servers:
    filesystem:
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
    sqlite:
      command: uvx
      args: ["mcp-server-sqlite", "--db-path", "./app.db"]
```

Each server's tools appear (prefixed by server name, e.g. `filesystem__read_file`)
in `/tools`. Opt-in — nothing runs unless you configure servers.

---

## Configuration

Auto-created at `~/.aicoder/config.yaml` on first run:

```yaml
model:
  provider: ollama
  name: qwen2.5-coder:7b
  base_url: http://localhost:11434
  temperature: 0.3
  temperature_precise: 0.1
  context_length: 16384

shell:
  confirmation: always           # always | smart | never

files:
  confirmation: auto             # always (ask) | auto (apply, show diff) | never
  backup: true                   # write .bak before overwriting

knowledge:
  embedding_model: "nomic-embed-text-v2-moe"
```

### Confirmation modes (you-in-the-loop)

- **Shell** — `always` asks before every command, `smart` asks only for destructive ones, `never` auto-runs.
- **Files** — `always` asks before each write, `auto` shows the diff and applies, `never` writes silently. Overwritten files are backed up as `*.bak`.

---

## Where data lives

```
~/.aicoder/
├── config.yaml
├── rag/chroma/                  # cached web/doc knowledge (vector store)
└── memory/<project_id>/
    ├── project_memory.json      # durable facts the agent remembers
    └── plan.json                # in-progress task plan (resumable)
```

All per-project, keyed by the workspace path. Code is read from / written to your workspace; nothing leaves your machine.

---

## Project structure

```
ai-coder/
├── cli.py                  # entry point (aicoder)
├── core/
│   ├── config.py           # config (~/.aicoder/config.yaml)
│   ├── model.py            # ChatOllama factory + native tool binding
│   ├── context.py          # workspace scanner / repo overview
│   └── project.py          # test-command detection
├── agent/
│   ├── loop.py             # the agentic tool-calling loop + REPL
│   ├── tools.py            # the agent's tools
│   ├── planner.py          # decompose + run resumable task plans
│   └── prompts.py          # system prompt
├── rag/
│   ├── store.py            # ChromaDB vector store with chunking
│   └── ingest.py           # PDF/docx/md document loaders
├── memory/
│   └── project.py          # persistent per-project memory
├── tools/                  # file / shell / web helpers
└── tests/                  # unit tests
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Architecture & status

This is **AICoder v3** — an agentic rewrite. The original 7-phase planning
pipeline has been removed; everything now runs through the single agentic loop.

## License

MIT
