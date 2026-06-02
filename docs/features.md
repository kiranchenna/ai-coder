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
| `search_code(query, path=".")` | Grep file contents (`file:line: text`) |
| `read_file(path)` | Read a file's full text |

### Editing & execution
| Tool | Description |
|---|---|
| `write_file(path, content)` | Create or overwrite a file (diff + confirm) |
| `edit_file(path, old_string, new_string)` | Replace an exact, unique snippet (diff + confirm) |
| `run_shell(command)` | Run a shell command (confirmation per config) |
| `run_tests()` | Auto-detect and run the test suite, report pass/fail |

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
| `plan <goal>` | Decompose a goal into an ordered, resumable task list and build it |
| `resume` | Continue an in-progress plan |
| `/model [name]` | Show or switch the model for this session |
| `/tools` | List the agent's tools |
| `/memory` | Show what's remembered about this project |
| `/knowledge` | RAG: `/knowledge learn <topic\|URL>` researches & caches; bare shows stats; `/knowledge clear` clears this project's docs, `/knowledge clear all` wipes everything |
| `/clear` | Forget the conversation (keeps saved memory) |
| `/help` | Show commands |
| `exit` / `quit` | Leave the session |

---

## The agentic loop

Per user turn (`AgentSession.send`):

1. The model is invoked with the conversation + bound tools.
2. **Native tool calls** are executed; each result is fed back as a tool message.
3. **Fallback:** if the model emitted tool calls as JSON *text* instead of
   natively (common with local models like `qwen2.5-coder`), they are recovered
   (`core/model.extract_text_tool_calls`), filtered to known tools, executed, and
   the results fed back.
4. When the model returns a plain answer (no tool calls), it is shown and the
   turn ends. A step cap (12) bounds runaway loops.

---

## Planning large tasks

`plan <goal>` (`agent/planner.py`):

- Asks the model for an ordered JSON task list, grounded in any ingested
  document via RAG.
- Persists the plan to `~/.aicoder/memory/<project_id>/plan.json`.
- Executes each task through the agent, saving status after every step, and
  pausing for confirmation between tasks.
- **Resumable** — quit anytime; `resume` continues from the first pending task.

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
- **Embeddings:** Ollama `nomic-embed-text-v2-moe`.
- **Chunking:** real overlapping chunks (≈1200 chars, 150 overlap).
- **Relevance cutoff:** cosine-distance threshold so unrelated queries return
  nothing instead of the nearest irrelevant chunk.
- **Document loaders:** PDF (pypdf), Word (python-docx, incl. tables),
  Markdown/txt/rst, HTML.
- **Scoping:** web research is **global** (shared cache across projects);
  ingested documents are **per-project** (tagged by workspace, so a PRD from one
  project doesn't surface in another). Recall returns this project's docs + the
  global web cache.

### Test detection (`core/project.py`)
- Auto-detects the test command from marker files: pytest, npm/yarn/pnpm test,
  cargo, go, make, maven, gradle. For pytest it prefers the project's `.venv`.

### Persistent memory (`memory/project.py`)
- **Storage:** `~/.aicoder/memory/<project_id>/project_memory.json`, keyed by
  workspace path.
- Durable categorized facts (decision/convention/fact/todo), idempotent add,
  auto-loaded into the system prompt at session start.

### Workspace context (`core/context.py`)
- Scans the tree (depth 3), detects languages, and injects a compact repo
  overview into the system prompt (no file-content dumps — the agent reads on
  demand).
