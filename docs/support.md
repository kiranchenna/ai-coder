# AICoder — Support & FAQ

Practical help for running AICoder. For *how it works* see
[features.md](features.md); for *how it's built* see
[architecture.md](architecture.md).

---

## Getting started checklist

If anything misbehaves, confirm these first — most issues are one of them:

1. **LM Studio is installed with its local server running** — Developer tab →
   Start Server. Check: `curl http://localhost:1234/v1/models`.
2. **The chat model is downloaded and loaded** — `lms load <model-id>` (or
   from LM Studio's UI), or use the in-session `/model` picker.
3. **The embedding model is downloaded** (only for web research / documents)
   — grab it in LM Studio's own model search, e.g.
   `nomic-ai/nomic-embed-text-v1.5-GGUF`.
4. **Tool calling works on your model** — `aicoder --selftest`.
5. **You're in the right project** — `aicoder` runs in the current directory;
   use `aicoder --workspace ./path` to point elsewhere.

See your active settings any time with `aicoder --config`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| **"Couldn't reach LM Studio"** | server not running, or model not loaded | start it (Developer tab → Start Server), `lms load <model-id>`; confirm `base_url` in `aicoder --config` |
| **`--selftest` says the model can't call tools** | weak/unsupported model | switch to `qwen2.5-coder-7b-instruct` or larger (`aicoder --model <name>`) |
| **Web research / `read_document` "couldn't ingest"** | embedding model missing | download one in LM Studio and set `knowledge.embedding_model`; RAG warns once at search time if it fails |
| **`rag_search` returns nothing** | empty/irrelevant cache, or relevance cutoff | research the topic first (`/knowledge learn <topic>`); the cutoff drops unrelated chunks by design |
| **Edits get declined / the agent loops** | small model struggling | rephrase the task, narrow it, or switch to a larger model |
| **"Reached the step limit for this turn"** | task needed >12 tool steps | ask it to continue, or break the task up with `/plan <goal>` |
| **`/init` stops after narrating what it'll do next, without doing it** | small model narrated instead of calling the tool | just say "continue" — it's a normal conversational session, not a special mode; a larger model tends to follow through more reliably |
| **MCP servers don't load** | extra not installed / bad config | `pip install "ai-coder[mcp]"`; check the server `command`/`args` |
| **"langchain-openai isn't installed"** | broken environment | `pip install langchain-openai` (it's a core dependency, so this should already be present) |
| **Generated project won't run after `/dev build`** | the verify→fix loop hit its 3-round cap | read the printed failure, fix manually, or re-run `/dev build` / `/dev revisit <phase>` |
| **Design feels generic / too slow** | wrong `devmode.profile` | speed: set `profile: fast`; depth: `thorough` (+ a `judge_model`) |

---

## Frequently asked questions

**Is my code sent anywhere?**
No. Inference and embeddings run locally via LM Studio; all data lives under
`~/.aicoder/`. The only outbound traffic is when *you* invoke web research
(`research` / `fetch_url` / `/knowledge learn`), which queries DuckDuckGo and
fetches the pages you asked for. Nothing else leaves your machine.

**Which model should I use?**
`qwen2.5-coder-7b-instruct` is the default and the sweet spot for ~16 GB RAM.
Smaller (Qwen2.5-Coder-3B) is faster but weaker at multi-step work; larger
(Qwen2.5-Coder-14B, Qwen3-Coder-30B) is stronger and needs more memory. The
model and its context share memory, so a bigger context window costs RAM too.
On Apple Silicon, prefer an MLX build over GGUF where available — measurably
faster for the same model/quant. Switch anytime with `/model` — type it alone
for an interactive picker listing every model already downloaded in LM Studio
(current one marked); grabbing a new model is a manual step in LM Studio
itself. Either way, the choice is saved as your default going forward.

**What if I want to use a different server than LM Studio?**
Set `model.base_url` in `config.yaml` to point at a different local server
sized to your own hardware (vLLM for a heavy GPU, llama.cpp server for
something lighter, text-generation-webui, LocalAI, ...) or a hosted API with
your own key (OpenAI, OpenRouter, Groq, Together, ...) — anything that speaks
the OpenAI chat-completions protocol, which is what AICoder always talks
under the hood (`model.provider: openai_compatible`). The rich `/model`
picker and the `lms`-based load/unload behavior are specific to LM Studio's
own APIs (detected by `base_url` matching its default), so they're skipped
for other endpoints; you'll see your current model/endpoint instead, and
`/model <name>` still switches the model id.

**How good is it, really?**
It's a strong *supervised* assistant, not an autonomous senior engineer. A local
7B writes a weak first draft and can't reason through every subtle case — review
its diffs and lean on the verify loop. Developer Mode compensates with
engineering (reflection, checklists, decomposition, review); those levers are
measured in [`evals/`](../evals/README.md). The design artifacts it produces are
valuable on their own, regardless of model strength.

**Why is `/dev build` / `/develop` slow?**
Each quality lever adds model calls, and local inference is the bottleneck. Use
`devmode.profile: fast` for speed (reflect only — it carries most of the quality
gain) or `balanced` (the default). `thorough` is the slowest and only adds
`best_of`, which needs a stronger `judge_model` to pay off.

**Can it work on my existing repo (not just new projects)?**
Yes. The agent works on any repo. Developer Mode is brownfield-aware: every phase
is grounded in your codebase and the Conventions phase infers your existing style
so generated code matches it.

**What's the difference between `/plan` and `/develop`?**
`/plan <goal>` decomposes a goal into a resumable task list and builds it directly
— good for a contained feature. `/develop` runs the full role-driven SDLC (design
in editable artifacts, then `/dev build`) — good for designing a whole application
with you in control of every decision.

**How do I make it stop asking before every command / edit?**
Shell: `--shell-mode smart` (asks only for destructive commands) or `never`.
Files: set `files.confirmation: auto` (default — shows the diff, applies) or
`never`. `always` is the safest. See "Safety" below.

**My terminal looks empty / I can't scroll up to see what happened after exiting — where did it go?**
AICoder runs full-screen on a real terminal (the same "alternate screen"
mechanism `vim`/`htop`/Claude Code use) and hands your terminal back exactly as
it was on exit — the session was never added to your normal scrollback in the
first place, so there's nothing to scroll back to. Run `/export` *before*
exiting to save a copy of the conversation to a file if you want to keep it.

**Where is everything stored?**
`~/.aicoder/config.yaml` (settings), `~/.aicoder/rag/chroma/` (knowledge base),
`~/.aicoder/memory/<project>/` (durable facts + plan state). Developer Mode design
artifacts live in your repo under `docs/dev/` and `AICODER.md`.

**How do I reset things?**
Conversation: `/clear` (keeps saved memory). Knowledge base: `/knowledge clear`
(this project's docs) or `/knowledge clear all` (everything). Start a Developer
Mode design over: delete `docs/dev/`. Reset all settings: delete
`~/.aicoder/config.yaml` (recreated with defaults on next run).

---

## Safety model (what protects you)

You are the boundary. Two independent gates:

- **Shell** (`shell.confirmation`): `always` (ask every time — default),
  `smart` (auto-run safe commands, ask for destructive ones), `never`.
- **Files** (`files.confirmation`): `always` (ask), `auto` (show diff + apply —
  default), `never`. Overwrites are backed up as `*.bak` when `files.backup` is on.

Caveats worth knowing: `smart` mode is a best-effort heuristic, **not** a
security boundary — use `always` if you need a hard gate. File paths are
sandboxed to the workspace, but `run_shell` is not sandboxed, so your shell
approval is the real gate before commands run. Hooks run arbitrary commands you
configure — only add ones you trust.

---

## Known limitations

- **Local-model intelligence** — review its diffs; bigger models help.
- **Context window** — bounded by your hardware (default 16k); long sessions are
  compacted, very large tasks benefit from `plan`.
- **No image input** — local code models are text-only.
- **Tool-calling reliability** varies by model — `--selftest` checks it.
- **MCP** is stdio-only; **Windows** support is best-effort.
- **Eval figures** are small-n with a same-model judge — directional, not precise.

---

## Getting help / reporting an issue

When opening an issue, include:

1. `aicoder --version` and `aicoder --config` (redact anything private).
2. Your OS, LM Studio version, and the model in use.
3. The exact command/prompt and the full error or unexpected output.
4. Whether `aicoder --selftest` passes.

Contributions are welcome — please open an issue first for significant changes.
