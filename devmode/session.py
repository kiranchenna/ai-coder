"""
devmode/session.py — Developer Mode session engine
===================================================
Runs the role-driven SDLC phases (devmode/phases.py), each as a full
back-and-forth discussion that produces an editable artifact. State is persisted
to docs/dev/state.json so a design can be paused and resumed, and any phase can
be revisited to change a decision.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from core.console import SafeConsole
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule

from devmode.phases import PHASES, PHASES_BY_ID, PhaseSpec

console = SafeConsole()

_STATUS_ICON = {"done": "✅", "skipped": "⏭", "in_progress": "🔄", "pending": "○"}


def _decision_section(text: str) -> str:
    """Extract the decision body of an artifact, ignoring volatile header lines."""
    import re
    m = re.search(r"## Decision\s*\n+(.*?)\n+---", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Conventions/other format has no '## Decision' block: drop the leading title
    # and the generated/decided timestamp line so the body compares stably across
    # regenerations (otherwise the changing timestamp looks like a content change).
    body = re.sub(r"(?m)\A#.*$\n?", "", text, count=1)
    body = re.sub(r"(?m)^_(Decided|Generated)[^\n]*_\s*$\n?", "", body)
    return body.strip()


def _stream(messages, precise: bool = False, model: str | None = None) -> str:
    """Stream a conversational model response to the terminal; return the text."""
    from core.model import get_chat_model

    llm = get_chat_model(precise=precise, model=model)
    acc = ""
    console.print()
    try:
        for chunk in llm.stream(messages):
            piece = chunk.content
            if isinstance(piece, str) and piece:
                acc += piece
                sys.stdout.write(piece)
                sys.stdout.flush()
    except Exception as e:  # noqa: BLE001
        console.print(f"\n[red]⚠ Model error: {e}[/red]")
        return acc
    sys.stdout.write("\n")
    return acc


def _critic_stream(messages, precise: bool = True) -> str:
    """Stream from the optional stronger 'judge' model for high-leverage critic
    steps (best-of judging, consistency, review); falls back to the main model."""
    from core.config import get_config
    judge = (get_config().get("devmode", "judge_model", default="") or "").strip()
    if judge:
        return _stream(messages, precise=precise, model=judge)
    return _stream(messages, precise=precise)


class DevSession:
    """Orchestrates the SDLC design phases for a project."""

    def __init__(self, workspace: Path, idea: str | None = None, auto: bool = False):
        self.workspace = workspace.resolve()
        self.dir = self.workspace / "docs" / "dev"
        self.state_file = self.dir / "state.json"
        self.auto = auto  # fast mode: the roles decide themselves, no back-and-forth
        self.state = self._load_or_init(idea)

    # ── State ──────────────────────────────────────────────────────────────────

    def _load_or_init(self, idea: str | None) -> dict:
        if self.state_file.exists():
            try:
                state = json.loads(self.state_file.read_text(encoding="utf-8"))
                if idea:
                    console.print("[dim]A Developer Mode session already exists here — "
                                  "resuming it. (Delete docs/dev to start over.)[/dim]")
                return state
            except Exception:
                pass
        return {
            "idea": idea or "",
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "phases": {p.id: {"title": p.title, "status": "pending"} for p in PHASES},
        }

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        self.state_file.write_text(json.dumps(self.state, indent=2, ensure_ascii=False),
                                   encoding="utf-8")

    def _set_status(self, phase_id: str, status: str) -> None:
        self.state["phases"][phase_id]["status"] = status
        self._save()

    def has_session(self) -> bool:
        return self.state_file.exists()

    # ── Artifacts ──────────────────────────────────────────────────────────────

    def _artifact_path(self, spec: PhaseSpec) -> Path:
        return (self.workspace / spec.filename) if spec.target == "conventions" \
            else (self.dir / spec.filename)

    def _prior_artifacts(self) -> str:
        """Concatenate completed decision artifacts in full (used by the reviewer,
        which wants the actual decisions, not summaries; it caps the length itself)."""
        parts = []
        for p in PHASES:
            if p.target != "doc" or p.kind == "review":
                continue
            path = self.dir / p.filename
            if path.exists():
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
        return "\n\n---\n\n".join(parts)

    def _prior_grounding(self, budget: int = 8000) -> str:
        """Compact grounding for the NEXT phase's discussion.

        Earlier phases are summarized to their cached digests (commitments and
        constraints) rather than concatenated in full. Chaining ~14 raw artifacts
        would blow past the model's context length by the later phases, and local
        servers typically truncate from the *front* — silently evicting the
        earliest, most foundational decisions (requirements, security). Digests
        keep every prior phase represented within a bounded budget. The digests
        are already computed (and cached in state) by
        the per-phase consistency check, so this is usually free.
        """
        pairs: list[tuple[str, str]] = []
        for p in PHASES:
            if p.target != "doc" or p.kind == "review":
                continue
            path = self.dir / p.filename
            if not path.exists():
                continue
            digest = self._phase_digest(p.id)
            if not digest:  # consistency checks off → fall back to the decision body
                digest = _decision_section(path.read_text(encoding="utf-8", errors="replace"))
            if digest.strip():
                pairs.append((p.title, digest.strip()))
        if not pairs:
            return ""
        # Bound the total: if the digests are large, trim each phase evenly so no
        # single phase (and never the earliest) is dropped wholesale.
        per_phase = max(600, budget // len(pairs))
        parts = [f"## {title}\n{text[:per_phase]}" for title, text in pairs]
        return "\n\n".join(parts)

    def _write_artifact(self, spec: PhaseSpec, decision: str, transcript: str) -> Path:
        path = self._artifact_path(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        if spec.target == "conventions":
            content = (f"# AICODER.md — project conventions\n_Generated by Developer Mode: {ts}_\n\n"
                       f"{decision}\n")
        else:
            content = (f"# {spec.title}\n_Decided: {ts}_\n\n## Decision\n\n{decision}\n\n"
                       f"---\n\n<details><summary>Discussion</summary>\n\n{transcript}\n\n</details>\n")
        path.write_text(content, encoding="utf-8")
        return path

    # ── Discussion ─────────────────────────────────────────────────────────────

    # ── Existing-codebase awareness (brownfield) ───────────────────────────────

    _CODE_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
                  ".rb", ".vue", ".php", ".cs", ".cpp", ".c", ".kt", ".swift"}
    _SKIP_NAMES = {"AICODER.md", "README.md", "CHANGELOG.md", "LICENSE"}

    def _iter_source_files(self):
        from core.config import get_config
        ignore = set(get_config().ignore_dirs)
        for p in sorted(self.workspace.rglob("*")):
            if not p.is_file() or any(part in ignore for part in p.parts):
                continue
            rel = p.relative_to(self.workspace)
            if rel.parts[:2] == ("docs", "dev") or p.name in self._SKIP_NAMES:
                continue
            if p.suffix.lower() in self._CODE_EXTS:
                yield p, rel

    def _has_existing_code(self) -> bool:
        return next(self._iter_source_files(), None) is not None

    def _repo_overview(self) -> str:
        if getattr(self, "_overview_cache", None) is None:
            try:
                from core.context import WorkspaceContext
                self._overview_cache = WorkspaceContext(self.workspace).overview()
            except Exception:
                self._overview_cache = ""
        return self._overview_cache

    def _sample_code(self, max_files: int = 5, max_chars: int = 1200) -> str:
        samples, n = [], 0
        for p, rel in self._iter_source_files():
            if n >= max_files:
                break
            try:
                samples.append(f"=== {rel} ===\n" + p.read_text(encoding="utf-8", errors="replace")[:max_chars])
                n += 1
            except Exception:
                continue
        return "\n\n".join(samples)

    def _relevant_docs(self) -> str:
        try:
            from core.config import project_id
            from rag.store import KnowledgeBase
            hits = KnowledgeBase.get().search(self.state.get("idea", ""), n=3,
                                              max_distance=0.8, project=project_id(self.workspace))
            return ("\n\n".join(h["content"] for h in hits))[:2000] if hits else ""
        except Exception:
            return ""

    def _infer_conventions(self) -> str:
        samples = self._sample_code()
        if not samples.strip():
            return ""
        system = (
            "You are a Tech Lead. Infer the project's EXISTING coding conventions from its "
            "structure and sample files: folder layout, file naming, function/variable naming, "
            "formatting, error-handling/logging patterns, docstring/comment style, and test "
            "layout. Output a concise, concrete summary of the conventions already in use."
        )
        prompt = f"# Repo overview\n{self._repo_overview()}\n\n# Sample files\n{samples}\n\nInfer the conventions."
        try:
            from core.model import get_chat_model
            ai = get_chat_model(precise=True).invoke(
                [SystemMessage(content=system), HumanMessage(content=prompt)])
            return ai.content if isinstance(ai.content, str) else str(ai.content)
        except Exception:
            return ""

    def _system_prompt(self, spec: PhaseSpec, context: str, research: str,
                       repo: str = "", docs: str = "", seed: str = "") -> str:
        parts = [
            f"You are a {spec.role} helping an experienced developer design a software "
            f"project, one SDLC phase at a time. This phase: {spec.title}.",
            f"Project idea: {self.state.get('idea') or '(see prior decisions)'}",
            f"Your goal: {spec.goal} Specifically decide: {spec.focus}.",
            "Be concrete and SENIOR: address the genuinely hard, defining parts of THIS "
            "specific product — not generic boilerplate. Name specific technologies, "
            "protocols, patterns, and current stable versions. Cover ALL of the requested "
            "scope; do not silently defer the hard features. Propose a clear recommendation "
            "with brief rationale, and ask focused questions only where you truly need the "
            "developer's decision. They are experienced — keep it tight but deep.",
        ]
        if spec.must_cover:
            parts += ["", "You MUST explicitly address each of these (skip one only if it "
                      "genuinely does not apply, and say why):",
                      *(f"- {item}" for item in spec.must_cover)]
        if repo:
            parts += ["", "# Existing codebase — build on and MATCH this (brownfield)", repo]
        if docs:
            parts += ["", "# Relevant project documents (e.g. an imported PRD/spec)", docs]
        if context:
            parts += ["", "# Decisions already made in earlier phases", context]
        if seed:
            parts += ["", f"# Inferred starting point for {spec.title} (confirm or adjust)", seed]
        if research:
            parts += ["", "# Current information from the web (use for versions/best practices)", research]
        return "\n".join(parts)

    def _research_queries(self, spec: PhaseSpec) -> list[str]:
        """Ask the model what to look up for this phase (2-3 targeted queries)."""
        from core.model import balanced_json_arrays, get_chat_model

        prompt = (
            f"For designing the '{spec.title}' of this project, list 2-3 SPECIFIC things to "
            f"look up online right now — current stable versions, protocols, proven patterns, "
            f"or common pitfalls. Project: {self.state.get('idea', '')}.\n"
            f'Output ONLY a JSON array of web-search queries, e.g. ["X latest stable version", '
            f'"Y architecture best practices 2026"].'
        )
        try:
            raw = get_chat_model(precise=True).invoke([HumanMessage(content=prompt)]).content
            raw = raw if isinstance(raw, str) else str(raw)
        except Exception:
            raw = ""
        for span in balanced_json_arrays(raw):
            try:
                data = json.loads(span)
            except Exception:
                continue
            if isinstance(data, list):
                qs = [str(x).strip() for x in data if isinstance(x, str) and x.strip()][:3]
                if qs:
                    return qs
        return [f"{self.state.get('idea', '')} {spec.title} best practices current versions 2026"]

    def _research(self, spec: PhaseSpec) -> str:
        """Targeted multi-query web research → current facts in context."""
        try:
            from rag.research import research_topic
            parts = []
            for q in self._research_queries(spec):
                r = research_topic(q, project="", fetch_pages=1)
                if r.get("text"):
                    parts.append(f"### {q}\n{r['text'][:1500]}")
            return "\n\n".join(parts)[:4500]
        except Exception:
            return ""

    def _discuss(self, spec: PhaseSpec) -> tuple[str, list]:
        context = self._prior_grounding()
        research = self._research(spec) if spec.research else ""
        if spec.research:
            console.print("[dim]🌐 Researching current best practices…[/dim]")

        brownfield = self._has_existing_code()
        repo = self._repo_overview() if brownfield else ""
        docs = self._relevant_docs()
        seed = ""
        if spec.id == "conventions" and brownfield:
            console.print("[dim]🔎 Inferring conventions from your existing code…[/dim]")
            seed = self._infer_conventions()

        system = self._system_prompt(spec, context, research, repo, docs, seed)
        opening = (
            f"Design the {spec.title} fully now: make the senior decisions yourself for "
            f"{spec.focus}. Do NOT ask me questions — produce a complete, concrete proposal."
            if self.auto else
            f"Open the {spec.title} phase: give a concise first-draft proposal for "
            f"{spec.focus}, then ask me the key questions you need answered."
        )
        messages = [SystemMessage(content=system), HumanMessage(content=opening)]

        console.print(Rule(f"[bold magenta]{spec.role} — {spec.title}[/bold magenta]"))
        ai = _stream(messages)
        messages.append(AIMessage(content=ai))
        transcript = [f"**{spec.role}:** {ai}"]

        if self.auto:  # fast mode — accept the role's proposal, no back-and-forth
            console.print("[dim]⚡ fast mode — capturing the proposal.[/dim]")
            return "done", messages

        while True:
            console.print("\n[dim]Reply, or type: [bold]done[/bold] · [bold]skip[/bold] · "
                          "[bold]revise[/bold] · [bold]pause[/bold][/dim]")
            try:
                user = Prompt.ask(f"[bold yellow]{spec.id}[/bold yellow]").strip()
            except (EOFError, KeyboardInterrupt):
                return "pause", messages
            if not user:
                continue
            cmd = user.lower()
            if cmd in ("done", "d"):
                return "done", messages
            if cmd in ("skip", "s"):
                return "skip", messages
            if cmd in ("revise", "r"):
                return "revise", messages
            if cmd in ("pause", "p"):
                return "pause", messages
            messages.append(HumanMessage(content=user))
            transcript.append(f"**You:** {user}")
            ai = _stream(messages)
            messages.append(AIMessage(content=ai))
            transcript.append(f"**{spec.role}:** {ai}")

    _UNIT_DETAIL = {
        "entity": ("Design the **{item}** entity in full: every field with its type, the "
                   "primary key, foreign keys, indexes, relationships to other entities, and "
                   "constraints. If the project uses end-to-end encryption, store ciphertext "
                   "(never plaintext). Use ### for any sub-headings. Output clean Markdown for just this entity."),
        "resource": ("Design the **{item}** API in full: every endpoint (HTTP method + path), "
                     "request and response shapes with field types, status codes, the error "
                     "format, the auth required, and any realtime/websocket events. Output "
                     "clean Markdown for just this resource."),
        "component": ("Detail the **{item}** component in full: its responsibilities, the "
                      "chosen technology and current stable version, its interfaces and "
                      "dependencies on other components, the data it owns, and how it scales. "
                      "Use ### for any sub-headings. Output clean Markdown for just this component."),
    }
    _UNIT_OVERVIEW = {
        "entity": ("Write the data-model OVERVIEW: an entity-relationship summary, the global "
                   "indexing strategy, and the migration approach."),
        "resource": ("Write the API OVERVIEW: base URL/versioning, the single consistent error "
                     "format, the auth scheme, the pagination convention, and the "
                     "realtime/websocket event model."),
        "component": ("Write the architecture OVERVIEW: the architecture style, how the "
                      "components fit together (the request/message flow), the data stores and "
                      "caches, and the cross-cutting concerns (the real-time backbone if "
                      "applicable)."),
    }

    def _summarize_decomposed(self, spec: PhaseSpec, messages: list, unit: str) -> str | None:
        """Design a heavy phase one unit at a time: list → detail each → assemble."""
        import json
        from core.model import balanced_json_arrays

        list_prompt = (
            f"Based on the requirements and this discussion, list the {unit}s to design for "
            f"the {spec.title}. Output ONLY a JSON array of short names in a sensible "
            f'dependency order, e.g. ["User", "Message"].'
        )
        raw = _stream(list(messages) + [HumanMessage(content=list_prompt)], precise=True)
        items: list[str] = []
        for span in balanced_json_arrays(raw):
            try:
                data = json.loads(span)
            except Exception:
                continue
            if isinstance(data, list):
                items = [str(x).strip() for x in data if isinstance(x, str) and x.strip()][:20]
                if items:
                    break
        if not items:
            return None  # fall back to the normal single-pass summarize

        console.print(f"[dim]🔬 Designing {len(items)} {unit}(s) one at a time: "
                      f"{', '.join(items)}[/dim]")
        sections = []
        overview = _stream(list(messages) + [HumanMessage(content=self._UNIT_OVERVIEW[unit])],
                           precise=True).strip()
        if overview:
            sections.append(f"## Overview\n\n{overview}")
        details = 0
        for item in items:
            console.print(Rule(f"[dim]{unit}: {item}[/dim]"))
            detail = _stream(
                list(messages) + [HumanMessage(content=self._UNIT_DETAIL[unit].format(item=item))],
                precise=True,
            ).strip()
            if detail:
                sections.append(f"## {item}\n\n{detail}")
                details += 1
        # If most units came back empty (a flaky weak model), don't ship an
        # overview-only spec — fall back to a single-pass summary instead.
        if details < max(1, len(items) // 2):
            console.print("[yellow]Too many sub-units came back empty — falling back to a "
                          "single-pass summary.[/yellow]")
            return None
        return "\n\n".join(sections)

    _ANGLES = (
        "Aim for the most rigorous, complete solution — cover every hard case and edge.",
        "Aim for a pragmatic solution that is realistic to actually build and ship.",
        "Aim for the most robust and secure solution — prioritise correctness and data protection.",
    )

    def _one_decision(self, spec: PhaseSpec, messages: list, angle: str = "") -> str:
        """Produce one decision: a draft, then (optionally) one critique-and-revise pass."""
        from core.config import get_config

        draft_prompt = (
            f"Summarize this {spec.title} discussion into a clean, structured Markdown "
            f"record of the DECISIONS made about: {spec.focus}. Use headings, bullets, and "
            f"tables where useful. This is the permanent spec for this phase — be precise "
            f"and complete, no fluff."
            + (f"\nEmphasis for this version: {angle}" if angle else "")
        )
        base = list(messages) + [HumanMessage(content=draft_prompt)]
        draft = _stream(base, precise=True).strip()

        if not draft or not get_config().devmode_lever("reflect"):
            return draft

        # Reflection: a small model improves a concrete draft far better than it
        # writes a perfect one first-shot. One critique-and-revise pass.
        console.print(Rule("[dim]🔁 Refining the decision (critique + revise)…[/dim]"))
        improve_prompt = (
            f"Above is a DRAFT {spec.title} decision for: {self.state.get('idea', '')}.\n"
            f"As a senior {spec.role}, critique it and output an IMPROVED, complete version. "
            f"Specifically:\n"
            f"- Address the genuinely HARD, defining parts of THIS product that the draft "
            f"missed or treated generically.\n"
            f"- Name concrete technologies, protocols, patterns, and current stable versions.\n"
            f"- Ensure it fully covers: {spec.focus}, and is consistent with the requirements "
            f"and earlier decisions.\n"
            + ("".join(f"- It MUST explicitly cover: {item}\n" for item in spec.must_cover))
            + "Output ONLY the improved decision in clean Markdown — no preamble."
        )
        improved = _stream(
            base + [AIMessage(content=draft), HumanMessage(content=improve_prompt)],
            precise=True,
        ).strip()
        # Strip a conversational lead-in the model sometimes prepends.
        import re
        improved = re.sub(r"\A(here.{0,40}version|the improved decision|improved version)\s*:?\s*\n+",
                          "", improved, flags=re.IGNORECASE)
        return improved or draft

    def _judge_best(self, spec: PhaseSpec, candidates: list[str]) -> str:
        """Pick the strongest of several candidate decisions for a phase."""
        listing = "\n\n".join(f"### Candidate {i + 1}\n{c[:3500]}" for i, c in enumerate(candidates))
        must = "".join(f"- {m}\n" for m in spec.must_cover)
        system = (
            f"You are a senior {spec.role}. Several candidate {spec.title} decisions are below. "
            f"Pick the SINGLE best one — the most complete, correct, specific, and consistent, "
            f"that best decides {spec.focus}."
            + (f"\nIt should best cover these must-haves:\n{must}" if must else "")
            + "\nReply with ONLY the number of the best candidate."
        )
        raw = _critic_stream([SystemMessage(content=system),
                              HumanMessage(content=f"{listing}\n\nBest candidate number:")])
        idx = self._parse_choice(raw or "", len(candidates))
        console.print(f"[dim]  ✓ selected candidate {idx + 1} of {len(candidates)}[/dim]")
        return candidates[idx]

    @staticmethod
    def _parse_choice(raw: str, n: int) -> int:
        """Parse the judge's chosen 1-based candidate number into a 0-based index.

        Tries explicit forms first ("candidate 2", "#2", "number 2"), then any
        bare integer — but only accepts a value in [1, n], so a stray digit from
        the judge's prose (e.g. "covers 5 requirements") can't select a
        nonexistent candidate. Falls back to 0.
        """
        import re
        for pat in (r"candidate\s*#?\s*(\d+)", r"#\s*(\d+)", r"\bnumber\s*(\d+)", r"\b(\d+)\b"):
            for m in re.finditer(pat, raw, re.IGNORECASE):
                v = int(m.group(1))
                if 1 <= v <= n:
                    return v - 1
        return 0

    def _summarize(self, spec: PhaseSpec, messages: list) -> str:
        from core.config import get_config

        if spec.decompose:
            decomposed = self._summarize_decomposed(spec, messages, spec.decompose)
            if decomposed:
                return decomposed   # decomposition is the quality mechanism here

        cfg = get_config()
        if spec.best_of > 1 and cfg.devmode_best_of_gated():
            console.print("[dim]ℹ best-of-N skipped: it needs a stronger devmode.judge_model "
                          "to pay off (see evals/). Using a single reflected pass.[/dim]")
        n = spec.best_of if cfg.devmode_lever("best_of") else 1
        if n > 1:
            console.print(f"[dim]🎲 Generating {n} candidate {spec.title} decisions, then "
                          f"judging the best…[/dim]")
            candidates = []
            for i in range(n):
                console.print(Rule(f"[dim]candidate {i + 1}/{n}[/dim]"))
                c = self._one_decision(spec, messages, angle=self._ANGLES[i % len(self._ANGLES)])
                if c.strip():
                    candidates.append(c)
            if not candidates:
                return ""
            return candidates[0] if len(candidates) == 1 else self._judge_best(spec, candidates)

        return self._one_decision(spec, messages)

    # ── Cross-phase consistency ─────────────────────────────────────────────────

    def _decision_digest(self, title: str, text: str) -> str:
        """Compress a decision into a compact bullet list of commitments/constraints.

        Chunked, so deep details in a large *decomposed* decision (40k+ chars of
        per-entity schema) aren't truncated away before the comparison.
        """
        system = (
            "Extract the concrete technical COMMITMENTS and CONSTRAINTS from this design text "
            "as a tight bullet list — the things other phases must stay consistent with. "
            "Include: the auth/authz model; the encryption scheme and exactly WHERE keys and "
            "plaintext may and may NOT live; data stores and any schema detail that encodes a "
            "rule (e.g. a column that stores a key, token, or plaintext); protocols, versions "
            "and scale targets; and any explicit hard rule. One short line each — no prose. "
            "If the text states none, output exactly: NONE"
        )
        step = 8000
        spans = [text[i:i + step] for i in range(0, min(len(text), 48000), step)] or [text]
        bullets: list[str] = []
        for span in spans:
            try:
                out = _stream([SystemMessage(content=system),
                               HumanMessage(content=f"# {title} (excerpt)\n{span}\n\n"
                                                    f"List the commitments/constraints.")],
                              precise=True).strip()
            except Exception:  # noqa: BLE001
                continue
            if out and not out.upper().lstrip().startswith("NONE"):
                bullets.append(out)
        return "\n".join(bullets)[:1600]

    def _phase_digest(self, pid: str) -> str:
        """Cached digest of a completed phase's artifact (computed on first use)."""
        cache = self.state.setdefault("digests", {})
        if pid in cache:
            return cache[pid]
        spec = PHASES_BY_ID.get(pid)
        if not spec:
            return ""
        path = self.dir / spec.filename
        if not path.exists():
            return ""
        cache[pid] = self._decision_digest(spec.title, path.read_text(encoding="utf-8", errors="replace"))
        self._save()
        return cache[pid]

    def _consistency_findings(self, spec: PhaseSpec, prior: str, digest: str) -> str:
        """Find contradictions between a new decision's digest and earlier ones. '' if clean."""
        system = (
            "You are a senior reviewer checking ONE new design decision against the decisions "
            "already made in EARLIER phases of the same project (given as constraint "
            "summaries). Report ONLY direct CONTRADICTIONS — where the new decision conflicts "
            "with, violates, or is incompatible with an earlier decision. Do NOT report gaps, "
            "improvements, or style; only genuine inconsistencies between phases.\n"
            "Reason carefully about these high-value contradiction classes:\n"
            "- SECURITY INVARIANTS: if a phase says the server must never see plaintext or hold "
            "private keys (e.g. end-to-end encryption), then storing a private key, password, "
            "or plaintext server-side is a HIGH contradiction — even if it is 'encrypted at "
            "rest' or stored as bytes. Encrypting a private key at rest does NOT satisfy E2E.\n"
            "- TECH MISMATCH: the same data assigned to a different datastore/technology than an "
            "earlier phase chose.\n"
            "- AUTH MISMATCH: a different authentication/authorization mechanism than decided.\n"
            "- DROPPED SCOPE: a required feature with no corresponding schema/endpoint/flow.\n"
            "For each contradiction output one line: 'SEVERITY — what conflicts (cite both "
            "phases): concrete fix' (SEVERITY = HIGH/MEDIUM/LOW).\n"
            "If there are no contradictions, output exactly: NONE"
        )
        prompt = (
            f"# Constraints from earlier phases\n{prior[:9000]}\n\n"
            f"# New decision just made — {spec.title} (its commitments/constraints)\n{digest}\n\n"
            f"List only the contradictions, or exactly NONE."
        )
        try:
            out = _critic_stream([SystemMessage(content=system), HumanMessage(content=prompt)]).strip()
        except Exception:  # noqa: BLE001
            return ""
        if not out or out.upper().lstrip().startswith("NONE"):
            return ""
        return out

    def _report_consistency(self, spec: PhaseSpec, decision: str) -> str:
        """Digest a freshly-made decision, check it against earlier phases, surface conflicts."""
        from core.config import get_config
        if not get_config().devmode_lever("consistency_check") or not decision.strip():
            return ""
        # Digest this phase and cache it so later phases can compare against it cheaply.
        digest = self._decision_digest(spec.title, decision)
        self.state.setdefault("digests", {})[spec.id] = digest
        self._save()
        if not digest:
            return ""
        prior_parts = []
        for p in PHASES:
            if p.id == spec.id or p.target != "doc" or p.kind == "review":
                continue
            d = self._phase_digest(p.id)  # only returns for phases whose artifact exists
            if d:
                prior_parts.append(f"## {p.title}\n{d}")
        if not prior_parts:
            return ""
        console.print("[dim]🔎 Checking consistency with earlier decisions…[/dim]")
        findings = self._consistency_findings(spec, "\n\n".join(prior_parts), digest)
        if not findings:
            console.print("[dim]  ✓ consistent with earlier phases[/dim]")
            return ""
        console.print(Panel(findings,
                            title=f"[bold yellow]⚠ Consistency — {spec.title} vs earlier phases[/bold yellow]",
                            border_style="yellow"))
        console.print("[dim]Resolve with 'dev revisit <phase>' on either side of the conflict.[/dim]")
        self._append_consistency_note(spec, findings)
        return findings

    def _append_consistency_note(self, spec: PhaseSpec, findings: str) -> None:
        path = self.workspace / "docs" / "dev" / "consistency_notes.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        header = "" if path.exists() else "# Consistency notes\n_Cross-phase contradictions flagged as each phase was decided._\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{header}\n## {spec.title} — {ts}\n\n{findings}\n")

    # ── Phase runner ───────────────────────────────────────────────────────────

    def _run_review(self, spec: PhaseSpec) -> str:
        """The Design Reviewer critiques the other phases' decisions (no discussion)."""
        artifacts = self._prior_artifacts()
        if not artifacts.strip():
            console.print("[yellow]Nothing to review yet — run the design phases first.[/yellow]")
            return "skip"
        console.print(Rule(f"[bold magenta]{spec.role} — {spec.title}[/bold magenta]"))
        system = (
            f"You are a {spec.role} reviewing the design for an experienced developer. "
            f"Critically review the decisions below. Check: {spec.focus}. Output a concise "
            f"findings list — each finding with a severity (HIGH/MEDIUM/LOW), the phase it "
            f"concerns, and a concrete recommendation. Be direct; if it's solid, say so."
        )
        findings = _critic_stream(
            [SystemMessage(content=system),
             HumanMessage(content=f"# Design decisions to review\n{artifacts[:8000]}\n\n"
                                  f"Produce the review findings.")],
        )
        path = self._artifact_path(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        path.write_text(f"# Design Review\n_Reviewed: {ts}_\n\n{findings.strip()}\n", encoding="utf-8")
        console.print(f"\n  [green]✔[/green] Review saved → "
                      f"[bold]{path.relative_to(self.workspace)}[/bold]")
        console.print("[dim]Address issues with 'dev revisit <phase>', then 'dev build'.[/dim]")
        return "done"

    def _run_phase(self, spec: PhaseSpec) -> str:
        if spec.kind == "review":
            return self._run_review(spec)
        while True:  # retry loop for 'revise'
            result, messages = self._discuss(spec)
            if result in ("pause", "skip"):
                return result
            if result == "revise":
                console.print("[dim]Restarting this phase…[/dim]")
                continue
            # result == "done" → summarize + write artifact
            console.print(Rule("[dim]📝 Capturing decision…[/dim]"))
            decision = self._summarize(spec, messages)
            self._report_consistency(spec, decision)
            transcript = "\n\n".join(
                (f"**You:** {m.content}" if isinstance(m, HumanMessage) else f"**{spec.role}:** {m.content}")
                for m in messages[2:]  # skip system + opening
            )
            path = self._write_artifact(spec, decision, transcript)
            console.print(f"  [green]✔[/green] Saved → [bold]{path.relative_to(self.workspace)}[/bold]")
            return "done"

    # ── Public ─────────────────────────────────────────────────────────────────

    def show_status(self) -> None:
        from core.config import get_config

        lines = []
        for p in PHASES:
            st = self.state["phases"].get(p.id, {}).get("status", "pending")
            lines.append(f"  {_STATUS_ICON.get(st, '○')}  {p.title}")
        console.print(Panel("\n".join(lines),
                            title=f"[bold magenta]🧭 Developer Mode — {self.state.get('idea','')[:50]}[/bold magenta]",
                            subtitle=f"[dim]docs/dev/  ·  profile: {get_config().devmode_profile()}[/dim]",
                            border_style="magenta"))

    def _build_exists(self) -> bool:
        bp = self.dir / "build_plan.json"
        if not bp.exists():
            return False
        try:
            plan = json.loads(bp.read_text(encoding="utf-8"))
            return any(f.get("status") == "done" for f in plan.get("files", []))
        except Exception:
            return False

    # ── Resolve: turn review findings into design fixes + code resync ───────────

    _SEV_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

    def _design_context(self) -> list[tuple[str, str]]:
        """Compact (phase_id, digest) pairs for every decided phase."""
        out = []
        for p in PHASES:
            if p.target != "doc" or p.kind == "review":
                continue
            if (self.dir / p.filename).exists():
                d = self._phase_digest(p.id)
                if d:
                    out.append((p.id, d))
        return out

    def _review_findings_structured(self) -> list[dict]:
        """Holistic cross-phase review → structured, fixable findings."""
        ctx = self._design_context()
        if len(ctx) < 2:
            return []
        ids = [pid for pid, _ in ctx]
        blocks = "\n\n".join(f"### {pid} — {PHASES_BY_ID[pid].title}\n{d}" for pid, d in ctx)
        system = (
            "You are a senior reviewer doing a final cross-phase check of a software design. "
            "Find real CONTRADICTIONS and critical GAPS between phases — e.g. a schema that "
            "stores a private key/plaintext server-side despite an end-to-end-encryption "
            "promise, a datastore or auth mechanism that disagrees with an earlier phase, or a "
            "required feature with no schema/endpoint/flow. For each issue, pick the SINGLE "
            "phase whose decision should be edited to fix it.\n"
            f"Valid phase ids: {ids}\n"
            'Output ONLY a JSON array: [{"severity":"HIGH|MEDIUM|LOW","target":"<phase id>",'
            '"issue":"what is wrong","fix":"the concrete change to make"}]. '
            "If the design is consistent, output []."
        )
        from core.model import balanced_json_arrays
        try:
            raw = _critic_stream([SystemMessage(content=system),
                                  HumanMessage(content=f"# Phase decisions\n{blocks}\n\nReturn the findings JSON.")])
        except Exception:  # noqa: BLE001
            return []
        findings: list[dict] = []
        for span in balanced_json_arrays(raw):
            try:
                data = json.loads(span)
            except Exception:
                continue
            if isinstance(data, list):
                for f in data:
                    if (isinstance(f, dict) and f.get("target") in ids
                            and f.get("issue") and f.get("fix")):
                        f["severity"] = str(f.get("severity", "MEDIUM")).upper()
                        findings.append(f)
                if findings:
                    break
        findings.sort(key=lambda f: self._SEV_RANK.get(f["severity"], 1))
        return findings

    def _apply_fix(self, finding: dict) -> bool:
        """Rewrite the target phase's decision to resolve a finding, then resync code."""
        import re
        spec = PHASES_BY_ID[finding["target"]]
        path = self._artifact_path(spec)
        if not path.exists():
            console.print(f"[yellow]No artifact for {spec.title} to fix.[/yellow]")
            return False
        old_full = path.read_text(encoding="utf-8", errors="replace")
        old_dec = _decision_section(old_full)
        system = (
            f"You are a senior {spec.role}. You are given the current {spec.title} decision, "
            f"then a problem and the required change. Rewrite the decision so the problem is "
            f"resolved, changing as LITTLE else as possible and keeping all unaffected content "
            f"intact. Respond with ONLY the corrected decision in clean Markdown — do NOT "
            f"restate the problem, the required change, or any of these instructions."
        )
        prompt = (f"CURRENT DECISION:\n{old_dec}\n\n"
                  f"PROBLEM: {finding['issue']}\nREQUIRED CHANGE: {finding['fix']}\n\n"
                  f"Now output the corrected decision only.")
        console.print(Rule(f"[dim]Revising {spec.title} to fix: {finding['issue'][:70]}[/dim]"))
        new_dec = _stream([SystemMessage(content=system), HumanMessage(content=prompt)],
                          precise=True).strip()
        # Strip a leading "Corrected decision:" lead-in if the model adds one.
        new_dec = re.sub(r"^(corrected decision|here.{0,20}decision)\s*:?\s*\n+", "",
                         new_dec, flags=re.IGNORECASE)
        if not new_dec or new_dec == old_dec.strip():
            console.print("[dim]No change produced — skipping.[/dim]")
            return False
        # Echo guard: a small model sometimes parrots the prompt back instead of revising.
        if any(marker in new_dec for marker in ("REQUIRED CHANGE:", "PROBLEM:", "CURRENT DECISION:")):
            console.print("[yellow]The model echoed the request instead of revising — skipping. "
                          f"Use 'dev revisit {spec.id}' to change this phase.[/yellow]")
            return False
        # Guard: a fix that drops most of a large (decomposed) decision is a
        # truncation, not a fix — don't let it silently shrink the spec.
        if len(new_dec) < len(old_dec.strip()) * 0.6:
            console.print("[yellow]The rewrite is much shorter than the original — it likely "
                          "dropped content. Skipping; use 'dev revisit "
                          f"{spec.id}' to change this phase safely.[/yellow]")
            return False
        if not Confirm.ask(f"Apply this fix to the {spec.title} decision?", default=True):
            return False
        # Preserve the original discussion transcript if present.
        m = re.search(r"<details><summary>Discussion</summary>\n+(.*?)\n+</details>",
                      old_full, re.DOTALL)
        transcript = (m.group(1).strip() if m else "") + f"\n\n_Fix applied: {finding['issue']}_"
        self._write_artifact(spec, new_dec, transcript)
        self.state.setdefault("digests", {}).pop(spec.id, None)  # invalidate stale digest
        self._save()
        console.print(f"  [green]✔[/green] {spec.title} decision updated.")
        if self._build_exists():
            from devmode.resync import resync
            resync(self.workspace, spec.title, old_dec, new_dec)
        return True

    def resolve(self) -> None:
        """Review the design for contradictions/gaps and fix them (design + code resync)."""
        console.print(Rule("[bold magenta]Resolve — cross-phase review & fix[/bold magenta]"))
        console.print("[dim]🔎 Reviewing all phase decisions for contradictions…[/dim]")
        findings = self._review_findings_structured()
        if not findings:
            console.print("[green]✓ No cross-phase contradictions found.[/green]")
            return
        console.print(f"[yellow]Found {len(findings)} issue(s).[/yellow]\n")
        fixed = 0
        for i, f in enumerate(findings, 1):
            console.print(Panel(
                f"[bold]{f['severity']}[/bold] → fix in [bold cyan]{f['target']}[/bold cyan]\n\n"
                f"[bold]Issue:[/bold] {f['issue']}\n[bold]Proposed fix:[/bold] {f['fix']}",
                title=f"[bold]Finding {i}/{len(findings)}[/bold]", border_style="yellow"))
            if not Confirm.ask(f"Apply the proposed fix to {f['target']}?", default=True):
                console.print("[dim]Skipped.[/dim]")
                continue
            if self._apply_fix(f):
                fixed += 1
        console.print()
        console.print(Panel(f"Resolved {fixed}/{len(findings)} issue(s).\n"
                            "[dim]Re-run 'dev resolve' to re-check, or 'dev build' to (re)generate code.[/dim]",
                            title="[bold green]Resolve complete[/bold green]", border_style="green"))

    def revisit(self, phase_id: str) -> None:
        """Re-run one phase to change its decision, then auto-resync the code if it changed."""
        spec = PHASES_BY_ID.get(phase_id)
        if not spec:
            console.print(f"[yellow]Unknown phase '{phase_id}'. Phases: "
                          f"{', '.join(PHASES_BY_ID)}[/yellow]")
            return

        old_path = self._artifact_path(spec)
        old = old_path.read_text(encoding="utf-8", errors="replace") if old_path.exists() else ""
        old_status = self.state["phases"][spec.id]["status"]

        console.print(Rule(f"[bold]Revisiting: {spec.title}[/bold]"))
        self._set_status(spec.id, "in_progress")
        result = self._run_phase(spec)
        if result == "done":
            self._set_status(spec.id, "done")
        elif result == "pause":
            self._set_status(spec.id, "pending")
        else:  # 'skip' — don't downgrade a phase that was already completed
            self._set_status(spec.id, old_status if old_status == "done" else "skipped")
        if result != "done":
            return

        # A changed decision invalidates the cached consistency digest.
        self.state.get("digests", {}).pop(spec.id, None)
        self._save()

        new = old_path.read_text(encoding="utf-8", errors="replace") if old_path.exists() else ""
        if not old or not self._build_exists():
            return

        old_dec, new_dec = _decision_section(old), _decision_section(new)
        if old_dec.strip() == new_dec.strip():
            console.print("[dim]Decision unchanged — no resync needed.[/dim]")
            return

        console.print()
        if Confirm.ask(f"The {spec.title} decision changed. Auto-resync the code to match?",
                       default=True):
            from devmode.resync import resync
            resync(self.workspace, spec.title, old_dec, new_dec)

    def run(self, resume: bool = False, only: str | None = None) -> None:
        if not self.state.get("idea") and not resume:
            console.print("[yellow]No idea set. Use: develop <your idea>[/yellow]")
            return
        self._save()
        self.show_status()

        specs = [PHASES_BY_ID[only]] if only and only in PHASES_BY_ID else PHASES
        for spec in specs:
            st = self.state["phases"][spec.id]["status"]
            if st == "done" and not only:
                continue
            console.print()
            if not self.auto and st != "in_progress" and not Confirm.ask(
                f"[bold]Run phase: {spec.title}?[/bold] [dim]({spec.role})[/dim]", default=True
            ):
                self._set_status(spec.id, "skipped")
                continue
            self._set_status(spec.id, "in_progress")
            result = self._run_phase(spec)
            if result == "pause":
                console.print("\n[yellow]Paused. Resume with [bold]dev resume[/bold].[/yellow]")
                self._set_status(spec.id, "pending")
                return
            self._set_status(spec.id, "skipped" if result == "skip" else "done")
            self.show_status()

        console.print()
        console.print(Panel(
            "🎉 Design phases complete.\n\n"
            "[dim]Artifacts are in docs/dev/ and conventions in AICODER.md — edit them "
            "anytime. Next: run [bold]dev build[/bold] to turn the design into code, or "
            "[bold]dev resolve[/bold] to cross-check the phases first.[/dim]",
            title="[bold green]Developer Mode — design done[/bold green]", border_style="green"))
