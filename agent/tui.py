"""
agent/tui.py — Full-screen terminal UI, the default interactive front-end
==========================================================================
A Textual application that gives AICoder a persistent bottom input box with a
scrolling chat area above it — the same layout Claude Code uses — instead of
the plain print-and-scroll REPL in `agent/loop.py` (which remains the fallback
for non-terminal output: piped/redirected/scripted usage, and the whole test
suite, none of which run against a real tty).

Design: rather than rewrite every one of the ~25 existing slash-command
handlers and the whole devmode/tool confirmation flow, this module makes the
*existing*, already-tested business logic (`_handle_command`, `AgentSession`,
every `console.print(...)` call site across the codebase, every
`Confirm.ask`/`Prompt.ask`) work unchanged inside the TUI:

- `RichLogConsole` is a drop-in replacement for `rich.console.Console` (its
  actual, verified call surface: bare content, plus occasional `end=`/
  `markup=` kwargs — nothing else is used anywhere in this codebase) that
  writes into the chat `RichLog` instead of stdout.
- `_patch_consoles` swaps the module-level `console` singleton in every module
  that has one (there's no shared console object today — each module built its
  own) to the adapter above, for the lifetime of the TUI session, restoring
  the originals on exit.
- `_patch_prompts` swaps `rich.prompt.Confirm.ask` / `Prompt.ask` themselves
  (the same monkeypatch technique the test suite already uses, just applied at
  runtime) to route through a Textual modal, so shell/file confirmations, the
  devmode discuss loop's done/skip/revise/pause, and every other
  confirmation/prompt across the app becomes TUI-native with zero changes to
  any of those call sites.

Blocking business logic (`AgentSession.send`, `DevSession.run`, ...) runs in a
Textual worker thread so the UI stays responsive; results are written back to
the chat log via the same adapter (thread-safe, since Textual widgets queue
updates from worker threads automatically when using their public API).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Callable

from rich.errors import MarkupError
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import Button, Footer, Header, Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option
from textual_autocomplete import AutoComplete, DropdownItem, TargetState

# Every module with its own `console = Console()` singleton that the agent
# actually exercises during a live session (evals/ scripts are standalone
# tools, not part of an interactive session, and are excluded).
_CONSOLE_MODULES = [
    "agent.loop",
    "agent.planner",
    "agent.tools",
    "agent.mcp_client",
    "devmode.session",
    "devmode.build",
    "devmode.resync",
    "tools.shell_tools",
    "tools.file_tools",
    "tools.web_tools",
]

# Matches BRAND in agent/loop.py (cyan brackets, "yellow1" accent) — the same
# identity, on Textual's design-system variables ($primary, $accent, ...) so
# every built-in widget (Header, Footer, Input focus ring, ...) and every
# panel/modal below picks it up consistently, not just the startup banner.
AICODER_THEME = Theme(
    name="aicoder",
    primary="cyan",
    secondary="cyan",
    accent="#ffd700",
    warning="#ffd700",
    success="green",
    error="red",
    dark=True,
)


def _safe_markup_text(content: str) -> Text:
    """Parse Rich markup in `content`, falling back to literal (unparsed)
    text if it contains something that *looks* like a markup tag but isn't
    valid (e.g. a code snippet with `[...]`/`<...>` that reads as a
    mismatched closing tag) — confirmed live: a model's own generated code
    triggered exactly this, and crashed the app because the direct
    `rich_log.write(f"...{e}...")` error-reporting path embedded the
    original bad content in the message and re-parsed *that* as markup too,
    cascading into a second, uncaught crash. A malformed tag must never
    crash the app, and *especially* never crash error-reporting itself —
    every string handed to RichLog.write() in this module goes through this
    first, rather than relying on RichLog's own markup=True auto-parsing."""
    try:
        return Text.from_markup(content)
    except MarkupError:
        return Text(content)


class RichLogConsole:
    """A `console.print(...)`-compatible adapter that writes into a Textual
    `RichLog` instead of stdout. Matches the actual call surface used
    throughout this codebase (verified by grepping every console.print call
    site): a single content argument (a markup string or a Rich renderable),
    plus the occasional `end=`/`markup=` kwarg used by streamed shell output.
    Any other keyword arg is accepted and ignored rather than raising, so an
    unanticipated call site degrades gracefully instead of crashing the app.
    """

    def __init__(self, rich_log: RichLog) -> None:
        self.rich_log = rich_log

    def print(self, content: object = "", *, end: str = "\n", markup: bool = True, **_ignored) -> None:
        if content == "" and end == "\n":
            self.rich_log.write("")  # blank-line spacing, e.g. bare console.print()
            return
        if isinstance(content, str):
            text = content.rstrip("\n") if end == "" else content
            self.rich_log.write(_safe_markup_text(text) if markup else Text(text))
        else:
            self.rich_log.write(content)

    # `Console.screen()` is used by the plain-REPL fallback only; the TUI is
    # itself the full-screen mode, so this is a no-op context manager here.
    def screen(self, *_args, **_kwargs):
        from contextlib import nullcontext

        return nullcontext()


def _patch_consoles(new_console: RichLogConsole) -> Callable[[], None]:
    """Swap `console` in every module listed above to `new_console`. Returns a
    function that restores every module's original console object."""
    import importlib

    originals: list[tuple[object, object]] = []
    for name in _CONSOLE_MODULES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(mod, "console"):
            originals.append((mod, mod.console))
            mod.console = new_console

    def restore() -> None:
        for mod, original in originals:
            mod.console = original

    return restore


# ── Confirm/Prompt modals — the TUI-native equivalent of rich.prompt ────────────
# Every existing Confirm.ask("...", default=...)/Prompt.ask("...") call site
# across the codebase (shell/file confirmations, the devmode discuss loop,
# the /model picker's numbered choice, ...) always uses one of these two
# shapes — verified by grepping every call site — so these two modals cover
# the whole app without touching any of those call sites.

class ConfirmModal(ModalScreen[bool]):
    """A yes/no modal — dismiss(True/False). Enter confirms the default,
    y/n choose explicitly, Escape cancels (= No)."""

    BINDINGS = [
        Binding("y", "yes", "Yes", show=False),
        Binding("n", "no", "No", show=False),
        Binding("enter", "confirm_default", "Confirm", show=False),
        Binding("escape", "no", "Cancel", show=False),
    ]

    DEFAULT_CSS = """
    ConfirmModal {
        align: center middle;
    }
    ConfirmModal > Vertical {
        width: auto;
        height: auto;
        max-width: 70%;
        border: heavy $primary;
        background: $surface;
        padding: 1 2;
    }
    ConfirmModal Horizontal {
        height: auto;
        align: center middle;
        margin-top: 1;
    }
    ConfirmModal Button {
        margin: 0 1;
    }
    """

    def __init__(self, question: str, default: bool = True) -> None:
        super().__init__()
        self.question = question
        self.default = default

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self.question, markup=True)
            with Horizontal():
                yield Button("Yes (y)", id="yes", variant="success")
                yield Button("No (n)", id="no", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)

    def action_confirm_default(self) -> None:
        self.dismiss(self.default)


class ChoiceModal(ModalScreen[int | None]):
    """An arrow-key navigable list — dismiss(index), or dismiss(None) if
    cancelled (Escape). Used for the `/model` picker's arrow-key upgrade over
    the plain numbered-Prompt.ask fallback (see ask_choice/is_tui_active).

    Two optional, Claude-Code-picker-style touches:
    - `groups`: a section name per option (parallel to `options`) — a
      disabled (so arrow keys skip it), bold header row is inserted whenever
      the group changes, matching the plain-REPL /model panel's grouping.
    - `initial_index`: which option is already highlighted when the list
      opens (e.g. the current model), instead of always starting at the top.
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    ChoiceModal {
        align: center middle;
    }
    ChoiceModal > Vertical {
        width: 90%;
        height: auto;
        max-height: 90%;
        border: heavy $primary;
        background: $surface;
        padding: 1 2;
    }
    ChoiceModal OptionList {
        height: auto;
        max-height: 20;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        question: str,
        options: list[str],
        *,
        groups: list[str] | None = None,
        initial_index: int = 0,
    ) -> None:
        super().__init__()
        self.question = question
        self.options = options
        self.groups = groups
        self.initial_index = initial_index
        # Maps each row's position in the rendered OptionList (headers included)
        # back to its index in `options` — None for a header row itself.
        self._row_to_option_index: list[int | None] = []

    def _build_rows(self) -> list[Option]:
        rows: list[Option] = []
        last_group = None
        for i, label in enumerate(self.options):
            group = self.groups[i] if self.groups else None
            if group is not None and group != last_group:
                rows.append(Option(f"[bold]{group}[/bold]", disabled=True))
                self._row_to_option_index.append(None)
                last_group = group
            rows.append(Option(label))
            self._row_to_option_index.append(i)
        return rows

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self.question, markup=True)
            yield OptionList(*self._build_rows())

    def on_mount(self) -> None:
        option_list = self.query_one(OptionList)
        option_list.focus()
        try:
            initial_row = self._row_to_option_index.index(self.initial_index)
        except ValueError:
            initial_row = 0
        option_list.highlighted = initial_row

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self._row_to_option_index[event.option_index])

    def action_cancel(self) -> None:
        self.dismiss(None)


class PromptModal(ModalScreen[str]):
    """A free-text modal — dismiss(the typed string). Escape cancels, returning
    the default (matching Rich's Prompt.ask, which returns "" with no default
    on an empty submission)."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    PromptModal {
        align: center middle;
    }
    PromptModal > Vertical {
        width: auto;
        height: auto;
        min-width: 40;
        max-width: 70%;
        border: heavy $primary;
        background: $surface;
        padding: 1 2;
    }
    PromptModal Input {
        margin-top: 1;
    }
    """

    def __init__(self, question: str, default: str = "") -> None:
        super().__init__()
        self.question = question
        self.default = default

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self.question, markup=True)
            yield Input(value=self.default)

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(self.default)


# The single running TUI app instance, if any — lets the patched Confirm.ask/
# Prompt.ask (called from a worker thread) find their way back to the main
# thread's event loop via App.call_from_thread. None when the plain REPL
# fallback is in use, in which case the patch below is simply never installed.
_active_app: "AICoderApp | None" = None


def _tui_confirm(prompt: str = "", *, default: bool = True, **_ignored) -> bool:
    app = _active_app
    if app is None:  # defensive — shouldn't happen, the patch implies an app
        return bool(default)
    return app.call_from_thread(app.push_screen_wait, ConfirmModal(prompt, default))


def _tui_prompt(prompt: str = "", *, default: str = "", **_ignored) -> str:
    app = _active_app
    if app is None:
        return default
    return app.call_from_thread(app.push_screen_wait, PromptModal(prompt, default))


def _patch_prompts() -> Callable[[], None]:
    """Route rich.prompt.Confirm.ask/Prompt.ask through the TUI modals above —
    the same monkeypatch technique this project's own test suite already uses
    (`monkeypatch.setattr(Confirm, "ask", ...)`), just applied at runtime for
    the lifetime of the TUI session.

    Usually neither `ask` is defined directly on Confirm/Prompt (both inherit
    it from PromptBase), so restoring means deleting the shadowing attribute
    added here — but if something else (e.g. a test's own
    monkeypatch.setattr(Confirm, "ask", ...)) already put a class-level `ask`
    there before this ran, restore must put *that* back, not delete it out
    from under it — otherwise a test that patches Confirm/Prompt and then
    exercises the TUI in the same test breaks pytest's own monkeypatch
    teardown (it tries to delattr something already gone)."""
    from rich.prompt import Confirm, Prompt

    had_confirm_ask = "ask" in Confirm.__dict__
    original_confirm_ask = Confirm.__dict__.get("ask")
    had_prompt_ask = "ask" in Prompt.__dict__
    original_prompt_ask = Prompt.__dict__.get("ask")

    Confirm.ask = staticmethod(_tui_confirm)
    Prompt.ask = staticmethod(_tui_prompt)

    def restore() -> None:
        if had_confirm_ask:
            Confirm.ask = original_confirm_ask
        else:
            del Confirm.ask
        if had_prompt_ask:
            Prompt.ask = original_prompt_ask
        else:
            del Prompt.ask

    return restore


def is_tui_active() -> bool:
    """Whether a TUI app is currently running — call sites that want a richer
    arrow-key experience than the generic Confirm/Prompt bridge (e.g. the
    `/model` picker) check this before calling `ask_choice`."""
    return _active_app is not None


def ask_choice(
    prompt: str,
    options: list[str],
    *,
    groups: list[str] | None = None,
    initial_index: int = 0,
) -> int | None:
    """Show an arrow-key-navigable list; returns the chosen index, or None if
    cancelled (Escape). Caller must check `is_tui_active()` first — this is
    only meaningful, and only safe to call from a worker thread, while a TUI
    app is running. `groups`/`initial_index`: see ChoiceModal."""
    app = _active_app
    if app is None:
        raise RuntimeError("ask_choice() called with no active TUI app")
    modal = ChoiceModal(prompt, options, groups=groups, initial_index=initial_index)
    return app.call_from_thread(app.push_screen_wait, modal)


def ask_inline(prompt: str, default: str = "") -> str:
    """Ask a free-text question via the main chat input at the bottom,
    instead of a popup — used by /develop's discuss loop (see
    devmode/session.py's _ask), which asks a question at every phase; a
    popup per question (confirmed live) reads as a confusing wall of
    modals showing nothing but a bare phase id. Answering in the normal
    chat input instead makes it read like an ordinary back-and-forth
    conversation. Caller must check `is_tui_active()` first, and only call
    this from a worker thread (like ask_choice)."""
    app = _active_app
    if app is None:
        raise RuntimeError("ask_inline() called with no active TUI app")
    return app.call_from_thread(app.wait_for_chat_reply, prompt, default)


def ask_inline_confirm(prompt: str, default: bool = True) -> bool:
    """Yes/no via the main chat input, instead of a popup — see ask_inline
    (used by devmode/session.py's _confirm). Accepts y/yes/n/no
    (case-insensitive); a blank reply or anything else falls back to
    `default`, matching Confirm.ask's own behavior."""
    app = _active_app
    if app is None:
        raise RuntimeError("ask_inline_confirm() called with no active TUI app")
    hint = "yes" if default else "no"
    reply = app.call_from_thread(
        app.wait_for_chat_reply, f"{prompt} (y/n, default {hint})", "",
    )
    reply = reply.strip().lower()
    if reply in ("y", "yes"):
        return True
    if reply in ("n", "no"):
        return False
    return default


def signal_turn_started() -> None:
    """Called by AgentSession._invoke() when it starts streaming a response —
    shows the status bar's elapsed-time indicator. A plain attribute write:
    thread-safe enough for a single float, and read only by the main thread's
    own timer, so no call_from_thread round-trip is needed for this."""
    import time

    app = _active_app
    if app is not None:
        app.turn_start_time = time.monotonic()


def signal_turn_ended() -> None:
    app = _active_app
    if app is not None:
        app.turn_start_time = None


class _SlashCommandItem(DropdownItem):
    """A dropdown row showing "command  description", but whose .value (what
    gets inserted into the input on selection) is just the command — the
    description is display-only."""

    def __init__(self, command: str, description: str) -> None:
        super().__init__(main=Content.assemble(command, (f"  {description}", "dim")))
        self._command = command

    @property
    def value(self) -> str:
        return self._command


def _slash_command_candidates(state: TargetState) -> list[DropdownItem]:
    """Candidates for the "/" autocomplete dropdown — only while the input is
    still an in-progress command (no space yet, i.e. no argument started) that
    isn't already a complete, exact command. Without that last check, typing
    a full valid command (e.g. "/status") leaves one exact-match row showing,
    and AutoComplete's own Enter handling "completes" it (a no-op re-append of
    the same text) instead of letting Enter reach Input.Submitted — so a fully
    typed command would need two Enters to actually run."""
    from agent.loop import SLASH_COMMANDS

    text = state.text
    if not text.startswith("/") or " " in text:
        return []
    query = text[1:].lower()
    if any(name[1:].lower() == query for name, _ in SLASH_COMMANDS):
        return []
    return [
        _SlashCommandItem(name, desc) for name, desc in SLASH_COMMANDS
        if name[1:].lower().startswith(query)
    ]


class SlashCommandAutoComplete(AutoComplete):
    """Inserts the chosen command plus a trailing space (ready to type an
    argument), rather than the default of replacing the whole input value.

    AutoComplete's own fuzzy-match highlighting rebuilds a plain
    DropdownItemHit once there's a non-empty search string (see
    AutoComplete.get_matches), which loses _SlashCommandItem's .value
    override — `value` here may be just "/model" (the unmatched-query path)
    or the full "/model  <description>" (the matched-query path). Since a
    command is always a single token, splitting on whitespace is robust to
    either case rather than trusting the override held all the way through.
    """

    def apply_completion(self, value: str, state: TargetState) -> None:
        command = value.split()[0] if value.split() else value
        target = self.target
        target.value = ""
        target.insert_text_at_cursor(command + " ")

    def _align_to_target(self) -> None:
        """Show the dropdown above the input's cursor line, not below it.

        AutoComplete's default (Region(x - 1, y + 1, ...)) opens downward —
        fine for an input near the top of a screen, but ours is docked at the
        bottom with the Footer directly beneath it, so "below" has no room and
        the dropdown would be clipped. Same constrain-to-screen logic as the
        base implementation, just anchored above (y - height) instead.
        """
        from textual.geometry import Offset, Region, Spacing

        x, y = self.target.cursor_screen_offset
        dropdown = self.option_list
        width, height = dropdown.outer_size

        region = Region(x - 1, y - height, width, height).constrain(
            "inside", "none", Spacing.all(0), self.screen.scrollable_content_region,
        )
        self.absolute_offset = Offset(region.x, region.y)


class ChatInput(Input):
    """The main chat input. Ctrl+V (Input's default paste binding) normally
    only pastes Textual's own app.clipboard — text copied *within* the app,
    tracked via OSC 52 if the terminal reports it. That's not the same as the
    real OS clipboard a screenshot gets copied to, which is what a user
    pasting an image actually expects (confirmed: os NSPasteboard access via
    Pillow's ImageGrab works and is independent of what the terminal forwards
    over stdin — an image has no text form to forward in the first place).
    So this checks the real clipboard for an image first, and only falls back
    to the default text-paste behavior when there isn't one.

    Platform notes (from reading Pillow's ImageGrab source directly, not just
    its docs): macOS shells out to `osascript`, no extra dependency. Windows
    uses a bundled C extension, also no extra dependency. Linux needs
    `wl-paste` (Wayland) or `xclip` (X11) on PATH — if neither is found,
    grabclipboard() raises NotImplementedError (unlike a real "no image on
    the clipboard", which just returns None on every platform), which is
    handled below with a one-time notification pointing at the missing tool,
    not a silent fall-through to text paste.

    Up/Down recall previous prompts, like Claude Code's own input box (and
    a normal shell) — Textual's base Input has no binding for either by
    default, and the slash-command autocomplete dropdown (see
    SlashCommandAutoComplete below) only intercepts them itself while it's
    actually showing suggestions, so there's no conflict."""

    BINDINGS = [
        Binding("up", "history_prev", "Previous prompt", show=False),
        Binding("down", "history_next", "Next prompt", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._history_index: int | None = None
        # What was being typed (if anything) before the first Up press —
        # restored once Down navigates back past the newest history entry,
        # so browsing history never loses an in-progress draft.
        self._history_draft: str = ""

    def record_history(self, text: str) -> None:
        """Called once per real submission (see AICoderApp.on_input_submitted)
        — every kind (slash command, regular message, or a /develop-style
        inline reply), skipping only blank input. Won't record an immediate
        repeat of the last entry (matching normal shell-history behavior),
        and always resets navigation back to "not currently browsing" so the
        next Up starts from the newest entry, not wherever a previous
        browsing session left off."""
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
        self._history_index = None
        self._history_draft = ""

    def action_history_prev(self) -> None:
        if not self._history:
            return
        if self._history_index is None:
            self._history_draft = self.value
            self._history_index = len(self._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        else:
            return  # already at the oldest entry
        self.value = self._history[self._history_index]
        self.cursor_position = len(self.value)

    def action_history_next(self) -> None:
        if self._history_index is None:
            return  # not currently browsing history
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self.value = self._history[self._history_index]
        else:
            self._history_index = None
            self.value = self._history_draft
            self._history_draft = ""
        self.cursor_position = len(self.value)

    def action_paste(self) -> None:
        from PIL import Image, ImageGrab

        app = self.app
        try:
            clipboard_content = ImageGrab.grabclipboard()
        except NotImplementedError:
            # Linux only (per Pillow's own source): raised when neither
            # wl-paste (Wayland) nor xclip (X11) is on PATH — unlike every
            # other "no image on the clipboard" case on any platform, which
            # just returns None. Worth a one-time, actionable notification
            # rather than silently falling through to text paste, which would
            # otherwise look like Ctrl+V just does nothing for an image.
            if isinstance(app, AICoderApp):
                app.notify(
                    "Image paste needs 'xclip' (X11) or 'wl-clipboard' (Wayland) installed — "
                    "falling back to text paste.",
                    severity="warning", timeout=6,
                )
            clipboard_content = None
        except Exception:  # noqa: BLE001 — clipboard access varies by platform/session
            clipboard_content = None

        if isinstance(clipboard_content, Image.Image) and isinstance(app, AICoderApp):
            path = app.attach_pasted_image(clipboard_content)
            self.insert_text_at_cursor(f"[image: {path.name}] ")
            return
        super().action_paste()


class AICoderApp(App):
    """The full-screen chat UI: a scrolling log with a pinned input box."""

    CSS = """
    Screen {
        background: $surface;
    }
    RichLog {
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    #status {
        height: 1;
        padding: 0 1;
        color: $warning;
    }
    Input {
        dock: bottom;
        border: heavy $primary;
        background: $surface;
        margin: 0 1 1 1;
    }
    Input:focus {
        border: heavy $accent;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit_app", "Quit", priority=True, show=False),
        # Not priority: a modal's own Escape (dismiss/cancel) must win when one
        # is on screen — this only fires when Escape reaches the app unhandled
        # (i.e. focus is on the main Input, not a modal).
        Binding("escape", "interrupt_turn", "Interrupt", show=False),
    ]

    def __init__(self, workspace: Path, continue_session: bool = False) -> None:
        super().__init__()
        self.workspace = workspace
        self.continue_session = continue_session
        self.session = None
        self.turn_start_time: float | None = None
        self.pending_images: list[Path] = []
        self._restore_consoles: Callable[[], None] | None = None
        self._restore_prompts: Callable[[], None] | None = None
        self._status_timer = None
        # Set while ask_inline/ask_inline_confirm is waiting for a reply
        # typed into the main chat input (not a popup) — see
        # wait_for_chat_reply and on_input_submitted.
        self._pending_reply_future: asyncio.Future | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield RichLog(id="chat", markup=True, wrap=True, highlight=False, auto_scroll=True)
        yield Static("", id="status")
        yield ChatInput(placeholder="Describe a task, or /help for commands… (Ctrl+V pastes "
                        "an image)", id="prompt")
        yield SlashCommandAutoComplete(target="#prompt", candidates=_slash_command_candidates)
        yield Footer()

    def attach_pasted_image(self, img) -> Path:
        """Save a clipboard-pasted image (a PIL.Image) to a scratch temp file
        and queue it for the next message — see ChatInput.action_paste."""
        import tempfile
        import uuid

        scratch = Path(tempfile.gettempdir()) / "aicoder-images"
        scratch.mkdir(exist_ok=True)
        path = scratch / f"paste-{uuid.uuid4().hex[:8]}.png"
        img.save(path, format="PNG")
        self.pending_images.append(path)
        self.query_one("#chat", RichLog).write(f"[dim]📎 Attached image: {path.name}[/dim]")
        return path

    def on_mount(self) -> None:
        global _active_app
        from agent.loop import AgentSession, _startup_banner
        from agent.planner import Planner
        from core.config import get_config

        self.register_theme(AICODER_THEME)
        self.theme = "aicoder"

        rich_log = self.query_one("#chat", RichLog)
        self._restore_consoles = _patch_consoles(RichLogConsole(rich_log))
        self._restore_prompts = _patch_prompts()
        _active_app = self

        self.title = "AICoder"
        self.sub_title = str(self.workspace)

        cfg = get_config()
        try:
            self.session = AgentSession(self.workspace)
        except RuntimeError as e:
            # A misconfigured/missing provider package — show it and stop;
            # there is no usable session to run.
            rich_log.write(_safe_markup_text(f"[red]⚠ {e}[/red]"))
            self.query_one("#prompt", Input).disabled = True
            return

        # rich_log.size is always (0, 0) here — confirmed live: RichLog
        # defers every write() until its own size is known (see its
        # docstring), so querying it this early in mount is pointless. Use
        # the App's own size instead (reliable synchronously) minus
        # RichLog's horizontal padding (`padding: 0 1` in its CSS above).
        # Passing `width=` explicitly to write() below (rather than baking
        # it only into the Panel and calling write() with no width) matters
        # just as much: without it, write() falls back to *measuring* the
        # Panel's own declared width, then *shrinking* it to whatever
        # RichLog's scrollable_content_region.width happens to be at the
        # moment the deferred write actually replays — which, confirmed
        # live, locks onto the *first* resize event RichLog ever receives
        # (see its on_resize), an early/intermediate one narrower than the
        # final layout. That shrink doesn't reflow a Table with fixed pixel
        # column widths gracefully — it mangles the block-letter logo mid
        # glyph. An explicit write(width=...) is used directly as the
        # render width, sidestepping that shrink path entirely.
        banner_width = self.size.width - 2
        banner_size = (banner_width, max(10, self.size.height - 6))
        rich_log.write(
            _startup_banner(cfg, cfg.model_name, self.workspace, self.session, size=banner_size),
            width=banner_width,
        )

        if self.continue_session:
            # Confirmed live: load_transcript() alone restores the previous
            # conversation into session.messages (so the *model* has full
            # context on the next turn) but never re-displays any of it —
            # from the user's own perspective --continue looked exactly
            # like a brand new session, no different from not passing the
            # flag at all, even though the context was genuinely there.
            # _render_session_detail (the same renderer /history <n> uses)
            # replays the actual prior turns into the chat log so you can
            # see what you're continuing, not just trust that it's there.
            prior_session_path = self.session._latest_session_file()
            if self.session.load_transcript():
                rich_log.write(_safe_markup_text(
                    f"[dim]↺ Resumed the previous conversation "
                    f"({len(self.session.messages) - 1} message(s)):[/dim]"
                ))
                if prior_session_path is not None:
                    from agent.loop import _render_session_detail
                    _render_session_detail(prior_session_path)
            else:
                rich_log.write("[dim]No previous conversation found for this workspace — "
                               "starting fresh.[/dim]")
        if self.session.instructions:
            rich_log.write("[dim]📄 Loaded project instructions (AICODER.md).[/dim]")
        planner = Planner(self.workspace, self.session)
        if planner.has_active_plan():
            rich_log.write("[dim]An in-progress plan exists for this project — type "
                           "'/resume' to continue it.[/dim]")
        from agent.loop import _has_devmode_session
        if _has_devmode_session(self.workspace):
            rich_log.write("[dim]A Developer Mode design exists for this project — type "
                           "'/dev status' to see progress, or '/dev' to resume it.[/dim]")

        self.query_one("#prompt", Input).focus()
        self._status_timer = self.set_interval(0.3, self._tick_status)

    def _idle_status_line(self) -> str:
        """Persistent bottom status — where we are, what model's driving,
        and how full the context window is (like Claude Code's own status
        line) — shown whenever a turn isn't in flight (see _tick_status)."""
        from core.config import get_config
        from agent.loop import _msg_chars

        cfg = get_config()
        used = sum(_msg_chars(m) for m in self.session.messages[1:])
        budget = self.session._history_budget
        pct = min(100, round(100 * used / budget)) if budget else 0
        return (
            f"[dim]📁 {self.workspace.name or self.workspace} · "
            f"🧠 {cfg.model_name} · ctx {pct}%[/dim]"
        )

    def _tick_status(self) -> None:
        import time

        try:
            status = self.query_one("#status", Static)
        except NoMatches:
            return  # a tick landed mid-teardown, after on_unmount stopped us
        if self.turn_start_time is None:
            status.update(self._idle_status_line() if self.session else "")
        else:
            elapsed = int(time.monotonic() - self.turn_start_time)
            status.update(f"[dim]✻ Thinking… ({elapsed}s · esc to interrupt)[/dim]")

    def action_quit_app(self) -> None:
        self.exit()

    def action_interrupt_turn(self) -> None:
        if self.turn_start_time is not None and self.session is not None:
            self.session.request_interrupt()

    def on_unmount(self) -> None:
        global _active_app
        if self._status_timer is not None:
            self._status_timer.stop()
        if self._restore_consoles:
            self._restore_consoles()
        if self._restore_prompts:
            self._restore_prompts()
        _active_app = None
        if self.session is not None:
            self.session.mcp.shutdown()
            from agent.loop import _try_lmstudio_unload
            from core.config import get_config

            _try_lmstudio_unload(get_config().model_name, get_config())

    async def wait_for_chat_reply(self, prompt: str, default: str) -> str:
        """Block (from ask_inline/ask_inline_confirm's calling worker
        thread's perspective, via call_from_thread) until the user's next
        message in the main chat input — used instead of a modal so a
        /develop-style back-and-forth reads like an ordinary conversation.
        Prints `prompt` into the chat log first so there's always visible
        context for what's being asked (a call site's own preceding
        console.print, if any, is a bonus, not something this relies on)."""
        rich_log = self.query_one("#chat", RichLog)
        rich_log.write(_safe_markup_text(f"[bold yellow]➤ {prompt}[/bold yellow]"))
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._pending_reply_future = future
        input_widget = self.query_one("#prompt", Input)
        original_placeholder = input_widget.placeholder
        input_widget.placeholder = "Type your reply and press Enter…"
        try:
            return await future
        finally:
            self._pending_reply_future = None
            input_widget.placeholder = original_placeholder

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user = event.value.strip()
        event.input.value = ""
        if isinstance(event.input, ChatInput):
            event.input.record_history(user)
        # A pending devmode reply (see wait_for_chat_reply) always wins —
        # even over a slash command, since a freeform reply to "type: done
        # · skip · revise · pause" should never be reinterpreted as one.
        pending = self._pending_reply_future
        if pending is not None and not pending.done():
            from rich.markup import escape

            rich_log = self.query_one("#chat", RichLog)
            rich_log.write(_safe_markup_text(
                f"[bold green]{self.workspace.name}>[/bold green] {escape(user)}"
            ))
            pending.set_result(user)
            return
        if self.session is None:
            return
        if not user and not self.pending_images:
            return
        rich_log = self.query_one("#chat", RichLog)

        # Escape the user's own text before embedding it in a markup string —
        # confirmed live: a message like "check the [broken markup] here"
        # gets a bracketed chunk silently swallowed (a syntactically valid,
        # if meaningless, markup tag doesn't raise — _safe_markup_text's
        # fallback only catches actual parse errors) unless escaped first.
        from rich.markup import escape

        echo_user = escape(user)

        # A slash command always takes priority and never consumes pending
        # images (e.g. an accidental stray "/" shouldn't drop an attachment
        # the user is still about to send with their next real message).
        if user.startswith("/"):
            rich_log.write(_safe_markup_text(f"[bold green]{self.workspace.name}>[/bold green] {echo_user}"))
            self.process_input(user, None)
            return

        images, self.pending_images = self.pending_images, []
        label = f"[bold green]{self.workspace.name}>[/bold green] {echo_user}"
        if images:
            label += "  " + " ".join(f"[dim]📎{p.name}[/dim]" for p in images)
        rich_log.write(_safe_markup_text(label))
        self.process_input(user, images or None)

    @work(thread=True, exclusive=True)
    def process_input(self, user: str, images: list[Path] | None = None) -> None:
        import time

        from agent.loop import _handle_command

        rich_log = self.query_one("#chat", RichLog)
        start = time.monotonic()
        should_exit = False
        try:
            if user.startswith("/"):
                should_exit = _handle_command(user, self.session, self.workspace)
                return
            if images:
                self.session.send_with_images(user, images)
            else:
                self.session.send(user)
        except Exception as e:  # noqa: BLE001 — keep the app alive on any failure
            # _safe_markup_text, not a raw f-string: `e` can be (and, live,
            # has been) an error message that itself contains something that
            # looks like a malformed markup tag — e.g. a MarkupError quoting
            # the bad tag verbatim. Passing that straight to a markup=True
            # RichLog.write would parse it *again* and crash a second time,
            # right inside the handler meant to report the first crash.
            message = _safe_markup_text(
                f"[red]⚠ Error: {e}[/red]\n"
                "[dim]If the model is unreachable, check that LM Studio's local server is "
                "running (Developer tab → Start Server).[/dim]"
            )
            self.call_from_thread(rich_log.write, message)
        finally:
            # finally, not after the try block — must still run (and time) a
            # slash command that returns early above, or one that crashed.
            elapsed = time.monotonic() - start
            self.call_from_thread(rich_log.write, _safe_markup_text(f"[dim]⏱ {elapsed:.1f}s[/dim]"))
            if should_exit:
                self.call_from_thread(self.exit)


def run(workspace: Path, continue_session: bool = False) -> None:
    """Entry point used by cli.py when output is a real terminal.
    continue_session (`aicoder --continue`): resume the most recently saved
    conversation for this workspace instead of starting fresh."""
    AICoderApp(workspace, continue_session=continue_session).run()
    sys.stdout.flush()
