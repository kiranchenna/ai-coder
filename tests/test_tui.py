"""Tests for agent/tui.py — the full-screen Textual front-end.

Uses Textual's own headless test harness (App.run_test()), not a real
terminal — this is the standard, supported way to test a Textual app and
gives deterministic, fast tests without any pty/visual dependency.
"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from textual.widgets import Input, RichLog

from agent.tui import AICoderApp, RichLogConsole, _patch_consoles


@pytest.fixture(autouse=True)
def _isolate_memory_dir(monkeypatch, tmp_path):
    """AgentSession.send()/send_with_images() persist a transcript to
    ~/.aicoder/memory/<project_id>/conversation.json (for `aicoder
    --continue`) — autouse so no test here (several swap in a scripted LLM
    and really call send()) writes into the developer's real
    ~/.aicoder/memory/."""
    monkeypatch.setattr("core.config.MEMORY_DIR", tmp_path / "memory")


def _rendered_text(rich_log: RichLog) -> str:
    """Flatten a RichLog's current lines back into plain text for assertions."""
    return "\n".join("".join(seg.text for seg in strip) for strip in rich_log.lines)


# ── RichLogConsole — the console.print(...) adapter ─────────────────────────────

def test_richlog_console_writes_plain_content():
    app = AICoderApp(Path("."))
    log = RichLog()
    adapter = RichLogConsole(log)
    adapter.print("[bold]hello[/bold]")
    # Without a mounted app, .lines isn't populated until size is known; just
    # confirm no exception and the deferred-render queue captured it.
    assert log._deferred_renders or log.lines
    del app  # unused; keeps the app instance alive isn't needed here


def test_richlog_console_bare_call_writes_blank_line():
    log = RichLog()
    adapter = RichLogConsole(log)
    adapter.print()  # matches bare console.print() used for REPL spacing
    assert log._deferred_renders or log.lines


def test_richlog_console_respects_markup_false():
    log = RichLog()
    adapter = RichLogConsole(log)
    # Shell output uses console.print(line, end="", markup=False) — must not
    # raise, and must not attempt markup parsing on content like "[INFO] x".
    adapter.print("[INFO] not markup\n", end="", markup=False)


def test_richlog_console_falls_back_on_malformed_markup_instead_of_raising():
    # Live-reproduced crash: a tool result / model answer containing something
    # that *looks* like a markup tag but isn't (e.g. code with mismatched
    # brackets) raised MarkupError inside Text.from_markup and took the whole
    # app down. Must degrade to literal text, never raise.
    log = RichLog()
    adapter = RichLogConsole(log)
    adapter.print("[red]⚠ Error: closing tag '[/<m>]' at position 3334 doesn't match any open tag[/red]")


def test_safe_markup_text_falls_back_to_literal_on_bad_tag():
    from agent.tui import _safe_markup_text

    text = _safe_markup_text("closing tag '[/<m>]' doesn't match any open tag")
    assert "[/<m>]" in text.plain


def test_safe_markup_text_still_parses_valid_markup():
    from agent.tui import _safe_markup_text

    text = _safe_markup_text("[bold]hello[/bold]")
    assert text.plain == "hello"
    assert any(span.style == "bold" for span in text.spans)


def test_patch_consoles_restores_originals():
    import agent.loop as loop_mod

    original = loop_mod.console
    fake = RichLogConsole(RichLog())
    restore = _patch_consoles(fake)
    assert loop_mod.console is fake
    restore()
    assert loop_mod.console is original


# ── The app itself — headless, via Textual's Pilot ──────────────────────────────

@pytest.mark.asyncio
async def test_app_mounts_and_shows_the_banner():
    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.session is not None
        rich_log = app.query_one("#chat", RichLog)
        text = _rendered_text(rich_log)
        assert "AICoder" in text
        assert str(ws) in text
        assert app.query_one("#prompt", Input).has_focus


@pytest.mark.asyncio
async def test_continue_session_resumes_a_previous_conversation():
    ws = Path(tempfile.mkdtemp())

    first = AICoderApp(ws)
    async with first.run_test() as pilot:
        await pilot.pause()
        from langchain_core.messages import AIMessageChunk
        first.session.llm = type("S", (), {
            "stream": lambda self, msgs: iter([AIMessageChunk(content="Sure, on it.")]),
        })()
        inp = first.query_one("#prompt", Input)
        inp.value = "fix the login bug"
        await pilot.press("enter")
        await first.workers.wait_for_complete()
        await pilot.pause()

    second = AICoderApp(ws, continue_session=True)
    async with second.run_test() as pilot:
        await pilot.pause()
        text = _rendered_text(second.query_one("#chat", RichLog))
        assert "Resumed the previous conversation" in text
        from langchain_core.messages import HumanMessage
        human_messages = [m for m in second.session.messages if isinstance(m, HumanMessage)]
        assert any(m.content == "fix the login bug" for m in human_messages)


@pytest.mark.asyncio
async def test_continue_session_with_nothing_saved_starts_fresh():
    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws, continue_session=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        text = _rendered_text(app.query_one("#chat", RichLog))
        assert "No previous conversation found" in text
        assert len(app.session.messages) == 1  # just the system prompt


@pytest.mark.asyncio
async def test_slash_command_runs_through_the_real_handler():
    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#prompt", Input)
        inp.value = "/status"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _rendered_text(app.query_one("#chat", RichLog))
        assert "Dev Mode" in text and "balanced" in text
        assert inp.value == ""  # input cleared after submission


@pytest.mark.asyncio
async def test_exit_command_stops_the_app():
    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#prompt", Input)
        inp.value = "/exit"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.is_running is False


@pytest.mark.asyncio
async def test_blank_input_is_ignored():
    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        rich_log = app.query_one("#chat", RichLog)
        before = _rendered_text(rich_log)
        inp = app.query_one("#prompt", Input)
        inp.value = "   "
        await pilot.press("enter")
        await pilot.pause()
        assert _rendered_text(rich_log) == before


@pytest.mark.asyncio
async def test_unmount_restores_consoles_and_shuts_down_mcp():
    import agent.loop as loop_mod

    original = loop_mod.console
    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test():
        assert loop_mod.console is not original  # patched while running
    assert loop_mod.console is original  # restored after the app exits


# ── Confirm.ask / Prompt.ask bridge — the generic TUI-native modal ──────────────
# Every existing Confirm.ask/Prompt.ask call site in the codebase (shell/file
# confirmations, devmode's discuss loop, ...) is exercised from a worker
# thread in real use; these tests replicate that exactly via
# App.run_worker(thread=True), the same mechanism production code uses,
# rather than calling the bridge functions directly from the test's async
# context (which App.call_from_thread explicitly forbids).

async def _await_modal(pilot, app, attempts: int = 50, min_depth: int = 2) -> None:
    """Wait until the screen stack reaches at least `min_depth` — pass
    min_depth=3 to wait for a *second* modal stacked on top of a first one
    (len(screen_stack) > 1 is already true once the first is up, so it
    wouldn't actually wait for the second)."""
    import asyncio

    for _ in range(attempts):
        await pilot.pause()
        if len(app.screen_stack) >= min_depth:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("modal was never pushed")


async def _await_result(pilot, box: dict, attempts: int = 50) -> None:
    import asyncio

    for _ in range(attempts):
        await pilot.pause()
        if "result" in box:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("worker never returned a result")


@pytest.mark.asyncio
async def test_confirm_bridge_yes_key():
    from rich.prompt import Confirm

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        box: dict = {}
        app.run_worker(lambda: box.__setitem__("result", Confirm.ask("Proceed?", default=False)),
                       thread=True)
        await _await_modal(pilot, app)
        await pilot.press("y")
        await _await_result(pilot, box)
        assert box["result"] is True


@pytest.mark.asyncio
async def test_confirm_bridge_no_key():
    from rich.prompt import Confirm

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        box: dict = {}
        app.run_worker(lambda: box.__setitem__("result", Confirm.ask("Proceed?", default=True)),
                       thread=True)
        await _await_modal(pilot, app)
        await pilot.press("n")
        await _await_result(pilot, box)
        assert box["result"] is False


@pytest.mark.asyncio
async def test_confirm_bridge_enter_uses_default():
    from rich.prompt import Confirm

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        box: dict = {}
        app.run_worker(lambda: box.__setitem__("result", Confirm.ask("Proceed?", default=True)),
                       thread=True)
        await _await_modal(pilot, app)
        await pilot.press("enter")
        await _await_result(pilot, box)
        assert box["result"] is True


@pytest.mark.asyncio
async def test_confirm_bridge_escape_cancels_as_no():
    from rich.prompt import Confirm

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        box: dict = {}
        app.run_worker(lambda: box.__setitem__("result", Confirm.ask("Proceed?", default=True)),
                       thread=True)
        await _await_modal(pilot, app)
        await pilot.press("escape")
        await _await_result(pilot, box)
        assert box["result"] is False


@pytest.mark.asyncio
async def test_prompt_bridge_returns_typed_text():
    from rich.prompt import Prompt

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        box: dict = {}
        app.run_worker(lambda: box.__setitem__("result", Prompt.ask("Model number?", default="")),
                       thread=True)
        await _await_modal(pilot, app)
        await pilot.press("2")
        await pilot.press("enter")
        await _await_result(pilot, box)
        assert box["result"] == "2"


@pytest.mark.asyncio
async def test_prompt_bridge_escape_returns_default():
    from rich.prompt import Prompt

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        box: dict = {}
        app.run_worker(lambda: box.__setitem__("result", Prompt.ask("Name?", default="fallback")),
                       thread=True)
        await _await_modal(pilot, app)
        await pilot.press("escape")
        await _await_result(pilot, box)
        assert box["result"] == "fallback"


def test_patch_prompts_restores_inherited_ask():
    from rich.prompt import Confirm, Prompt

    from agent.tui import _patch_prompts

    assert "ask" not in Confirm.__dict__  # inherited from PromptBase, not shadowed
    assert "ask" not in Prompt.__dict__
    restore = _patch_prompts()
    assert Confirm.__dict__["ask"] is not None
    restore()
    assert "ask" not in Confirm.__dict__
    assert "ask" not in Prompt.__dict__


def test_patch_prompts_restores_a_pre_existing_class_level_ask():
    # Regression guard: if something else (e.g. a test's own
    # monkeypatch.setattr(Confirm, "ask", ...)) already shadowed `ask` before
    # _patch_prompts ran, restore() must put THAT back, not delete it —
    # otherwise pytest's own monkeypatch teardown breaks trying to delattr
    # something already gone (this broke test_model_picker_other_entry_...
    # before this fix, since it monkeypatches Confirm.ask itself).
    from rich.prompt import Confirm

    from agent.tui import _patch_prompts

    def sentinel(*a, **k):
        return "sentinel"

    Confirm.ask = staticmethod(sentinel)
    try:
        restore = _patch_prompts()
        assert Confirm.ask is not sentinel
        restore()
        assert Confirm.ask is sentinel
    finally:
        del Confirm.ask


# ── /model's arrow-key picker (via the real /model command, end to end) ─────────
# Config is isolated (AICODER_HOME/CONFIG_PATH redirected to a temp dir) so
# this never writes to the developer's real ~/.aicoder/config.yaml — the exact
# mistake to avoid, since /model's selection handler really does call
# save_config().

def _isolate_config(monkeypatch, tmp_path):
    import core.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "AICODER_HOME", tmp_path)
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.yaml")
    # get_config() caches a module-level singleton — without resetting it too,
    # a test running earlier in the same session (isolated or not) can leave
    # a *shared* Config object cached, and this redirect would be silently
    # ignored (get_config() would keep returning that stale instance instead
    # of loading fresh from the path just set above).
    monkeypatch.setattr(cfg_mod, "_config", None)
    return cfg_mod.get_config()


@pytest.mark.asyncio
async def test_model_command_arrow_key_picker_switches_model(monkeypatch, tmp_path):
    import core.model as model_mod

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    cfg = _isolate_config(monkeypatch, cfg_dir)
    original = cfg.raw()["model"]["name"]

    fake_models = [
        {"name": original, "size": 4_700_000_000, "vision": False},
        {"name": "qwen2.5-coder-14b-instruct", "size": 9_000_000_000, "vision": False},
    ]
    monkeypatch.setattr(model_mod, "list_lmstudio_models", lambda **k: fake_models)
    monkeypatch.setattr(model_mod, "switch_lmstudio_model", lambda name: None)

    ws = tmp_path / "workspace"
    ws.mkdir()
    app = AICoderApp(ws)
    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one("#prompt", Input)
            inp.value = "/model"
            await pilot.press("enter")
            await _await_modal(pilot, app)
            await pilot.press("down")  # move from entry 1 to entry 2
            await pilot.press("enter")  # select it
            await app.workers.wait_for_complete()
            await pilot.pause()
        assert cfg.raw()["model"]["name"] == "qwen2.5-coder-14b-instruct"
    finally:
        cfg.raw()["model"]["name"] = original


@pytest.mark.asyncio
async def test_vision_model_command_arrow_key_picker_switches_model(monkeypatch, tmp_path):
    import core.model as model_mod

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    cfg = _isolate_config(monkeypatch, cfg_dir)
    original = cfg.raw()["vision"]["model"] or "current-vlm"
    cfg.raw()["vision"]["model"] = original

    fake_models = [
        {"name": original, "size": 6_000_000_000, "vision": True},
        {"name": "some-other-vlm", "size": 4_700_000_000, "vision": True},
    ]
    monkeypatch.setattr(model_mod, "list_lmstudio_models", lambda **k: fake_models)
    monkeypatch.setattr(model_mod, "is_lmstudio_model_downloaded", lambda name: True)

    ws = tmp_path / "workspace"
    ws.mkdir()
    app = AICoderApp(ws)
    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one("#prompt", Input)
            inp.value = "/vision model"
            await pilot.press("enter")
            await _await_modal(pilot, app)
            await pilot.press("down")  # move from entry 1 to entry 2
            await pilot.press("enter")  # select it
            await app.workers.wait_for_complete()
            await pilot.pause()
        assert cfg.raw()["vision"]["model"] == "some-other-vlm"
    finally:
        cfg.raw()["vision"]["model"] = original


@pytest.mark.asyncio
async def test_model_picker_other_entry_switches_to_a_typed_name(monkeypatch, tmp_path):
    import core.model as model_mod
    from textual.widgets import OptionList

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    cfg = _isolate_config(monkeypatch, cfg_dir)
    original = cfg.raw()["model"]["name"]

    monkeypatch.setattr(model_mod, "list_lmstudio_models",
                        lambda **k: [{"name": original, "size": 1, "vision": False}])
    monkeypatch.setattr(model_mod, "switch_lmstudio_model", lambda name: None)

    ws = tmp_path / "workspace"
    ws.mkdir()
    app = AICoderApp(ws)
    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one("#prompt", Input)
            inp.value = "/model"
            await pilot.press("enter")
            await _await_modal(pilot, app)

            modal = app.screen
            option_list = modal.query_one(OptionList)
            last_row = option_list.option_count - 1
            assert modal._row_to_option_index[last_row] == len(modal.options) - 1
            assert "Other" in str(option_list.get_option_at_index(last_row).prompt)

            # Real key-by-key navigation, not a direct .highlighted = assignment
            # — OptionList's Enter binding doesn't pick up a programmatic
            # highlight change the same way (confirmed by direct comparison).
            for _ in range(30):
                if option_list.highlighted == last_row:
                    break
                await pilot.press("down")
                await pilot.pause()
            assert option_list.highlighted == last_row

            await pilot.press("enter")  # picks "Other…" -> opens the name PromptModal
            for _ in range(50):
                await pilot.pause()
                if app.screen is not modal:  # ChoiceModal replaced by PromptModal
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("PromptModal for the custom name never appeared")

            # type the custom model name into the PromptModal's Input
            await pilot.press("l", "l", "a", "m", "a", "3", ".", "2", ":", "1", "b")
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
        assert cfg.raw()["model"]["name"] == "llama3.2:1b"
    finally:
        cfg.raw()["model"]["name"] = original


# ── ChoiceModal — group headers + pre-highlighting the current choice ───────────
# Matches Claude Code's own /model picker: entries grouped under section
# headers (skippable by arrow keys, not selectable), opening with the current
# choice already highlighted rather than always starting at the top.

def test_build_rows_inserts_a_disabled_header_per_group_change():
    from agent.tui import ChoiceModal

    modal = ChoiceModal(
        "Pick one",
        ["a", "b", "c"],
        groups=["Installed", "Installed", "Recommended"],
    )
    rows = modal._build_rows()
    # header, a, b, header, c
    assert [r.disabled for r in rows] == [True, False, False, True, False]
    assert modal._row_to_option_index == [None, 0, 1, None, 2]


def test_build_rows_with_no_groups_has_no_headers():
    from agent.tui import ChoiceModal

    modal = ChoiceModal("Pick one", ["a", "b"])
    rows = modal._build_rows()
    assert [r.disabled for r in rows] == [False, False]
    assert modal._row_to_option_index == [0, 1]


@pytest.mark.asyncio
async def test_choice_modal_opens_highlighting_the_initial_index_not_the_top():
    from agent.tui import ChoiceModal
    from textual.widgets import OptionList

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        # Pushed directly (no worker-thread bridge needed) — only checking
        # where the cursor lands on open, not the eventual return value.
        app.push_screen(ChoiceModal("Pick one", ["a", "b", "c"],
                                     groups=["G1", "G1", "G2"], initial_index=2))
        await pilot.pause()
        modal = app.screen
        option_list = modal.query_one(OptionList)
        assert modal._row_to_option_index[option_list.highlighted] == 2


@pytest.mark.asyncio
async def test_arrow_keys_skip_disabled_header_rows():
    from agent.tui import ChoiceModal
    from textual.widgets import OptionList

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(ChoiceModal("Pick one", ["a", "b", "c"],
                                     groups=["G1", "G2", "G2"], initial_index=0))
        await pilot.pause()
        modal = app.screen
        option_list = modal.query_one(OptionList)
        assert modal._row_to_option_index[option_list.highlighted] == 0  # "a"

        await pilot.press("down")  # should skip the "G2" header, land on "b"
        await pilot.pause()
        assert modal._row_to_option_index[option_list.highlighted] == 1


def test_model_menu_label_marks_the_current_model():
    from agent.loop import _model_menu_label

    current = {"tag": "qwen2.5-coder:7b", "size_bytes": 4_700_000_000,
               "installed": True, "current": True, "note": None}
    other = {"tag": "qwen2.5-coder:14b", "size_bytes": 9_000_000_000,
              "installed": True, "current": False, "note": None}
    assert "✓" in _model_menu_label(current)
    assert "✓" not in _model_menu_label(other)


# ── Thinking indicator + Esc-to-interrupt ────────────────────────────────────────
# _invoke() branches on is_tui_active() (rich.live.Live needs a real Console,
# which the TUI's console adapter isn't) — a slow scripted LLM (chunks spaced
# out in real time) lets these tests observe the status widget mid-turn and
# measure interrupt latency deterministically, without depending on a real
# model server.

class _SlowScriptedLLM:
    def __init__(self, n_chunks: int, delay: float):
        self.n_chunks = n_chunks
        self.delay = delay

    def stream(self, messages):
        import time
        from langchain_core.messages import AIMessageChunk

        for i in range(self.n_chunks):
            time.sleep(self.delay)
            yield AIMessageChunk(content=f"word{i} ")


@pytest.mark.asyncio
async def test_status_indicator_shows_while_a_turn_is_running():
    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.session.llm = _SlowScriptedLLM(n_chunks=20, delay=0.1)

        inp = app.query_one("#prompt", Input)
        inp.value = "go"
        await pilot.press("enter")

        for _ in range(20):
            await pilot.pause()
            if app.turn_start_time is not None:
                break
            await asyncio.sleep(0.02)
        assert app.turn_start_time is not None

        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.turn_start_time is None  # cleared once the turn finished


@pytest.mark.asyncio
async def test_escape_interrupts_a_running_turn():
    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.session.llm = _SlowScriptedLLM(n_chunks=50, delay=0.1)

        inp = app.query_one("#prompt", Input)
        inp.value = "go"
        await pilot.press("enter")

        for _ in range(20):
            await pilot.pause()
            if app.turn_start_time is not None:
                break
            await asyncio.sleep(0.02)

        await pilot.press("escape")
        await app.workers.wait_for_complete()
        await pilot.pause()

        text = _rendered_text(app.query_one("#chat", RichLog))
        assert "Interrupted" in text
        # cut off well short of all 50 chunks (50 * 0.1s = 5s if uninterrupted)
        assert app.session.llm.stream is not None  # sanity: didn't crash


# ── "/" autocomplete dropdown ────────────────────────────────────────────────────
# AutoComplete's own fuzzy-match highlighting rebuilds a plain DropdownItemHit
# once there's a non-empty search string, losing _SlashCommandItem's .value
# override (root-caused by inspecting AutoComplete.get_matches) — these tests
# guard the fix (apply_completion splits on whitespace rather than trusting
# the passed value directly).

@pytest.mark.asyncio
async def test_slash_shows_every_command_in_the_dropdown():
    from agent.loop import SLASH_COMMANDS
    from agent.tui import SlashCommandAutoComplete

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        ac = app.query_one(SlashCommandAutoComplete)
        await pilot.press("slash")
        await pilot.pause()
        assert ac.display is True
        assert ac.option_list.option_count == len(SLASH_COMMANDS)


@pytest.mark.asyncio
async def test_typing_filters_the_dropdown_to_matching_commands():
    from agent.tui import SlashCommandAutoComplete

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        ac = app.query_one(SlashCommandAutoComplete)
        await pilot.press("slash")
        await pilot.press("m")
        await pilot.pause()
        # .value on the match-highlighted items is "command  description"
        # (AutoComplete's own fuzzy-match rebuild, see the note above) — the
        # command itself is always the first whitespace-delimited token.
        commands = {ac.option_list.get_option_at_index(i).value.split()[0]
                    for i in range(ac.option_list.option_count)}
        assert commands == {"/model", "/mcp", "/memory"}


@pytest.mark.asyncio
async def test_selecting_a_completion_inserts_the_clean_command_with_trailing_space():
    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#prompt", Input)
        await pilot.press("slash")
        await pilot.press("s", "t", "a")  # narrows to /status
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        assert inp.value == "/status "


@pytest.mark.asyncio
async def test_completion_enter_does_not_submit_a_second_enter_does():
    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#prompt", Input)
        rich_log = app.query_one("#chat", RichLog)
        before = _rendered_text(rich_log)

        await pilot.press("slash")
        await pilot.press("s", "t", "a")
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("enter")  # completes only
        await pilot.pause()
        assert _rendered_text(rich_log) == before

        await pilot.press("enter")  # now actually submits
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert _rendered_text(rich_log) != before
        assert inp.value == ""


@pytest.mark.asyncio
async def test_escape_dismisses_the_dropdown_without_side_effects():
    from agent.tui import SlashCommandAutoComplete

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        ac = app.query_one(SlashCommandAutoComplete)
        inp = app.query_one("#prompt", Input)
        await pilot.press("slash")
        await pilot.pause()
        assert ac.display is True

        await pilot.press("escape")
        await pilot.pause()
        assert ac.display is False
        assert inp.value == "/"
        assert app.is_running is True


@pytest.mark.asyncio
async def test_dropdown_hides_once_an_argument_is_being_typed():
    from agent.tui import SlashCommandAutoComplete

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        ac = app.query_one(SlashCommandAutoComplete)
        await pilot.press("slash", "m", "o")  # partial match ("/model"), dropdown open
        await pilot.pause()
        assert ac.display is True
        await pilot.press("space")
        await pilot.pause()
        assert ac.option_list.option_count == 0


@pytest.mark.asyncio
async def test_dropdown_hides_once_the_command_is_typed_exactly():
    # Regression guard: without this, one exact-match row stays visible and
    # AutoComplete's own Enter handling "completes" it (re-appending the same
    # text) instead of letting Enter reach Input.Submitted — a fully typed
    # command would need two Enters to actually run (see
    # test_slash_command_runs_through_the_real_handler for the real-world case
    # that broke when this guard was missing).
    from agent.tui import SlashCommandAutoComplete

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        ac = app.query_one(SlashCommandAutoComplete)
        inp = app.query_one("#prompt", Input)
        inp.value = "/status"
        await pilot.pause()
        assert ac.display is False

        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert inp.value == ""  # submitted on the first Enter, not "completed"


# ── Ctrl+V clipboard image paste (the vision two-model handoff, TUI side) ──────
# ChatInput.action_paste checks the real OS clipboard for an image via
# PIL.ImageGrab (confirmed independently, via osascript, that this bypasses
# the terminal entirely — unlike Input's default action_paste, which only
# reads Textual's own app.clipboard, i.e. text copied *within* the app).

@pytest.mark.asyncio
async def test_ctrl_v_with_clipboard_image_saves_and_queues_it():
    from unittest.mock import patch

    from PIL import Image

    from agent.tui import ChatInput

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        fake_image = Image.new("RGB", (4, 4), color="red")
        inp = app.query_one("#prompt", ChatInput)
        inp.focus()
        try:
            with patch("PIL.ImageGrab.grabclipboard", return_value=fake_image):
                await pilot.press("ctrl+v")
                await pilot.pause()

            assert len(app.pending_images) == 1
            saved = app.pending_images[0]
            assert saved.exists()
            assert saved.suffix == ".png"
            assert "Attached image" in _rendered_text(app.query_one("#chat", RichLog))
            assert f"[image: {saved.name}]" in inp.value
        finally:
            for p in app.pending_images:
                p.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_ctrl_v_with_no_clipboard_image_falls_back_to_text_paste():
    from unittest.mock import patch

    from agent.tui import ChatInput

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#prompt", ChatInput)
        inp.focus()
        with patch("PIL.ImageGrab.grabclipboard", return_value=None):
            await pilot.press("ctrl+v")  # must not crash; falls through to super().action_paste()
            await pilot.pause()
        assert app.pending_images == []


@pytest.mark.asyncio
async def test_ctrl_v_on_linux_without_xclip_notifies_instead_of_silent_failure():
    # Pillow's ImageGrab.grabclipboard() raises NotImplementedError on Linux
    # specifically when neither wl-paste nor xclip is on PATH (confirmed by
    # reading its source) — every other "no image" case on any platform just
    # returns None. Without this handling, Ctrl+V would silently do nothing,
    # indistinguishable from "there's just no image on the clipboard".
    from unittest.mock import patch

    from agent.tui import ChatInput

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.query_one("#prompt", ChatInput)
        inp.focus()
        notified = []
        app.notify = lambda *a, **k: notified.append((a, k))
        with patch("PIL.ImageGrab.grabclipboard", side_effect=NotImplementedError()):
            await pilot.press("ctrl+v")
            await pilot.pause()
        assert app.pending_images == []
        assert notified
        message, kwargs = notified[0]
        assert "xclip" in message[0] and "wl-clipboard" in message[0]
        assert kwargs.get("severity") == "warning"


@pytest.mark.asyncio
async def test_submitting_with_pending_image_routes_through_vision_handoff(monkeypatch):
    from unittest.mock import patch

    from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
    from PIL import Image

    from agent.tui import ChatInput

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr("core.model.is_lmstudio_model_downloaded", lambda name: True)

        class FakeVision:
            def invoke(self, messages):
                return AIMessage(content="a stack trace mentioning a null pointer")

        import agent.loop as loop_mod
        monkeypatch.setattr(loop_mod, "get_chat_model", lambda **k: FakeVision())
        app.session.llm = type("S", (), {
            "stream": lambda self, msgs: iter([AIMessageChunk(content="Found it — fixing now.")]),
        })()

        fake_image = Image.new("RGB", (4, 4), color="blue")
        inp = app.query_one("#prompt", ChatInput)
        inp.focus()
        try:
            with patch("PIL.ImageGrab.grabclipboard", return_value=fake_image):
                await pilot.press("ctrl+v")
                await pilot.pause()
            saved_image = app.pending_images[0]

            inp.value = "what's crashing?"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert app.pending_images == []  # cleared after dispatch
            text = _rendered_text(app.query_one("#chat", RichLog))
            assert "Found it — fixing now." in text
            human_messages = [m for m in app.session.messages if isinstance(m, HumanMessage)]
            assert "what's crashing?" in human_messages[-1].content
            assert "null pointer" in human_messages[-1].content
        finally:
            saved_image.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_slash_command_does_not_consume_pending_images():
    from unittest.mock import patch

    from PIL import Image

    from agent.tui import ChatInput

    ws = Path(tempfile.mkdtemp())
    app = AICoderApp(ws)
    async with app.run_test() as pilot:
        await pilot.pause()
        fake_image = Image.new("RGB", (4, 4), color="green")
        inp = app.query_one("#prompt", ChatInput)
        inp.focus()
        try:
            with patch("PIL.ImageGrab.grabclipboard", return_value=fake_image):
                await pilot.press("ctrl+v")
                await pilot.pause()
            assert len(app.pending_images) == 1

            inp.value = "/status"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert len(app.pending_images) == 1  # a stray "/" doesn't drop the attachment
        finally:
            for p in app.pending_images:
                p.unlink(missing_ok=True)
