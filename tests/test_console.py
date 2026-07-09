"""
tests/test_console.py — SafeConsole never crashes on malformed markup
=======================================================================
Live-reproduced bug: rich.console.Console.print raises MarkupError when
content contains something that looks like a Rich markup tag but isn't
valid (e.g. code with mismatched `[...]`/`<...>`). This took down the
plain REPL entirely, including the error-reporting code path itself
(which embeds the very content that broke the first parse). Every
`console = Console()` in the codebase was replaced with SafeConsole.
"""

from core.console import SafeConsole


def test_safe_console_does_not_raise_on_malformed_markup(capsys):
    console = SafeConsole(width=200)
    # the exact string that crashed the app live
    console.print("[red]⚠ Error: closing tag '[/<m>]' at position 3334 doesn't match any open tag[/red]")
    out = capsys.readouterr().out
    assert "doesn't match any open tag" in out


def test_safe_console_still_renders_valid_markup(capsys):
    console = SafeConsole()
    console.print("[bold]hello[/bold]")
    out = capsys.readouterr().out
    assert "hello" in out


def test_safe_console_falls_back_with_other_kwargs_intact(capsys):
    console = SafeConsole()
    # end="" must still be honored on the fallback path, not silently dropped
    console.print("[broken <tag]", end="")
    console.print("next")
    out = capsys.readouterr().out
    assert "next" in out
