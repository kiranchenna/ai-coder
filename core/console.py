"""
core/console.py — A Console that can't be crashed by malformed markup
=======================================================================
Any f-string embedding dynamic content (an exception message, a tool
argument, a user's own typed text) into `console.print(f"[red]{e}[/red]")`
can contain something that *looks* like a Rich markup tag but isn't valid
(e.g. code with mismatched `[...]`/`<...>`). rich.console.Console.print
raises MarkupError in that case — confirmed live, this took down the whole
plain REPL, including the error-reporting code path itself (which embeds
the very content that broke the first parse). SafeConsole degrades to
plain text instead of crashing.
"""

from __future__ import annotations

from rich.console import Console
from rich.errors import MarkupError


class SafeConsole(Console):
    def print(self, *args, **kwargs) -> None:
        try:
            super().print(*args, **kwargs)
        except MarkupError:
            kwargs["markup"] = False
            super().print(*args, **kwargs)
