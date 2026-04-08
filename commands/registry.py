"""
commands/registry.py — Slash command registry for aicoder
==========================================================
Commands are registered with: name, description, usage, and handler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class Command:
    """A registered slash command."""
    name: str                  # e.g. "build"
    description: str           # Short description shown in /help
    usage: str                 # e.g. "/build <idea>"
    handler: Callable[..., Any]
    aliases: list[str] = field(default_factory=list)


class CommandRegistry:
    def __init__(self):
        self._commands: dict[str, Command] = {}

    def register(
        self,
        name: str,
        description: str,
        usage: str,
        handler: Callable,
        aliases: list[str] | None = None,
    ) -> None:
        cmd = Command(
            name=name,
            description=description,
            usage=usage,
            handler=handler,
            aliases=aliases or [],
        )
        self._commands[name] = cmd
        for alias in cmd.aliases:
            self._commands[alias] = cmd

    def get(self, name: str) -> Command | None:
        return self._commands.get(name)

    def all_commands(self) -> list[Command]:
        """Return unique commands (no aliases)."""
        seen: set[str] = set()
        result: list[Command] = []
        for cmd in self._commands.values():
            if cmd.name not in seen:
                seen.add(cmd.name)
                result.append(cmd)
        
        def sort_key(c: Command):
            if c.name == "project":
                return "000_project"
            if c.name == "build":
                return "001_build"
            return c.name

        return sorted(result, key=sort_key)


# ─── Global registry instance ─────────────────────────────────────────────────

registry = CommandRegistry()
