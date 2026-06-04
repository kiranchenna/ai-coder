"""
agent/hooks.py — User-defined lifecycle hooks
==============================================
Run user shell commands on agent events. Configured in config.yaml (opt-in):

    hooks:
      PreToolUse:                 # before a tool runs; non-zero exit BLOCKS it
        - matcher: "run_shell"    # regex against the tool name (omit = all)
          command: "my-guard.sh"
      PostToolUse:                # after a tool runs (auto-format, notify, …)
        - matcher: "write_file|edit_file"
          command: "ruff format ."
      Stop:                       # when the agent finishes a turn
        - command: "notify-send 'AICoder done'"

Each command receives a JSON payload on stdin and these env vars:
  AICODER_EVENT, AICODER_TOOL, AICODER_TOOL_ARGS (JSON).

Hooks run arbitrary commands you configure — only add ones you trust.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

HOOK_TIMEOUT = 60


class HookRunner:
    def __init__(self, hooks: dict | None = None):
        if hooks is None:
            from core.config import get_config
            hooks = get_config().get("hooks", default={}) or {}
        self._hooks = hooks if isinstance(hooks, dict) else {}

    def has_any(self) -> bool:
        return any(self._hooks.get(e) for e in ("PreToolUse", "PostToolUse", "Stop"))

    # ── Events ──────────────────────────────────────────────────────────────────

    def pre_tool_use(self, tool_name: str, args: dict, cwd: Path) -> str | None:
        """Run PreToolUse hooks. Returns a block reason if any hook denies the tool."""
        for hook in self._matching("PreToolUse", tool_name):
            code, out = self._run(hook, "PreToolUse", tool_name, args, cwd)
            if code != 0:
                return out.strip() or f"denied by a PreToolUse hook (exit {code})"
        return None

    def post_tool_use(self, tool_name: str, args: dict, result: str, cwd: Path) -> str:
        """Run PostToolUse hooks. Returns any combined output to surface to the agent."""
        notes = []
        for hook in self._matching("PostToolUse", tool_name):
            _code, out = self._run(hook, "PostToolUse", tool_name, args, cwd, result=result)
            if out.strip():
                notes.append(out.strip())
        return "\n".join(notes)

    def stop(self, cwd: Path) -> None:
        """Run Stop hooks (fire-and-forget) when a turn completes."""
        for hook in self._matching("Stop", ""):
            self._run(hook, "Stop", "", {}, cwd)

    # ── Internals ───────────────────────────────────────────────────────────────

    def _matching(self, event: str, tool_name: str):
        for hook in self._hooks.get(event, []) or []:
            if not isinstance(hook, dict) or not hook.get("command"):
                continue
            matcher = hook.get("matcher")
            if not matcher or matcher == "*":
                yield hook
                continue
            try:
                if re.search(matcher, tool_name or ""):
                    yield hook
            except re.error:
                if matcher == tool_name:
                    yield hook

    def _run(self, hook: dict, event: str, tool_name: str, args: dict,
             cwd: Path, result: str = "") -> tuple[int, str]:
        payload = json.dumps({
            "event": event, "tool": tool_name, "args": args,
            "result": result[:4000] if result else "",
        })
        env = os.environ.copy()
        env["AICODER_EVENT"] = event
        env["AICODER_TOOL"] = tool_name or ""
        env["AICODER_TOOL_ARGS"] = json.dumps(args or {})
        try:
            proc = subprocess.run(
                hook["command"], shell=True, cwd=str(cwd), input=payload,
                text=True, capture_output=True, timeout=HOOK_TIMEOUT, env=env,
            )
            return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
        except Exception as e:  # noqa: BLE001 — a broken hook must not break the agent
            return 0, f"(hook '{hook.get('command')}' error: {e})"
