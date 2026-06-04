"""
agent/mcp_client.py — MCP client support
=========================================
Connects to configured MCP (Model Context Protocol) servers and exposes their
tools to the agent alongside the built-in toolset.

The MCP Python SDK is async; the agent loop is synchronous. So we run all MCP
sessions on a dedicated background event loop (in a thread) and bridge each tool
call from the sync loop with run_coroutine_threadsafe. Sessions are kept open
for the lifetime of the AICoder session and closed on shutdown.

Configured opt-in via config.yaml:

    mcp:
      servers:
        filesystem:
          command: npx
          args: ["-y", "@modelcontextprotocol/server-filesystem", "/path"]

Requires the optional `mcp` package (pip install "ai-coder[mcp]").
"""

from __future__ import annotations

import asyncio
import threading

from rich.console import Console

console = Console()

CONNECT_TIMEOUT = 30.0
CALL_TIMEOUT = 120.0


def _content_to_text(result) -> str:
    """Flatten an MCP tool result's content blocks into plain text."""
    if getattr(result, "isError", False):
        prefix = "ERROR: "
    else:
        prefix = ""
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else str(block))
    return prefix + ("\n".join(parts) if parts else "(no output)")


def _schema_to_model(name: str, schema: dict):
    """Build a pydantic model from an MCP tool's JSON-schema inputSchema."""
    from pydantic import create_model

    type_map = {
        "string": str, "number": float, "integer": int,
        "boolean": bool, "array": list, "object": dict,
    }
    props = (schema or {}).get("properties", {}) or {}
    required = set((schema or {}).get("required", []) or [])
    fields = {}
    for pname, pdef in props.items():
        if not pname.isidentifier():
            continue
        pytype = type_map.get((pdef or {}).get("type"), str)
        fields[pname] = (pytype, ... if pname in required else None)
    return create_model(name, **fields) if fields else create_model(name)


class MCPManager:
    """Owns the background event loop and the connected MCP sessions."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._sessions: dict = {}
        self._shutdown_events: dict = {}
        self._discovered: list[tuple[str, object]] = []  # (server_name, tool_def)
        self._started = False

    # ── Construction ───────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls) -> "MCPManager":
        from core.config import get_config

        servers = get_config().get("mcp", "servers", default={}) or {}
        mgr = cls()
        if servers:
            mgr.start(servers)
        return mgr

    def start(self, servers: dict) -> None:
        try:
            import mcp  # noqa: F401
        except ImportError:
            console.print(
                "[yellow]MCP servers are configured but the 'mcp' package isn't installed. "
                'Run: pip install "ai-coder[mcp]". Skipping MCP.[/yellow]'
            )
            return

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._started = True

        for name, spec in servers.items():
            try:
                self._submit(self._connect(name, spec or {})).result(timeout=CONNECT_TIMEOUT)
            except Exception as e:  # noqa: BLE001
                console.print(f"[yellow]MCP server '{name}' failed to start: {e}[/yellow]")

        if self._discovered:
            n_tools = len(self._discovered)
            servers_ok = sorted({s for s, _ in self._discovered})
            console.print(
                f"[dim]🔌 MCP: {n_tools} tool(s) from {', '.join(servers_ok)}[/dim]"
            )

    # ── Background loop ─────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _connect(self, name: str, spec: dict) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=spec.get("command", ""),
            args=spec.get("args", []) or [],
            env=spec.get("env") or None,
        )
        ready = asyncio.Event()
        shutdown = asyncio.Event()
        self._shutdown_events[name] = shutdown
        captured: dict = {}

        async def runner():
            try:
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self._sessions[name] = session
                        resp = await session.list_tools()
                        for tool in resp.tools:
                            self._discovered.append((name, tool))
                        ready.set()
                        await shutdown.wait()
            except Exception as e:  # noqa: BLE001
                captured["error"] = e
                ready.set()

        self._loop.create_task(runner())
        await ready.wait()
        if "error" in captured:
            raise captured["error"]

    # ── Tool calls ──────────────────────────────────────────────────────────────

    def call_tool(self, server: str, tool_name: str, args: dict) -> str:
        try:
            return self._submit(self._call(server, tool_name, args)).result(timeout=CALL_TIMEOUT)
        except Exception as e:  # noqa: BLE001
            return f"ERROR calling MCP tool {server}.{tool_name}: {e}"

    async def _call(self, server: str, tool_name: str, args: dict) -> str:
        session = self._sessions.get(server)
        if session is None:
            return f"ERROR: MCP server '{server}' is not connected."
        result = await session.call_tool(tool_name, args or {})
        return _content_to_text(result)

    # ── LangChain tools ─────────────────────────────────────────────────────────

    def langchain_tools(self) -> list:
        """Expose discovered MCP tools as LangChain tools (prefixed by server)."""
        from langchain_core.tools import StructuredTool

        tools = []
        for server, tdef in self._discovered:
            full_name = f"{server}__{tdef.name}"

            def _make(srv: str, tname: str):
                def _fn(**kwargs):
                    return self.call_tool(srv, tname, kwargs)
                return _fn

            try:
                schema = _schema_to_model(full_name, getattr(tdef, "inputSchema", {}) or {})
                tools.append(StructuredTool.from_function(
                    func=_make(server, tdef.name),
                    name=full_name,
                    description=(tdef.description or f"{tdef.name} (via MCP server '{server}')")[:1024],
                    args_schema=schema,
                ))
            except Exception:  # noqa: BLE001 — skip a malformed tool, keep the rest
                continue
        return tools

    # ── Shutdown ────────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        if not self._started or self._loop is None:
            return
        for ev in self._shutdown_events.values():
            try:
                self._loop.call_soon_threadsafe(ev.set)
            except Exception:  # noqa: BLE001
                pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:  # noqa: BLE001
            pass
        self._started = False
