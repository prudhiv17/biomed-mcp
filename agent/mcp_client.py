import os
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from agent import sessions

ROOT = Path(__file__).resolve().parent.parent


class ToolBridge:
    def __init__(self, session: ClientSession, tools: list[Any], session_id: str | None) -> None:
        self.session = session
        self.session_id = session_id
        self._tools = tools

    @property
    def names(self) -> list[str]:
        return [t.name for t in self._tools]

    def openai_tools(self) -> list[dict[str, Any]]:
        # Whatever the server advertises becomes the model's tool surface; nothing here
        # names a specific tool, so adding one server-side needs no client change.
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            for tool in self._tools
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        started = time.monotonic()
        try:
            result = await self.session.call_tool(name, arguments)
        except Exception as exc:
            elapsed = int((time.monotonic() - started) * 1000)
            sessions.log_call(self.session_id, name, "error", elapsed)
            return f"Tool {name} failed: {type(exc).__name__}: {exc}", True

        elapsed = int((time.monotonic() - started) * 1000)
        text = "\n".join(part.text for part in result.content if getattr(part, "text", None))
        sessions.log_call(
            self.session_id, name, "error" if result.is_error else "ok", elapsed
        )
        return text, bool(result.is_error)

    async def read_resource(self, uri: str) -> str:
        result = await self.session.read_resource(uri)
        return "\n".join(part.text for part in result.contents if getattr(part, "text", None))


@asynccontextmanager
async def connect(session_id: str | None = None) -> AsyncIterator[ToolBridge]:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.main", "stdio"],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        listed = await session.list_tools()
        yield ToolBridge(session, listed.tools, session_id)
