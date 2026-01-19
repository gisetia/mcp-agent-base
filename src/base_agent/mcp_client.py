"""Thin MCP client used by the Claude agent to discover and call tools."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List

from .settings import CLIENT_NAME

try:  # Optional dependency; used only for more specific error detection.
    import httpcore  # type: ignore
    import httpx  # type: ignore
except Exception:  # pragma: no cover - fallback if httpx isn't available
    httpcore = None
    httpx = None


class MCPConnectionError(RuntimeError):
    """Raised when the MCP server cannot be reached after retries."""


def _iter_exceptions(exc: BaseException):
    if isinstance(exc, BaseExceptionGroup):
        for nested in exc.exceptions:
            yield from _iter_exceptions(nested)
    else:
        yield exc


@dataclass
class MCPToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]

    def to_anthropic(self) -> Dict[str, Any]:
        """Return the format expected by the Anthropic Messages API."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class MCPClient:
    """Wrapper around the MCP SSE client."""

    def __init__(self, server_url: str, max_retries: int = 3, retry_backoff_seconds: float = 0.5):
        self.server_url = server_url
        self._max_retries = max(1, max_retries)
        self._retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        try:
            from mcp.client.session import ClientSession  # type: ignore
            from mcp.client.streamable_http import streamablehttp_client  # type: ignore
            from mcp import types  # type: ignore
        except ImportError as exc:  # pragma: no cover - import-time guard
            raise RuntimeError(
                "The 'mcp' package is required to talk to MCP servers. "
                "Install it with `pip install mcp`."
            ) from exc

        self._ClientSession = ClientSession
        self._transport_client = streamablehttp_client
        self._types = types

    def _is_connection_error(self, exc: BaseException) -> bool:
        for err in _iter_exceptions(exc):
            if isinstance(err, (ConnectionError, TimeoutError)):
                return True
            if httpx is not None and isinstance(err, httpx.ConnectError):
                return True
            if httpcore is not None and isinstance(err, httpcore.ConnectError):
                return True
        return False

    async def _with_retries(self, action: str, func: Callable[[], Awaitable[Any]]) -> Any:
        last_exc: BaseException | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return await func()
            except Exception as exc:
                if not self._is_connection_error(exc):
                    raise
                last_exc = exc
                if attempt < self._max_retries and self._retry_backoff_seconds:
                    await asyncio.sleep(self._retry_backoff_seconds * attempt)
        raise MCPConnectionError(
            f"Failed to {action} from MCP server at {self.server_url} "
            f"after {self._max_retries} attempts."
        ) from last_exc

    @asynccontextmanager
    async def _session(self):
        """Yield an initialized MCP session."""
        async with self._transport_client(self.server_url) as (
            read_stream,
            write_stream,
            _get_session_id,
        ):
            client_info = self._types.Implementation(name=CLIENT_NAME, version="0.1.0")
            async with self._ClientSession(
                read_stream, write_stream, client_info=client_info
            ) as session:
                await session.initialize()
                yield session

    async def list_tools(self) -> List[MCPToolDefinition]:
        """Return the tool definitions exposed by the MCP server."""
        async def _fetch() -> List[MCPToolDefinition]:
            async with self._session() as session:
                response = await session.list_tools()
                tools = []
                for tool in response.tools:
                    # The MCP types use camelCase; Anthropic expects snake_case.
                    input_schema = getattr(tool, "inputSchema", None) or getattr(
                        tool, "input_schema", None
                    )
                    if input_schema is None:
                        input_schema = {"type": "object"}
                    tools.append(
                        MCPToolDefinition(
                            name=tool.name,
                            description=tool.description or "",
                            input_schema=input_schema,
                        )
                    )
                return tools

        return await self._with_retries("list tools", _fetch)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Invoke a specific tool and return a readable string result."""
        async def _call() -> str:
            async with self._session() as session:
                response = await session.call_tool(tool_name, arguments)
                # Collect text content; fall back to a repr if necessary.
                rendered: List[str] = []
                for item in response.content:
                    text = getattr(item, "text", None) or getattr(item, "data", None)
                    if text is None:
                        text = str(item)
                    rendered.append(text)
                return "\n".join(rendered)

        return await self._with_retries(f"call tool '{tool_name}'", _call)


class EmptyMCPClient:
    """Mock MCP client that advertises no tools and does not allow calls."""

    def __init__(self, server_url: str = "mock://empty"):
        self.server_url = server_url

    async def list_tools(self) -> List[MCPToolDefinition]:
        return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        raise RuntimeError("call_tool should not be invoked when no tools are available")


async def discover_tools(server_url: str) -> List[MCPToolDefinition]:
    """Convenience helper to fetch tools outside of a class instance."""
    client = MCPClient(server_url)
    return await client.list_tools()


async def call_tool(server_url: str, tool_name: str, arguments: Dict[str, Any]) -> str:
    """Convenience helper to invoke a tool outside of a class instance."""
    client = MCPClient(server_url)
    return await client.call_tool(tool_name, arguments)
