import pytest

from base_agent.mcp_client import MCPClient, MCPConnectionError


class DummyClient:
    def __init__(self, max_retries=3, backoff=0.0):
        self._max_retries = max_retries
        self._retry_backoff_seconds = backoff
        self.server_url = "http://example.test"
        self.calls = 0

    def _is_connection_error(self, exc: BaseException) -> bool:
        return isinstance(exc, RuntimeError)


@pytest.mark.asyncio
async def test_with_retries_eventually_succeeds():
    client = DummyClient(max_retries=3, backoff=0.0)

    async def flaky_call():
        client.calls += 1
        if client.calls < 3:
            raise RuntimeError("temporary failure")
        return "ok"

    result = await MCPClient._with_retries(client, "list tools", flaky_call)
    assert result == "ok"
    assert client.calls == 3


@pytest.mark.asyncio
async def test_with_retries_raises_after_max_attempts():
    client = DummyClient(max_retries=2, backoff=0.0)

    async def always_fails():
        client.calls += 1
        raise RuntimeError("temporary failure")

    with pytest.raises(MCPConnectionError):
        await MCPClient._with_retries(client, "list tools", always_fails)
    assert client.calls == 2
