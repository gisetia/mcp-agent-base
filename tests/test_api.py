import json

from fastapi.testclient import TestClient


class FakeAgent:
    def __init__(self, *args, **kwargs):
        self.model = "fake-model"

    async def ask(self, messages):
        return "Simulated answer"

    async def ask_with_model_info(self, messages):
        return "Simulated answer", {"model": "fake-model", "usage": {"input_tokens": 1}}

    def ask_stream(self, messages, **kwargs):
        include_model_info = kwargs.get("include_model_info", False)

        async def gen():
            yield {"delta": {"content": "partial "}}
            yield {"delta": {"content": "answer"}}
            final = {"delta": {}, "finish_reason": "stop"}
            if include_model_info:
                final["model_info"] = {"model": "fake-model", "usage": {"input_tokens": 1}}
            yield final

        return gen()


def test_api_returns_openai_like_response(monkeypatch):
    import base_agent.api as api

    monkeypatch.setattr(api, "ClaudeMCPAgent", FakeAgent)
    client = TestClient(api.app)

    payload = {
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "model": "other",
    }
    resp = client.post("/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Simulated answer"
    assert data["model"] == "fake-model"


def test_api_streams_openai_sse(monkeypatch):
    import base_agent.api as api

    monkeypatch.setattr(api, "ClaudeMCPAgent", FakeAgent)
    client = TestClient(api.app)
    payload = {"messages": [{"role": "user", "content": "ping"}], "stream": True}
    with client.stream("POST", "/chat/completions", json=payload) as resp:
        lines = [
            line.decode() if hasattr(line, "decode") else line
            for line in resp.iter_lines()
            if line
        ]
    assert any("chat.completion.chunk" in line for line in lines)
    assert lines[-1].strip() == "data: [DONE]"


def test_api_includes_model_info_non_stream(monkeypatch):
    import base_agent.api as api

    monkeypatch.setattr(api, "ClaudeMCPAgent", FakeAgent)
    client = TestClient(api.app)

    payload = {
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "include_model_info": True,
    }
    resp = client.post("/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_info"]["model"] == "fake-model"
    assert data["model_info"]["usage"]["input_tokens"] == 1


def test_api_streams_model_info(monkeypatch):
    import base_agent.api as api

    monkeypatch.setattr(api, "ClaudeMCPAgent", FakeAgent)
    client = TestClient(api.app)
    payload = {
        "messages": [{"role": "user", "content": "ping"}],
        "stream": True,
        "include_model_info": True,
    }
    with client.stream("POST", "/chat/completions", json=payload) as resp:
        lines = [
            line.decode() if hasattr(line, "decode") else line
            for line in resp.iter_lines()
            if line
        ]
    payloads = []
    for line in lines:
        if line.strip() == "data: [DONE]":
            continue
        payloads.append(json.loads(line.split("data: ", 1)[1]))
    assert any(payload.get("model_info", {}).get("model") == "fake-model" for payload in payloads)


def test_api_requires_user_message(monkeypatch):
    import base_agent.api as api

    monkeypatch.setattr(api, "ClaudeMCPAgent", FakeAgent)
    client = TestClient(api.app)
    payload = {"messages": [{"role": "system", "content": "context"}], "stream": False}
    resp = client.post("/chat/completions", json=payload)
    assert resp.status_code == 400
