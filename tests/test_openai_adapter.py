import json

import pytest

from base_agent.openai_adapter import (
    async_sse_chat_completions,
    chat_completion_response,
    sse_chat_completions,
)


def test_sse_chat_completions_formats_payload_and_done_marker():
    chunks = list(sse_chat_completions(["hi", " there"], model="test-model"))
    assert chunks[-1].strip() == "data: [DONE]"

    payload = json.loads(chunks[0].split("data: ", 1)[1])
    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"][0]["delta"]["content"] == "hi"
    assert payload["model"] == "test-model"


def test_chat_completion_response_passes_through_structured_content():
    structured_chunk = [{"type": "text", "text": "hello"}]
    payload = chat_completion_response(
        chunk_iterable=structured_chunk, model="model-x", prompt_messages=["p1"]
    )
    message = payload["choices"][0]["message"]
    assert message["content"] == structured_chunk[0]
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["model"] == "model-x"


@pytest.mark.asyncio
async def test_async_sse_chat_completions_handles_preformed_chunks():
    async def gen():
        yield {"delta": {"content": "piece"}, "finish_reason": None}
        # Already-shaped chunk should be passed straight through.
        yield {
            "choices": [
                {"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}
            ],
            "object": "chat.completion.chunk",
        }

    payload_lines = []
    async for chunk in async_sse_chat_completions(gen(), "model-async"):
        payload_lines.append(chunk.strip())

    first_payload = json.loads(payload_lines[0].split("data: ", 1)[1])
    assert first_payload["choices"][0]["delta"]["content"] == "piece"

    passthrough_payload = json.loads(payload_lines[1].split("data: ", 1)[1])
    assert passthrough_payload["choices"][0]["finish_reason"] == "stop"
    assert payload_lines[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_async_sse_chat_completions_passes_tool_calls_and_model_info():
    async def gen():
        yield {
            "delta": {
                "tool_calls": [
                    {
                        "id": "tool-1",
                        "type": "function",
                        "function": {"name": "demo", "arguments": "{}"},
                    }
                ]
            },
            "finish_reason": None,
            "model_info": {"usage": {"input_tokens": 1}},
        }
        yield {"delta": {}, "finish_reason": "stop", "model_info": {"usage": {"input_tokens": 1}}}

    payload_lines = []
    async for chunk in async_sse_chat_completions(gen(), "model-async"):
        payload_lines.append(chunk.strip())

    first_payload = json.loads(payload_lines[0].split("data: ", 1)[1])
    assert first_payload["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "demo"
    assert first_payload["model_info"]["usage"]["input_tokens"] == 1

    second_payload = json.loads(payload_lines[1].split("data: ", 1)[1])
    assert second_payload["choices"][0]["finish_reason"] == "stop"
    assert second_payload["model_info"]["usage"]["input_tokens"] == 1
    assert payload_lines[-1] == "data: [DONE]"
