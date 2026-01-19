"""Utility functions to emit OpenAI-style chat completion payloads."""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterable, Iterable, Sequence, Any


def _base_payload(model: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
    }


def _delta_content(chunk: Any) -> Any:
    """
    Return the chunk in a shape suitable for the OpenAI delta payload.

    - Strings remain strings (legacy behavior).
    - Pre-shaped content blocks (dict/list) are passed through so callers can
      supply OpenAI-compatible structured content, e.g. [{"type": "text", ...}].
    """
    return chunk


def sse_chat_completions(
    chunk_iterable: Iterable[Any], model: str
) -> Iterable[str]:
    """Yield SSE-formatted chunks that mimic the OpenAI Chat Completions API."""
    base = _base_payload(model)
    for chunk in chunk_iterable:
        payload = {
            **base,
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": _delta_content(chunk)},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(payload)}\n\n"

    yield "data: [DONE]\n\n"


def chat_completion_response(
    chunk_iterable: Sequence[Any],
    model: str,
    prompt_messages: Sequence[str],
    model_info: Any = None,
) -> dict:
    """Return a one-shot JSON response matching the OpenAI API shape."""
    if all(isinstance(chunk, str) for chunk in chunk_iterable):
        content: Any = "".join(chunk_iterable)
    elif len(chunk_iterable) == 1:
        # If a single structured chunk (dict/list) was provided, pass it straight through.
        content = chunk_iterable[0]
    else:
        # For structured content (e.g., list[{"type": "text", ...}]) pass through.
        # If multiple structured chunks arrive, keep them as a list in order.
        content = list(chunk_iterable)
    payload = _base_payload(model)
    payload.update(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    "prompt": prompt_messages,
                }
            ],
        }
    )
    if model_info is not None:
        payload["model_info"] = model_info
    return payload


async def async_sse_chat_completions(
    chunk_async_iterable: AsyncIterable[Any], model: str
) -> AsyncIterable[str]:
    """Async generator yielding SSE-formatted chunks for async sources."""
    base = _base_payload(model)
    async for chunk in chunk_async_iterable:
        # Allow callers to pass either raw text, or a structured chunk that already
        # contains delta/finish_reason/tool_calls, etc.
        choice = {"index": 0, "delta": {}, "finish_reason": None}
        if isinstance(chunk, dict):
            if "choices" in chunk:
                payload = {**base, **chunk}
                yield f"data: {json.dumps(payload)}\n\n"
                continue
            delta = chunk.get("delta", {})
            choice["delta"] = delta if delta is not None else {}
            if "finish_reason" in chunk:
                choice["finish_reason"] = chunk.get("finish_reason")
            model_info = chunk.get("model_info")
        else:
            choice["delta"] = {"content": _delta_content(chunk)}
            model_info = None

        payload = {
            **base,
            "object": "chat.completion.chunk",
            "choices": [choice],
        }
        if model_info is not None:
            payload["model_info"] = model_info
        yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"
