"""FastAPI wrapper exposing a Claude+MCP agent with an OpenAI-compatible interface."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, StreamingResponse

from .agent import ClaudeMCPAgent
from .openai_adapter import chat_completion_response, async_sse_chat_completions
from .settings import SIMULATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g., user).")
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: bool = True
    simulate: Optional[bool] = None
    include_model_info: bool = False


def _extract_user_question(messages: List[Message]) -> Optional[str]:
    """Return the most recent user message content."""
    for message in reversed(messages):
        if message.role.lower() == "user":
            return message.content
    return None

@app.post("/chat/completions")
async def chat_completions(chat_request: ChatCompletionRequest):
    logger.info("incoming request %s", chat_request.model_dump(exclude_none=True))

    question = _extract_user_question(chat_request.messages)
    if not question:
        raise HTTPException(
            status_code=400, detail="A user message is required to ask a question."
        )

    # Convert incoming Pydantic messages to the format expected by the Anthropic client.
    conversation = [
        {"role": message.role, "content": message.content} for message in chat_request.messages
    ]

    agent = ClaudeMCPAgent(simulate=SIMULATE if chat_request.simulate is None else chat_request.simulate)

    if chat_request.stream:
        stream = agent.ask_stream(
            conversation,
            include_model_info=chat_request.include_model_info,
        )
        response = StreamingResponse(
            async_sse_chat_completions(stream, agent.model),
            media_type="text/event-stream",
        )
        return response

    if chat_request.include_model_info:
        answer, model_info = await agent.ask_with_model_info(conversation)
    else:
        answer = await agent.ask(conversation)
        model_info = None

    payload = chat_completion_response(
        chunk_iterable=[answer],
        model=agent.model,
        prompt_messages=[message.content for message in chat_request.messages],
        model_info=model_info,
    )

    return JSONResponse(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("base_agent.api:app", host="0.0.0.0", port=5000, reload=False)
