# MCP Agent Base

Minimal FastAPI service that exposes an OpenAI-compatible `/chat/completions` endpoint backed by Anthropic Claude with MCP tool calls and SSE streaming.

## How to run
- Requirements: Python 3.10+. Install deps: `pip install -e .` (or `pip install -e .[dev]` for tests).
- Local server: `uvicorn base_agent.api:app --reload --port 5000`
- Docker: `docker compose up --build` (binds `5000:5000`, mounts `./prompts` read-only).
- The system prompt is always read from `prompts/system_prompt.txt`; edit that file to change behavior.

## Environment variables
- `ANTHROPIC_API_KEY`: required unless `SIMULATE=true`.
- `MCP_SERVER_URL`: MCP server root used for tool discovery/invocation.
- `DEFAULT_MODEL`: Anthropic model name (e.g., `claude-3-haiku-20240307`).
- `DEFAULT_TEMPERATURE`, `DEFAULT_MAX_OUTPUT_TOKENS`: generation defaults forwarded to Anthropic.
- `THINKING_ENABLED`, `THINKING_BUDGET_TOKENS`: toggle and budget for Claude “thinking” mode.
- `INCLUDE_TOOL_LOGS`: when true, tool inputs/outputs are echoed into the answer.
- `SIMULATE`: when true, skip Anthropic/MCP and return a canned response.
- `CLIENT_NAME`: identifier sent to MCP servers (defaults to `mcp-agent-base`).
- `MOCK_EMPTY_MCP`: when true, advertises zero MCP tools without contacting a server.

## API usage
- Endpoint: `POST /chat/completions`
- Accepted params:
  - `messages`: list of `{role, content}` (required).
  - `stream`: bool, default `true` (controls SSE vs one-shot JSON).
  - `simulate`: optional bool to override the `SIMULATE` env var for a single request.
- `include_model_info`: bool, default `false` (attach model settings + usage stats to the response).
- Ignored params: model, temperature, max_tokens, top_p, presence_penalty, frequency_penalty, stop, tools, tool_choice, and other OpenAI fields.

Non-streaming example:
```bash
curl -X POST http://localhost:5000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"stream":false}'
```

Streaming (SSE) example:
```bash
curl -N -X POST http://localhost:5000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"stream":true}'
```

Streaming yields OpenAI-shaped `chat.completion.chunk` events for assistant text, tool calls, and tool results, followed by `data: [DONE]`. When `include_model_info=true`, the final chunk includes a `model_info` object, and structured deltas (such as `tool_calls` and `finish_reason`) pass through in the SSE payloads.

## Simulation mode
- Set `SIMULATE=true` to bypass Anthropic and MCP entirely and return the canned string `Simulated response.` (streaming still emits OpenAI-style chunks). Useful for local testing without secrets or MCP server access.

## Running tests
- Install dev deps: `pip install -e .[dev]`
- Run: `pytest`

## Project layout
- `src/base_agent/agent.py` — Claude + MCP agent loop (streaming/tool use, simulation).
- `src/base_agent/api.py` — FastAPI wiring for `/chat/completions`.
- `src/base_agent/openai_adapter.py` — helpers to format SSE and JSON responses like OpenAI.
- `src/base_agent/mcp_client.py` — thin MCP client for tool discovery and invocation.
- `src/base_agent/settings.py` — environment/config defaults and system prompt loader.
- `prompts/system_prompt.txt` — system prompt (only source used).
- `tests/` — pytest suite with fakes for Anthropic/MCP.
- `docker-compose.yml`, `Dockerfile`, `.env.example` — deployment helpers.

## Notes
- MCP tools are discovered at request time; make sure `MCP_SERVER_URL` is reachable.
- When `THINKING_ENABLED=true`, the agent forwards a thinking budget to Claude.
- The service retries transient MCP/LLM connection failures with backoff before returning an error.
