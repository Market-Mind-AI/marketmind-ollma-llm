# marketmind-ollma-llm

Stateless **Ollama gateway** for MarketMind. It forwards text (`/chat`) and
vision (`/vision`) inference to a local Ollama server and returns the model's
answer with reasoning blocks stripped. That is all it does.

It knows nothing about the MOS, the evidence packet or the narrative: the
caller — today the scoring service's `OllamaProvider` — owns the prompt, the
output schema and the choice of model. This service is transport.

## Design

- No database, no state between requests.
- FastAPI + httpx (async) + tenacity.
- Talks to Ollama's chat API (`POST /api/chat`, `stream:false`, `think:false`).
- `qwen3` is a reasoning model, so any `<think>...</think>` block in the reply
  is stripped before it is returned; a reply that is empty afterwards is an
  error, not an empty answer.
- **Retries**: 3 attempts, exponential backoff (2–30 s), only on timeouts and
  Ollama 5xx. A 4xx (unknown model, bad request), a reply that does not parse,
  an empty answer or an unreachable host fail at once — asking again would not
  change them.

## Routes

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | `{"status":"ok","service":"marketmind-ollma-llm","ollama":"available\|unavailable"}`. The Ollama ping is best-effort and never fails the check. |
| `POST` | `/chat` | Text inference. |
| `POST` | `/vision` | Image + prompt inference against a vision model (e.g. `qwen3-vl:8b-instruct`). |

Errors: Ollama timeout → `504`; anything else Ollama refused or answered
badly → `502`.

### `POST /chat`

```json
{
  "model": "qwen3:14b",
  "system": "You are a helpful assistant.",
  "prompt": "Say hello.",
  "format": {"type": "object", "properties": {"answer": {"type": "string"}}},
  "temperature": 0.3,
  "num_ctx": 8192
}
```

- `model` is the caller's choice; omitted means `OLLAMA_MODEL`.
- `format` is a JSON Schema, sent as Ollama's structured-outputs `format`.
- `num_ctx` falls back to `OLLAMA_NUM_CTX`.

### `POST /vision`

Same fields plus `image_base64` (a `data:...;base64,` prefix is stripped) and
`image_media_type`, which is accepted for compatibility with callers that send
it and not used — Ollama takes raw base64.

### Response

```json
{"content": "Hello there!", "model": "qwen3:14b"}
```

## Configuration

Copy `.env.example` to `.env`.

| Variable | Default | Notes |
|---|---|---|
| `APP_PORT` | `8005` | Read by `docker-compose.yml` only; the app listens on 8005. |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama on the host; `http://localhost:11434` when running this service on the host too. |
| `OLLAMA_MODEL` | `qwen3:14b` | Only the default when a request names no model. |
| `OLLAMA_TIMEOUT` | `300` | Seconds. Local inference on a 14B model is slow. |
| `OLLAMA_NUM_CTX` | `32768` | Context window when the request gives none. |
| `LOG_LEVEL` | `INFO` | |

## Development

```bash
poetry install
poetry run uvicorn app.main:app --reload --port 8005
```

The repository has no virtualenv of its own; the tests run with any
interpreter that has fastapi, httpx, respx, tenacity, pytest and
pytest-asyncio — the orchestrator's does:

```bash
/Users/alvaroneto/Projects/Mestrado/marketmind-orchestrator/.venv/bin/python -m pytest -q -p no:cacheprovider
```

The tests patch the retry backoff out, so the suite runs in well under a second.

## Docker

```bash
docker compose up --build
```

Ollama is **not** part of Compose — it runs on the host and is reached through
`host.docker.internal` (declared via `extra_hosts`).
