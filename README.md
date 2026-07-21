# marketmind-ollma-llm

Stateless final-inference microservice for the **MarketMind** workspace.

It wraps a **local Ollama model** (`qwen3:14b`, run on the host with
`ollama run qwen3:14b`) and turns a MarketMind MOS (Market Opportunity Score)
**evidence packet** into a business-facing **narrative** that explains *where to
sell* a product.

It is the explanation layer only: it **explains** the deterministic MOS results
and **never invents, estimates or recomputes** any number.

## Design

- No database — stateless LLM proxy.
- FastAPI + httpx (async) + tenacity retries.
- Talks to Ollama's chat API (`POST /api/chat`, `stream:false`, `think:false`).
- `qwen3` is a reasoning model, so any `<think>...</think>` block in the
  response is stripped before returning the narrative.

## Endpoints

- `GET /health` → `{"status":"ok","service":"marketmind-ollma-llm","ollama":"available|unavailable"}`
  (the Ollama ping is best-effort and never fails the health check).
- `POST /explain` → `ExplainRequest` → `ExplainResponse`.
  - Ollama server error → `502`; inference timeout → `504`.

### `POST /explain` request

```json
{
  "product_name": "Whey Protein",
  "language": "pt",
  "countries": [
    {
      "country": "BRA",
      "mos": 72.4,
      "mos_band": "Alta",
      "dimensions": {"demand": 80.0, "macro": 55.0},
      "signals": [
        {
          "code": "demand_volume",
          "label": "Volume de demanda",
          "score": 80.0,
          "band": "Alta",
          "meaning": "Volume absoluto de buscas mensais",
          "detail": "1526130 total monthly searches; population 212,812,405; penetration 717.1 searches per 100k inhabitants"
        }
      ]
    }
  ]
}
```

### `POST /explain` response

```json
{
  "narrative": "...",
  "model": "qwen3:14b",
  "product_name": "Whey Protein"
}
```

## Configuration

Copy `.env.example` to `.env`. Key settings:

| Variable | Default | Notes |
|---|---|---|
| `APP_PORT` | `8005` | Host/container port. |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama on the host; use `http://localhost:11434` when running this service directly on the host. |
| `OLLAMA_MODEL` | `qwen3:14b` | |
| `OLLAMA_TIMEOUT` | `300` | Local 14B inference is slow. |
| `OLLAMA_NUM_CTX` | `8192` | Context window. |

## Development

```bash
poetry install
poetry run uvicorn app.main:app --reload --port 8005
poetry run pytest
poetry run ruff check
poetry run mypy app
```

## Docker

```bash
docker compose up --build
```

Ollama is **not** part of Compose — it runs on the host and is reached through
`host.docker.internal` (declared via `extra_hosts`).
