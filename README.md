# RAG-ROLE-ENTERPRISE

## Project Structure

- `api/` - REST API layer
- `services/` - Business logic
- `repositories/` - Database access layer
- `models/` - ORM models
- `workers/` - Celery tasks
- `utils/` - Helper utilities

## Tech Stack

FastAPI, MySQL, Redis, Celery, MinIO, ChromaDB, Docker Compose

## Run the Project

```bash
docker-compose up --build
```

## Access Links

- API Backend: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- MinIO: http://localhost:9001

## Data Storage

- MySQL: database storage
- ChromaDB: stored in the API service volume at `data/chromadb/chroma.sqlite3`

## LLM Integration (OpenAI + Olama local)

This project can optionally call an LLM when answering chat messages. Two providers are supported:

- OpenAI (remote) ? set `LLM_PROVIDER=openai` and provide `OPENAI_API_KEY`. Optionally set `OPENAI_API_BASE` for a custom base URL and `OPENAI_MODEL` to choose the model.
- Olama (local) ? set `LLM_PROVIDER=olama` and provide `OLAMA_URL` (e.g. `http://localhost:11434`) and `OLAMA_MODEL`.

If no LLM is configured the system will fallback to a minimal answer generator that returns concatenated excerpts from retrieved documents.

Environment variables used by the LLM integration:

- `LLM_PROVIDER` ? `openai` or `olama` (default: unset)
- `OPENAI_API_KEY` ? OpenAI API key
- `OPENAI_API_BASE` ? (optional) OpenAI base URL
- `OPENAI_MODEL` ? (optional) model name, default `gpt-4o-mini`
- `OLAMA_URL` ? Olama HTTP endpoint (e.g. `http://localhost:11434`)
- `OLAMA_MODEL` ? Olama model name
- `LLM_TIMEOUT_SECONDS` ? HTTP timeout for LLM calls (default 30)

Quick example (.env):

```
LLM_PROVIDER=olama
OLAMA_URL=http://localhost:11434
OLAMA_MODEL=ggml-vicuna-13b
```

To test OpenAI locally, set `LLM_PROVIDER=openai` and `OPENAI_API_KEY` then restart the service.

## Production deployment on one EC2 instance

The production stack keeps the API, Celery worker, MySQL, Redis, MinIO, ChromaDB,
OpenFGA, and its PostgreSQL datastore on one EC2 instance. Caddy provides HTTPS
for the API domain and forwards requests to FastAPI.

1. Copy the project to the EC2 instance and enter this directory:

```bash
cd back-end
cp .env.prod.example .env
```

2. Replace every `change-me` value in `.env`, set `API_DOMAIN` to the DNS name
   pointing to the EC2 Elastic IP, and set `CORS_ORIGINS` to the deployed front-end URL.

3. Start the production stack:

```bash
docker compose -f docker-compose.prod.yml run --rm openfga-migrate
docker compose -f docker-compose.prod.yml up -d --build
```

4. Create the OpenFGA store and model once, then copy the printed IDs into `.env`:

```bash
docker compose -f docker-compose.prod.yml exec api python -m app.fga.setup
docker compose -f docker-compose.prod.yml up -d
```

5. Check the API:

```bash
curl https://api.example.com/health
docker compose -f docker-compose.prod.yml logs -f api
```

Do not expose MySQL, Redis, PostgreSQL, MinIO, ChromaDB, OpenFGA, or Neo4j ports
in the EC2 security group — Neo4j's Bolt/Browser ports have no app-level auth
of their own, unlike the API. Keep `.env` outside Git and back up the Docker
volumes (including `neo4j_data`, which holds the ontology graph).
