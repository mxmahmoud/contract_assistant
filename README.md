# Contract Assistant

An AI contract analysis and Q&A system designed to run containerized by default (Ollama + TEI + LiteLLM + Streamlit), with optional online mode via OpenAI.

## Features

- **Containerized by default**: One command spins up the entire stack
- **Dual mode**: Local (offline) or OpenAI (online) without code changes
- **Grounded QA**: Retrieval-augmented Q&A over your contracts
- **NER**: Extract parties, dates, monetary values
- **Persistent storage**: Contracts and vectors persist across restarts
- **Web UI + CLI**: Streamlit app and Typer CLI
## Quick Start

1) Clone and configure
```bash
git clone <repository-url>
cd contract-assistant
cp env.example .env
# Edit .env as needed (defaults target local container stack)
```

2) Start the stack (dev)
```bash
make dev-up
```

3) First run behavior
- Ollama model is pulled automatically by the container entry script on first start (no need to visit the UI).
- TEI downloads embeddings artifacts on first start.

4) Open the app
```
http://localhost:8501
```

Notes
- Services: `app` (Streamlit), `litellm`, `ollama`, `tei`
- Persistence:
  - Vectors: `data/chroma_data`
  - Contracts (PDFs, metadata, entities): `data/contracts`
  - Ollama cache: `data/weights/ollama_data`
  - TEI cache: `data/weights/tei_data`
- Default local models: LLM `ollama/llama3.2:3b`, embeddings `intfloat/multilingual-e5-large-instruct`

Stop the stack
```bash
make dev-down
```

## Using the App

- Upload a PDF to ingest. It will be chunked, embedded, and registered persistently.
- The sidebar lists previously ingested contracts; select and click “Load Contract.”
- Ask questions in chat; responses are grounded in retrieved excerpts.

## CLI Usage (Inside Containers)

Run CLI inside the `app` container in dev mode:
```bash
docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.dev.yml exec app \
  poetry run typer cli/main.py ingest data/examples/sample_nda.pdf

docker compose -f docker/docker-compose.base.yml -f docker/docker-compose.dev.yml exec app \
  poetry run typer cli/main.py ask "What is the governing law?" --id <contract_id>
```

## Optional: Online Mode (OpenAI)

If you prefer OpenAI instead of local models:
1) Set in `.env`:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```
2) Run the app locally (no containers needed):
```bash
make setup
make run-app
```

## Docker Architecture

- Compose files
  - `docker/docker-compose.base.yml`: common services & volumes
  - `docker/docker-compose.dev.yml`: dev overrides (hot reload, volumes)
  - `docker/docker-compose.prod.yml`: prod overrides (restart policies)
- Services
  - `app`: Streamlit frontend and orchestration
  - `litellm`: OpenAI-compatible router. Healthchecked via `/health/readiness` (per LiteLLM docs)
  - `ollama`: local LLM server. Entry script pulls `LOCAL_LLM_MODEL` if missing
  - `tei`: embeddings server (intfloat/multilingual-e5-large-instruct)

## Configuration & Commands

- Make targets
  - Dev: `dev-up`, `dev-down`, `dev-logs`, `dev-shell`
  - Prod: `prod-up`, `prod-down`, `prod-logs`, `prod-shell`
  - Local run (OpenAI): `run-app`
  - Code quality: `setup`, `format`, `lint`
- See [CONFIGURATION.md](CONFIGURATION.md) for environment variables and advanced options. Note that document chunking is dynamic based on your embedding model's token limits, which can be configured in your `.env` file.

## Persistence Details

- Vector store: ChromaDB persistent mode at `data/chroma_data`
- Contract registry: `data/contracts/<contract_id>/` with
  - Original PDF
  - `meta.json` (filename, pages, timestamps)
  - `entities.json` (NER output)

These paths are mounted into containers via bind volumes to persist across restarts.