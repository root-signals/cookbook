# RAG Eval Example

A minimal RAG agent that answers questions about the [Scorable](https://scorable.ai) documentation, with two approaches to evaluation: vibe-coded LLM-as-judge and Scorable's built-in evaluators.

## Setup

Start pgvector:

```bash
mkdir postgres-data
docker run --rm -e POSTGRES_PASSWORD=postgres \
    -p 5432:5432 \
    -v $(pwd)/postgres-data:/var/lib/postgresql/data \
    pgvector/pgvector:pg17
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
SCORABLE_API_KEY=sk-...
```

Install dependencies:

```bash
uv sync
```

Build the search database (fetches and indexes the Scorable docs):

```bash
uv run --env-file .env python -m src.rag build
```

## Run the agent

```bash
uv run --env-file .env python -m src.rag search "How do I run a RAG evaluation?"
```

## Run evals

Vibe-coded LLM-as-judge:

```bash
uv run --env-file .env pytest src/test_vibe_evals.py -v -s
```

Scorable evaluators (requires `SCORABLE_API_KEY`):

```bash
uv run --env-file .env pytest src/scorable_evals.py -v -s
```
