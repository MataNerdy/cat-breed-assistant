# Project Summary: Cat Breed Assistant

Status: Working CatAPI-powered LLM assistant MVP

## Problem

Many beginner LLM projects start as one large script where the UI, prompt logic, API calls and data are tightly mixed. This makes the project hard to test, hard to extend and hard to present as a service-oriented portfolio project.

The project also needs a controlled knowledge source, because free-form LLM answers without grounded context can easily become generic or inconsistent.

## Current Solution

Cat Breed Assistant is a small educational application that answers user questions about cat breeds. It separates the user interface, backend API, retrieval layer and answer providers while keeping the architecture understandable.

The current app uses CatAPI breed data as its primary retrieval source. It supports a local mock mode for demos without API keys and optional LLM modes through OpenAI, Gemini and Mistral.

## Architecture

```text
Streamlit frontend
→ FastAPI backend
→ CatAPI hybrid retriever
→ retrieved CatAPI context
→ Mock / OpenAI / Gemini / Mistral provider
```

The frontend sends questions to the backend over HTTP. The backend retrieves relevant CatAPI breed context and routes the request to the selected answer provider.

## Data Source

The current controlled data source is TheCatAPI `/v1/breeds`.

Processed files:

- `data/processed/catapi_breed_documents.jsonl`
- `data/processed/catapi_chunks.jsonl`

Current size:

- 67 breeds
- 67 documents
- 67 chunks

One chunk is one breed profile.

## Retrieval Strategy

The current retrieval layer is a hybrid baseline, not pure vector search:

1. Russian and English breed alias detection.
2. Structured CatAPI field scoring.
3. No-match guard for irrelevant questions.
4. Grounded provider answer using retrieved context.

This strategy was selected because the CatAPI corpus is short and structured. Embeddings were tested, but structured field retrieval produced more controlled results for the target questions.

## Backend Integration

FastAPI exposes:

- `GET /health`
- `POST /ask`

The `/ask` endpoint accepts a question, provider mode and retrieval flag. It returns the generated answer plus retrieval metadata such as detected breed, retrieval strategy, retrieved context, warning and CatAPI image/reference fields.

Supported provider modes:

- Mock
- OpenAI
- Gemini
- Mistral

## Frontend Behavior

The Streamlit frontend sends every user request to FastAPI. It currently exposes user-facing modes:

- Mock
- Gemini
- Mistral

CatAPI retrieval is always enabled from the Streamlit UI. Diagnostic retrieval metadata is logged instead of being shown on the main page, keeping the interface cleaner for a normal user.

## Tech Stack

- Python 3.12
- Streamlit
- FastAPI
- Pydantic
- Uvicorn
- Requests
- OpenAI Python SDK
- Google Gen AI SDK
- Mistral AI SDK
- python-dotenv
- Docker Compose
- Sentence Transformers and ChromaDB kept for experiments, not primary backend retrieval

## Current Features

- Streamlit user interface for asking cat breed questions.
- FastAPI backend with `/health` and `/ask` endpoints.
- CatAPI processed data layer with 67 breed profiles.
- Hybrid retrieval by aliases and structured CatAPI fields.
- No-match guard for irrelevant questions.
- Mock mode that works without API keys.
- Gemini and Mistral modes in the Streamlit UI.
- OpenAI provider still available in the backend.
- Safe `.env` handling for API keys.
- Docker Compose setup with separate frontend and backend services.
- Lightweight regression check for CatAPI retrieval behavior.

## Current Limitations

- CatAPI descriptions are short.
- `image_url` may be empty even when `reference_image_id` exists.
- No Wikipedia enrichment yet.
- No production vector database yet.
- Embeddings were tested but not selected as the primary retriever.
- No CV or image classification branch.
- Multi-breed comparison is still limited.

## What I Learned

- How to split an AI-style app into frontend and backend layers.
- How to pass retrieved context from a data layer into mock and LLM providers.
- How to compare retrieval approaches and choose a controlled baseline when dense retrieval is unstable.
- How to keep API keys out of source code.
- How to package a small two-service Python app with Docker Compose.
- How to keep an MVP understandable without adding unnecessary infrastructure.

## Next Steps

- Finish Mistral and Gemini answer evaluation.
- Add CatAPI image fetching by `reference_image_id`.
- Add Wikipedia enrichment for richer breed context.
- Improve aliases for Russian and English breed names.
- Add optional semantic retriever after model bake-off.
- Consider ChromaDB only after embedding quality is acceptable.
