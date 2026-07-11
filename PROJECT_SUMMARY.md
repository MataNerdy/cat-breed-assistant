# Project Summary: Cat Breed Assistant

## Problem

Many beginner LLM projects start as one large script where the UI, prompt logic, API calls and data are tightly mixed. This makes the project hard to test, hard to extend and hard to present as a service-oriented portfolio project.

## Solution

Cat Breed Assistant is a small educational application that answers user questions about cat breeds. It separates the user interface, backend API, local breed data retrieval and answer providers while keeping the architecture understandable.

The app supports a local mock mode for demos without API keys and optional LLM modes through OpenAI, Gemini and Mistral.

## Architecture

```text
Streamlit frontend → FastAPI backend → breed retriever → Mock/OpenAI/Gemini/Mistral provider
```

The frontend sends questions to the backend over HTTP. The backend retrieves relevant breed facts from a local JSON file and routes the request to the selected answer provider.

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
- Sentence Transformers and ChromaDB kept for a future local retrieval layer
- Docker Compose

## Current Features

- Streamlit user interface for asking cat breed questions.
- FastAPI backend with `/health` and `/ask` endpoints.
- Local JSON breed knowledge base.
- Simple RAG-lite retrieval by breed names and aliases.
- Safe placeholder for future vector retrieval through the existing `use_rag` API flag.
- Mock mode that works without API keys.
- OpenAI, Gemini and Mistral modes that use retrieved breed context.
- Safe `.env` handling for API keys.
- Docker Compose setup with separate frontend and backend services.
- Graceful fallback for unknown breeds.

## What I Learned

- How to split an AI-style app into frontend and backend layers.
- How to pass structured context from a retrieval layer into mock and LLM providers.
- How to keep an app usable while a richer retrieval layer is still being designed.
- How to keep API keys out of source code.
- How to package a small two-service Python app with Docker Compose.
- How to keep an MVP understandable without adding unnecessary infrastructure.

## Future Improvements

- Add screenshots and a short demo recording.
- Add tests for breed retrieval and backend response contracts.
- Add tests for the future retrieval contract when a new data source is connected.
- Expand the breed knowledge base.
- Improve multi-breed comparison responses.
- Add deployment notes for a simple cloud target.
