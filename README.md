# Cat Breed Assistant

Небольшой LLM-сервис для ответов на вопросы о породах кошек.
Проект демонстрирует разделение интерфейса и backend-логики: Streamlit отвечает за пользовательский интерфейс, FastAPI принимает HTTP-запросы, валидирует данные и вызывает mock/LLM provider.

## Architecture

```text
Streamlit frontend -> FastAPI backend -> data layer -> Mock/OpenAI/Gemini provider
```

Такой формат показывает базовый сервисный подход: UI не вызывает LLM напрямую, а общается с backend через HTTP.

## What It Does

- показывает простой Streamlit-интерфейс;
- принимает вопрос пользователя про кошек;
- отправляет вопрос в FastAPI backend;
- распознаёт вопросы про British Shorthair / британских короткошёрстных кошек;
- использует локальный JSON с фактами о породах;
- поддерживает режимы `mock`, `openai`, `gemini`;
- аккуратно сообщает, если backend не запущен или API-ключ не настроен.

## Tech Stack

- Python 3.12
- Streamlit
- FastAPI
- Uvicorn
- Requests
- OpenAI Python SDK
- Google Gen AI SDK (`google-genai`)
- python-dotenv
- Docker / Docker Compose

## Project Structure

```text
cat-breed-assistant/
├── Dockerfile
├── app.py
├── docker-compose.yml
├── data/
│   └── breed_profiles.json
├── requirements.txt
├── README.md
├── .dockerignore
├── .env.example
├── .gitignore
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── services.py
└── src/
    ├── __init__.py
    ├── breed_retriever.py
    ├── cat_knowledge.py
    ├── llm_client.py
    └── gemini_client.py
```

## Setup

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Add your own keys only if you want to use API modes:

```bash
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

Optional frontend setting:

```bash
BACKEND_URL=http://localhost:8000
```

`Mock mode` works without API keys.

## Run The App

You need two terminal windows.

Terminal 1: start the backend:

```bash
uvicorn backend.main:app --reload --port 8000
```

Terminal 2: start the Streamlit frontend:

```bash
streamlit run app.py
```

Open the Streamlit URL printed in the terminal and ask a question.

## Data Layer / RAG-lite

The app uses a small local knowledge base:

```text
data/breed_profiles.json
```

This JSON file contains breed profiles with aliases, origin, appearance, temperament, care notes, cautious health notes, fun facts and differences from other breeds.

This is not a full vector RAG system. There is no Chroma, FAISS, LangChain or LlamaIndex. It is a first simple retrieval layer:

1. The backend receives the question.
2. `src/breed_retriever.py` searches breed names and aliases in the local JSON.
3. The backend builds a `breed_context`.
4. Mock/OpenAI/Gemini modes use that same context to answer.

If no breed is detected, the backend uses British Shorthair as a default fallback and marks that in the context.

## Docker Quick Start

Build and start both services:

```bash
docker compose up --build
```

The frontend will be available at:

```text
http://localhost:8501
```

FastAPI docs will be available at:

```text
http://localhost:8000/docs
```

Inside the Docker Compose network, the Streamlit frontend does not call `localhost`. It calls the backend service by its compose service name:

```text
http://backend:8000
```

That value is passed to the frontend as:

```bash
BACKEND_URL=http://backend:8000
```

`Mock mode` works without a `.env` file. API modes work only when the corresponding keys are present in `.env`. Docker Compose reads `.env` through `env_file`, but `.env` is ignored by Git and excluded from the Docker build context.

Stop the services:

```bash
docker compose down
```

## Backend Checks

Healthcheck:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

Ask endpoint:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Расскажи про британских котиков","mode":"mock"}'
```

Example response:

```json
{
  "answer": "...",
  "breed": "Британская короткошёрстная кошка",
  "mode": "mock"
}
```

## Modes

`mock` uses a local prepared response from `src/cat_knowledge.py`. It is the safest mode for demos and local development because it does not require API keys.

`openai` uses `OPENAI_API_KEY` and calls OpenAI through `src/llm_client.py`.

`gemini` uses `GEMINI_API_KEY` and calls Gemini through `src/gemini_client.py`.

Gemini API keys can be created in Google AI Studio: https://aistudio.google.com/app/apikey

Free API limits, quotas and model availability can change over time. If an API mode stops working, check the provider dashboard, billing/quota settings and current limits.

## Do Not Commit API Keys

Never commit `.env` to GitHub. API keys are secrets: if they become public, someone else can use your quota or account.

This project keeps `.env` in `.gitignore` and provides `.env.example` only as a safe template.

## Example Question

```text
Расскажи про британских короткошёрстных кошек
```

The answer should explain who British Shorthair cats are, what they look like, their character, how they differ from other breeds and what basic care is important.

More examples:

```text
Чем британские котики отличаются от обычных?
Расскажи про мейн-куна
Сравни сиамскую кошку и перса
Как ухаживать за сфинксом?
```

## What I Learned

- how to split a small app into frontend and backend;
- how to call a FastAPI backend from Streamlit;
- how to add a simple local data layer before LLM providers;
- how to keep mock logic and LLM provider logic behind a service layer;
- how to load API keys from environment variables;
- how to handle missing credentials and backend errors gracefully.

## Notes

This is intentionally a small learning MVP. It does not include RAG, a database or authentication.
