# Cat Breed Assistant

An educational portfolio project that turns simple LLM-style logic into a small working service with a Streamlit UI, FastAPI backend, local breed knowledge base, mock mode, optional LLM providers and Docker Compose packaging.

Небольшой учебный LLM-сервис, рассказывающий пользователю про породы кошек. Проект как демонстрация разделения пользовательского интерфейса, backend-логики, локального data layer и внешних API-провайдеров, без превращения MVP в сложную платформу.

## Demo

### Streamlit frontend

![Streamlit frontend](assets/streamlit_demo.png)

### FastAPI backend docs

![FastAPI backend docs](assets/fastapi_docs.png)

## What This Project Demonstrates

- A frontend/backend split for a small AI-style application.
- HTTP communication between Streamlit and FastAPI.
- A simple local data layer before mock/LLM answer generation.
- Safe API-key handling with `.env` and `.env.example`.
- Docker Compose packaging with separate frontend and backend services.
- Graceful handling of missing API keys, unknown breeds and backend errors.

## Architecture

```text
Streamlit frontend → FastAPI backend → breed retriever → LLM/mock provider
```

The Streamlit app never calls mock, OpenAI or Gemini logic directly. It sends user questions to the FastAPI backend. The backend retrieves breed facts from local JSON and then routes the request to mock, OpenAI or Gemini mode.

## Features

- Ask questions about cat breeds in a Streamlit UI.
- Use `Mock mode` without any API keys.
- Use `OpenAI mode` with `OPENAI_API_KEY`.
- Use `Gemini mode` with `GEMINI_API_KEY`.
- Retrieve breed facts from a local JSON knowledge base.
- Detect known breeds by English and Russian aliases.
- Return a neutral fallback when a breed is not found.
- Run locally with two processes or with one Docker Compose command.

## Tech Stack

- Python 3.12
- Streamlit
- FastAPI
- Uvicorn
- Requests
- Pydantic
- OpenAI Python SDK
- Google Gen AI SDK (`google-genai`)
- python-dotenv
- Docker / Docker Compose

## Project Structure

```text
cat-breed-assistant/
├── assets/
│   └── .gitkeep
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── services.py
├── data/
│   └── breed_profiles.json
├── src/
│   ├── __init__.py
│   ├── breed_retriever.py
│   ├── cat_knowledge.py
│   ├── gemini_client.py
│   └── llm_client.py
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── PROJECT_SUMMARY.md
├── README.md
├── app.py
├── docker-compose.yml
└── requirements.txt
```

## Run Locally

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend in terminal 1:

```bash
uvicorn backend.main:app --reload --port 8000
```

Start the frontend in terminal 2:

```bash
streamlit run app.py
```

Open the Streamlit URL printed in the terminal, usually:

```text
http://localhost:8501
```

Backend healthcheck:

```bash
curl http://localhost:8000/health
```

Mock ask endpoint:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Расскажи про мейн-куна","mode":"mock"}'
```

## Run With Docker

Build and start both services:

```bash
docker compose up --build
```

The Streamlit frontend will be available at:

```text
http://localhost:8501
```

FastAPI docs will be available at:

```text
http://localhost:8000/docs
```

Inside the Docker Compose network, the frontend calls the backend by service name:

```text
http://backend:8000
```

That value is passed to the frontend as:

```bash
BACKEND_URL=http://backend:8000
```

Stop the services:

```bash
docker compose down
```

## Environment Variables

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Add real keys only if you want to use LLM modes:

```bash
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
BACKEND_URL=http://localhost:8000
```

`Mock mode` works without `.env`.

Do not commit `.env`. The repository includes only `.env.example`, which contains placeholders. `.env` is ignored by Git and excluded from the Docker build context.

## Mock Mode Vs LLM Modes

`Mock mode` builds a deterministic answer from local breed facts. It is useful for demos, tests and development without paid API access.

`OpenAI mode` sends the question and retrieved breed context to OpenAI through `src/llm_client.py`.

`Gemini mode` sends the question and retrieved breed context to Gemini through `src/gemini_client.py`.

LLM prompts instruct the model to answer in Russian, use only the provided context and avoid invented medical advice. If the local data is not enough, the model should say so.

## Data Layer / RAG-lite

The project uses a small local knowledge base:

```text
data/breed_profiles.json
```

Each profile contains aliases, origin, appearance, temperament, care notes, cautious health notes, fun facts and differences from other breeds.

This is not vector RAG. There are no embeddings, Chroma, FAISS, LangChain or LlamaIndex. The retrieval layer is intentionally simple:

1. FastAPI receives the user question.
2. `src/breed_retriever.py` searches breed names and aliases in local JSON.
3. The backend builds `breed_context`.
4. Mock/OpenAI/Gemini modes use the same context.

If no breed is detected, the backend returns `Unknown breed` and shows which breeds are currently available.

## Example Questions

```text
Чем британские котики отличаются от обычных?
Расскажи про мейн-куна
Как ухаживать за сфинксом?
Сравни сиамскую кошку и перса
Какая порода подойдёт спокойному человеку?
```

## Sample Output

```text
Maine Coon — не просто красивое название, а целый набор породных особенностей.

Внешний вид: крупное тело, длинный пушистый хвост и мощный костяк.
Характер: обычно дружелюбные, общительные и любопытные.
Уход: регулярное расчёсывание, пространство для движения и устойчивые когтеточки.
```

## What I Learned

- How to split a small app into frontend and backend.
- How to call a FastAPI backend from Streamlit.
- How to add a local data layer before LLM providers.
- How to keep mock and LLM provider logic behind a service layer.
- How to load API keys from environment variables.
- How to handle missing credentials and unknown retrieval results gracefully.
- How to package a two-service app with Docker Compose.

## Next Steps

- Add a real UI screenshot to the `Demo` section.
- Add focused unit tests for breed retrieval and provider routing.
- Add more breed profiles and richer aliases.
- Add a lightweight comparison response for multi-breed questions.
- Improve frontend styling without changing the core architecture.

## Notes

This is intentionally a small learning MVP. It does not include full RAG, embeddings, a vector database, authentication or a production deployment pipeline.
