# Cat Breed Assistant

An educational portfolio project that turns LLM-style answer logic into a small working service with a Streamlit UI, FastAPI backend, local breed knowledge base, mock mode, optional LLM providers and Docker Compose packaging.

Небольшой учебный сервис про породы кошек: пользователь задаёт текстовый вопрос, приложение определяет породу по локальной базе знаний и возвращает дружелюбный ответ через mock-логику или внешний LLM API.

## Demo

### Streamlit frontend

![Streamlit frontend](assets/streamlit_demo.png)

### FastAPI backend docs

![FastAPI backend docs](assets/fastapi_docs.png)

## What This Project Demonstrates

- A frontend/backend split for a small AI-style application.
- HTTP communication between Streamlit and FastAPI.
- A simple local data layer before answer generation.
- Multiple provider modes behind one backend service layer.
- Safe API-key handling with `.env` and `.env.example`.
- Docker Compose packaging with separate frontend and backend services.
- Graceful handling of missing API keys, unknown breeds and backend errors.

## Architecture

```text
Streamlit frontend → FastAPI backend → breed retriever → LLM/mock provider
```

The Streamlit app does not call mock, OpenAI, Gemini or Mistral logic directly. It sends user questions to the FastAPI backend. The backend builds breed context from local JSON and routes the request to the selected answer provider.

## Features

- Ask text questions about cat breeds in a Streamlit UI.
- Use `Mock mode` without any API keys.
- Use `OpenAI mode` with `OPENAI_API_KEY`.
- Use `Gemini mode` with `GEMINI_API_KEY`.
- Use `Mistral mode` with `MISTRAL_API_KEY`.
- Retrieve breed facts from a local JSON knowledge base.
- Detect known breeds by English and Russian aliases.
- Return a neutral fallback when a breed is not found.
- Keep `use_rag=true` safe while the next data source is not connected yet.
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
- Mistral AI SDK (`mistralai`)
- python-dotenv
- Sentence Transformers and ChromaDB kept for the next local retrieval layer
- Docker / Docker Compose

## Project Structure

```text
cat-breed-assistant/
├── assets/
│   ├── fastapi_docs.png
│   └── streamlit_demo.png
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
│   ├── llm_client.py
│   └── mistral_client.py
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

Mock ask endpoint with the currently disabled vector retrieval flag:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Расскажи про сибирскую кошку","mode":"mock","use_rag":true}'
```

`use_rag=true` is intentionally safe: until the next retrieval data source is added, the backend returns an empty `retrieved_context` instead of crashing.

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
MISTRAL_API_KEY=your_mistral_api_key_here
BACKEND_URL=http://localhost:8000
```

`Mock mode` works without `.env`.

Do not commit `.env`. The repository includes only `.env.example`, which contains placeholders. `.env` is ignored by Git and excluded from the Docker build context.

## Mock Mode Vs LLM Modes

`Mock mode` builds a deterministic answer from local breed facts. It is useful for demos, tests and development without paid API access.

`OpenAI mode` sends the question and breed context to OpenAI through `src/llm_client.py`.

`Gemini mode` sends the question and breed context to Gemini through `src/gemini_client.py`.

`Mistral mode` sends the question and breed context to Mistral through `src/mistral_client.py`.

LLM prompts instruct the model to answer in Russian, use only the provided context and avoid invented medical advice. If the local data is not enough, the model should say so.

## Data Layer / RAG-lite

The project uses a small local knowledge base:

```text
data/breed_profiles.json
```

Each profile contains aliases, origin, appearance, temperament, care notes, cautious health notes, fun facts and differences from other breeds.

This layer is intentionally simple:

1. FastAPI receives the user question.
2. `src/breed_retriever.py` searches breed names and aliases in local JSON.
3. The backend builds `breed_context`.
4. Mock/OpenAI/Gemini/Mistral modes use the same context.

If no breed is detected, the backend returns `Unknown breed` and shows which breeds are currently available.

## Vector Retrieval Status

The previous experimental vector retrieval branch has been removed from the active project. The backend keeps the `use_rag` request flag as a safe placeholder so the API contract does not break while the next data source is being designed.

Current behavior:

- `use_rag=false`: normal local breed profile flow.
- `use_rag=true`: normal local breed profile flow plus `retrieved_context=[]`.

Future retrieval data can be added later without changing the Streamlit-to-FastAPI contract.

## Example Questions

```text
Чем британские котики отличаются от обычных?
Расскажи про мейн-куна
Как ухаживать за сфинксом?
Сравни сиамскую кошку и перса
Какая порода подойдёт спокойному человеку?
Чем бенгальская кошка отличается от оцелота?
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
- How to pass structured breed context into mock and LLM providers.
- How to keep mock and LLM provider logic behind a service layer.
- How to load API keys from environment variables.
- How to handle missing credentials and unknown retrieval results gracefully.
- How to package a two-service app with Docker Compose.

## Next Steps

- Add focused unit tests for breed retrieval and backend response contracts.
- Add a cleaner local retrieval source for richer breed facts.
- Add more breed profiles and richer aliases.
- Add a lightweight comparison response for multi-breed questions.
- Improve frontend styling without changing the core architecture.

## Notes

This is intentionally a small learning MVP. It includes a local RAG-lite layer, but does not include authentication, a database, background workers or a production deployment pipeline.
