# Cat Breed Assistant

Учебный portfolio-проект: маленький сервис, который отвечает на вопросы пользователя про породы кошек.

Коротко по-русски: это минимальное приложение с разделением frontend и backend. Streamlit отвечает за пользовательский интерфейс, FastAPI принимает запросы, а backend уже выбирает provider ответа: локальный mock, OpenAI или Gemini.

## Architecture

```text
Streamlit frontend -> FastAPI backend -> Mock/OpenAI/Gemini provider
```

Такой формат показывает базовый сервисный подход: UI не вызывает LLM напрямую, а общается с backend через HTTP.

## What It Does

- показывает простой Streamlit-интерфейс;
- принимает вопрос пользователя про кошек;
- отправляет вопрос в FastAPI backend;
- распознаёт вопросы про British Shorthair / британских короткошёрстных кошек;
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

## Project Structure

```text
cat-breed-assistant/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── services.py
└── src/
    ├── __init__.py
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

## What I Learned

- how to split a small app into frontend and backend;
- how to call a FastAPI backend from Streamlit;
- how to keep mock logic and LLM provider logic behind a service layer;
- how to load API keys from environment variables;
- how to handle missing credentials and backend errors gracefully.

## Notes

This is intentionally a small learning MVP. It does not include Docker, RAG, a database or authentication.
