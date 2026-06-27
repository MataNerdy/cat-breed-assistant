# Cat Breed Assistant

Учебный portfolio-проект на Streamlit: маленькое веб-приложение, которое принимает вопрос пользователя про породы кошек и возвращает дружелюбный ответ.

Коротко по-русски: это минимальный LLM-сервис с пользовательским интерфейсом. Проект показывает, как вынести логику ответа в отдельные модули, подключить внешние LLM API через переменные окружения и оставить безопасный mock-режим для запуска без ключей.

## What It Does

- показывает простой Streamlit-интерфейс;
- принимает вопрос пользователя про кошек;
- распознаёт вопросы про British Shorthair / британских короткошёрстных кошек;
- отвечает в трёх режимах: `Mock mode`, `OpenAI mode`, `Gemini mode`;
- не падает, если API-ключ не задан, а показывает понятное сообщение;
- хранит секреты только в локальном `.env`, который не должен попадать в GitHub.

## Tech Stack

- Python 3.12
- Streamlit
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
└── src/
    ├── __init__.py
    ├── cat_knowledge.py
    ├── llm_client.py
    └── gemini_client.py
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

Start the app:

```bash
streamlit run app.py
```

## Environment Variables

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Then add your own keys if you want to use API modes:

```bash
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

The `.env.example` file contains only placeholders. Real API keys must stay only in `.env`.

## Mock Mode

`Mock mode` is the default mode. It works without any API keys and uses a local prepared answer from `src/cat_knowledge.py`.

This is useful for:

- testing the Streamlit UI;
- demoing the project without paid API access;
- developing service logic before connecting a real LLM.

## LLM/API Modes

`OpenAI mode` uses `OPENAI_API_KEY` and calls OpenAI through `src/llm_client.py`.

`Gemini mode` uses `GEMINI_API_KEY` and calls Gemini through `src/gemini_client.py`.

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

- how to build a small Streamlit service around LLM-style logic;
- how to keep mock logic and API logic separate;
- how to load API keys from environment variables;
- how to handle missing credentials gracefully;
- how to prepare a small Python project for a clean GitHub portfolio repository.

## Notes

This is intentionally a small learning MVP. It does not include FastAPI, Docker, RAG or a database.
