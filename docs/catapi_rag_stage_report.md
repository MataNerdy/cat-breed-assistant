# CatAPI RAG Stage Report

## Stage goal

The goal of this stage was to replace the previous unsuitable Hugging Face dataset experiment with a controlled breed data source and wire that source into the working application.

The app remains a text assistant: the user asks a text question, the backend retrieves relevant breed context, and the selected provider returns a text answer.

## Data source

The current source is TheCatAPI `/v1/breeds` endpoint.

CatAPI provides compact structured breed profiles with fields such as breed name, temperament, origin, description, life span, weight, grooming level, energy level, health issue score, friendliness scores, hypoallergenic flag, Wikipedia URL and reference image id.

## Processed data

The pipeline produces two committed processed files:

- `data/processed/catapi_breed_documents.jsonl`
- `data/processed/catapi_chunks.jsonl`

Current size:

- 67 breeds
- 67 documents
- 67 chunks

One chunk is one breed profile. This keeps the current retrieval layer easy to inspect and debug.

## Retrieval strategy

The current app does not use pure vector search as the primary retrieval path. It uses a controlled hybrid baseline:

1. Breed alias detection in Russian and English.
2. Structured CatAPI field scoring for questions without an explicit breed.
3. No-match guard for irrelevant or absurd questions.
4. LLM-grounded answer based on the retrieved CatAPI context.

This approach is intentionally simple, but it is easier to control than unstable dense retrieval on short breed profiles.

## Backend integration

FastAPI `/ask` accepts a user question, answer mode and `use_rag` flag. Streamlit currently sends `use_rag=true` by default.

The backend calls `retrieve_catapi_context()`, passes retrieved context to the selected provider and returns retrieval metadata in the JSON response.

Supported providers:

- Mock
- Mistral
- Gemini
- OpenAI

## Frontend visibility

The Streamlit UI now keeps diagnostic retrieval metadata off the main page. The user sees the answer and optional image when an image URL is available.

Retrieval diagnostics are logged by the frontend:

- mode
- detected breed
- retrieval strategy
- retrieved context count
- reference image id
- ranked retrieved context metadata

## Evaluation examples

| Question | Strategy | Expected retrieved context |
| -------- | -------- | -------------------------- |
| `Чем британские котики отличаются от обычных?` | `alias_exact_breed` | British Shorthair |
| `Расскажи про мейн-куна` | `alias_exact_breed` | Maine Coon |
| `Как ухаживать за сфинксом?` | `alias_exact_breed` | Sphynx |
| `Какая кошка большая, спокойная и ласковая?` | `structured_fields` | British Shorthair / Ragamuffin / Ragdoll |
| `Какая кошка разговорчивая и умная?` | `structured_fields` | Balinese / Bengal / Bombay / Burmese-like candidates |
| `long hair grooming` | `structured_fields` | British Longhair / Chantilly-Tiffany / Himalayan |
| `Какая кошка умеет чинить ноутбук?` | `no_match` | No random breed returned |

## Embedding experiment note

Embeddings were tested in notebooks and the embedding model technically worked. However, dense retrieval quality on the short CatAPI breed corpus was unstable for the target Russian-language questions.

Structured CatAPI fields produced more predictable and inspectable results, so the current application uses the hybrid CatAPI retriever first.

This is an engineering decision, not a failed experiment. Vector retrieval can still be added later as an optional layer after a better model bake-off.

## Current conclusion

The project now has a working CatAPI-powered LLM assistant MVP:

```text
User text query
→ FastAPI backend
→ CatAPI hybrid retriever
→ retrieved CatAPI context
→ LLM provider
→ Streamlit answer
```

The retrieval layer is small, deterministic and easy to explain in a portfolio review.

## Next steps

- Finish Mistral and Gemini answer evaluation.
- Add CatAPI image fetching by `reference_image_id`.
- Add Wikipedia enrichment for richer breed descriptions.
- Improve aliases for Russian and English breed names.
- Add optional semantic retrieval after model evaluation.
- Consider ChromaDB only after embedding quality is acceptable.
