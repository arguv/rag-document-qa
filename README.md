# RAG Document Q&A

A Python/FastAPI foundation for Retrieval-Augmented Generation (RAG). The
project loads PDF, plain-text, and Markdown documents, chunks and embeds their
content, stores vectors in PostgreSQL with pgvector, retrieves relevant
passages, and can generate source-aware answers through the OpenAI Responses
API.

The HTTP API exposes document ingestion, management, retrieval and cited
question answering. Runnable scripts also demonstrate the underlying services.

## Current status

The implementation includes:

- FastAPI routes for health, readiness, document ingestion/management, search,
  cited question answering, Swagger, OpenAPI JSON, and Prometheus-style metrics;
- PDF, TXT, and Markdown loading with PDF page metadata;
- immutable document, chunk, embedding, retrieval, and answer models;
- fixed-size, recursive, and Markdown-aware chunking;
- Sentence Transformers embeddings using `all-MiniLM-L6-v2`;
- SQLAlchemy/pgvector persistence, Alembic migrations, cosine search, and an
  HNSW index;
- semantic retrieval with document and cosine-similarity filtering;
- optional exact BM25/vector reciprocal-rank fusion, deterministic multi-query
  decomposition, cross-encoder reranking, and neighboring context;
- source-labeled context capped at 12,000 characters;
- OpenAI-backed structured answer generation with citation validation, plus an
  explicit mock backend for local wiring tests;
- offline retrieval/answer-overlap evaluation; and
- API-key authentication, request limits, request IDs, rate limiting,
  concurrency bounds, Docker Compose, and CI configuration.

## How it works

```text
Document -> Parse -> Chunk -> Embed -> PostgreSQL/pgvector
Question -> Embed -> Retrieve -> Filter -> Context -> OpenAI -> Answer + citations
```

The answer service does not call the LLM when retrieval returns no evidence.
It also rejects answers that are unanswerable or cite nonexistent retrieval
sources.

## Technology stack

| Area | Technology |
| --- | --- |
| Language | Python 3.10+ |
| API server | FastAPI and Uvicorn |
| Document parsing | `pypdf` |
| Embeddings | `sentence-transformers` and `torch` |
| Answer generation | `openai` and `langgraph` |
| Database access | SQLAlchemy and Psycopg |
| Vector storage | PostgreSQL and `pgvector` |
| Migrations | Alembic |
| Configuration | `python-dotenv` |
| Testing | Pytest and ReportLab |
| Local database | Docker Compose |

Pinned runtime dependencies are in [requirements.txt](requirements.txt).
Development and test dependencies are added by
[requirements-dev.txt](requirements-dev.txt).

## Getting started

Prerequisites: Python 3.10+, Git, Docker Compose for database demos, and an
OpenAI API key for answer generation.

```bash
git clone https://github.com/arguv/rag-document-qa.git
cd rag-document-qa
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The first embedding run may download
`sentence-transformers/all-MiniLM-L6-v2`. Do not commit `.env` or a real API
key. Set `OPENAI_MODEL` to a model available in your OpenAI account.

### Start PostgreSQL

```bash
docker compose up -d db
python -m alembic upgrade head
```

PostgreSQL is exposed on `127.0.0.1:5433` and uses the named
`rag_postgres_data` volume. The default database is `rag_documents`.

## Run the API

```bash
python -m uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`.

```bash
curl http://127.0.0.1:8000/health
```

Swagger UI: `http://127.0.0.1:8000/swagger`

OpenAPI JSON: `http://127.0.0.1:8000/json.json`

## Run the demos

Run these from the project root in order when database-backed behavior is
needed:

```bash
python -m scripts.step_04_embeddings
python -m scripts.step_05_vector_storage
python -m scripts.step_06_retrieval
python -m scripts.step_07_answers
```

The final demo requires `OPENAI_API_KEY`, expects the sample document created
by step 5, and prints a validated answer with source metadata.

Supported document extensions are `.pdf`, `.txt`, and `.md`. Scanned PDFs
without an extractable text layer and encrypted PDFs are not supported yet.

## Run tests

```bash
python -m pytest -v
```

The suite covers the health endpoint, loading, chunking, metadata, embeddings,
vector persistence, retrieval, and answer-service validation. Database tests
require the configured PostgreSQL test database.

## Project structure

```text
.
├── app/
│   ├── main.py                              # FastAPI application
│   ├── application/
│   │   ├── answering.py                     # Retrieval + answer orchestration
│   │   ├── chunking.py                       # Chunking strategies
│   │   ├── embeddings.py                     # Embedding orchestration
│   │   └── retrieval.py                      # Semantic retrieval service
│   ├── api/                                  # REST routes, dependencies and operational boundary
│   ├── core/config.py                        # Environment-backed settings
│   ├── domain/                               # Domain models and protocols
│   └── infrastructure/                       # Database, embeddings, and OpenAI adapters
├── examples/                                 # Demo input documents
├── scripts/                                  # Steps 4–7 demos and offline evaluation
├── migrations/                               # Alembic migrations
├── tests/                                    # Unit and integration tests
├── compose.yaml                              # Local PostgreSQL service
├── .env.example                              # Environment template
├── requirements.txt                          # Runtime dependencies
├── requirements-dev.txt                      # Runtime + test dependencies
└── EXPLANATION.md                            # Architecture and implementation notes
```

## Steps 8–15: API, retrieval, evaluation and deployment

All of these capabilities have runnable implementations. Local unit/API tests
use fake embeddings and a mock generator. A mock answer demonstrates wiring
only; it does not establish real LLM answer quality.

| Step | Implementation |
|---|---|
| 8 | PDF/TXT/Markdown upload, document list/detail/delete, search, cited Q&A, Swagger |
| 9 | Exact BM25 over persisted model/document-filtered chunks, fused with vector rankings using reciprocal rank fusion |
| 10 | Optional Sentence Transformers cross-encoder reranking and deterministic compound-query decomposition |
| 11 | Multi-query retrieval, adjacent-chunk context expansion, deduplication and a 12,000-character context budget |
| 12 | Offline precision/recall@K, MRR, nDCG, exact match and token F1 evaluator |
| 13 | LangGraph retrieve → evidence check → optional broader retrieval → validated answer; at most two retrieval rounds |
| 14 | API-key authentication, bounded requests/uploads/context/concurrency, rate limits, request IDs, metrics, DB timeouts and HNSW index |
| 15 | Non-root Docker image, migration-aware Compose stack, PostgreSQL integration CI and container build |

### Run locally

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
docker compose up -d db
python -m alembic upgrade head
python -m uvicorn app.main:app --reload
```

Keep `LLM_BACKEND=mock` for demonstrations. Only `LLM_BACKEND=openai` enables the existing paid OpenAI generator. Embeddings still require the Sentence Transformers model, downloaded on first use; pre-cache it for offline deployments. Optional reranking also downloads its configured model.

Swagger is at `/swagger` and the schema at `/json.json`. Set `API_KEY` to require `X-API-Key` on data, readiness and metrics routes. Health, Swagger and the schema are public; use Swagger’s **Authorize** button to supply the key for API calls. Production refuses requests if no key is set. This is a shared-corpus, single-key service; document UUIDs are filters, not access permissions.

```bash
curl -H "X-API-Key: $API_KEY" -F 'file=@examples/intro.md' http://127.0.0.1:8000/documents
curl -H "X-API-Key: $API_KEY" http://127.0.0.1:8000/documents
curl -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
  -d '{"question":"How does RAG work?","search_mode":"hybrid","multi_query":true,"neighbors":true,"agentic":true}' \
  http://127.0.0.1:8000/questions
```

`POST /search` accepts the same request and returns source matches and assembled context. `document_id` scopes retrieval; unknown IDs return 404. `DELETE /documents/{id}` removes the document and its chunks. `GET /ready` verifies DB connectivity and the document table; it does not load models or check the LLM. `GET /metrics` exposes process-level request counts and cumulative duration in Prometheus text format.

To enable reranking, set `RERANKER_MODEL` to a compatible cross-encoder model and send `"rerank":true`. An unset model produces 422. Similarity thresholds always refer to vector cosine similarity, not BM25 or fusion scores. The agent may relax the threshold on its single retry, but never removes the document filter. Neighbor chunks are surrounding context and are not threshold-filtered. Citation validation verifies source numbers, not whether every answer claim follows from the evidence.

### Evaluation and tests

```bash
PYTHONPATH=. pytest -q -m 'not integration'
python -m scripts.evaluate examples/evaluation.jsonl --k 5
```

The evaluator reads JSONL predictions with `retrieved` and `relevant` chunk IDs, plus optional `answer` and `expected_answer`. The bundled row is illustrative, not a measured benchmark. Lexical metrics do not measure factual faithfulness. Use a representative labeled corpus and real provider runs before making quality claims.

Integration tests require a migrated database named exactly `rag_documents_test`, selected by `TEST_DATABASE_URL`. CI creates that database, runs migrations and the full suite. No test requires paid OpenAI calls.

### Docker and operations

Set a strong `API_KEY` in `.env`, then run:

```bash
docker compose -f compose.yaml -f compose.api.yaml up --build -d
```

Compose waits for PostgreSQL health, applies migrations in a one-shot service, then starts the API on loopback port 8000. The base Compose credentials are for local development; configure separate secrets and TLS termination for deployment. No cloud deployment is performed by these files. CI builds the image; publishing and deployment require your chosen registry/host.

Current limits: BM25 loads at most `BM25_MAX_CHUNKS` (default 20,000) for exact scoring and fails explicitly above that bound. Use a dedicated search backend at larger scale. Rate limits, concurrency controls and metrics are per process; use an upstream gateway/shared limiter for multiple workers. Upload parsing is synchronous in a worker thread; extracted-text caps do not make PDF parsing a hostile-file sandbox. HNSW is approximate and selective filters may reduce recall; benchmark before tuning. There is no tenant isolation, OCR, distributed job queue, chat memory, external-tool agent, automated factuality judge, or production load certification.

Implementation references: [FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/) and [LangGraph StateGraph API](https://reference.langchain.com/python/langgraph/graph/state/).

### Verification notes

The repository contains 77 tests, including PostgreSQL persistence and an HTTP
ingestion/search/answer/delete round trip with deterministic fake providers.
Run the unit/API tests with `PYTHONPATH=. pytest -q -m 'not integration'`.
Integration tests require the migrated `rag_documents_test` database. No test
requires paid OpenAI calls; real embedding/reranker downloads and real LLM
answer quality are not covered by the suite.
