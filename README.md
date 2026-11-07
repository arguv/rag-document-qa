# RAG Document Q&A

A Python/FastAPI foundation for Retrieval-Augmented Generation (RAG). The
project loads PDF, plain-text, and Markdown documents, chunks and embeds their
content, stores vectors in PostgreSQL with pgvector, retrieves relevant
passages, and can generate source-aware answers through the OpenAI Responses
API.

The HTTP API currently exposes a health check. Ingestion, retrieval, and
answering are available as reusable services and runnable demo scripts.

## Current status

Implemented:

- FastAPI application with `/health` and OpenAPI documentation;
- PDF, TXT, and Markdown loading with PDF page metadata;
- immutable document, chunk, embedding, retrieval, and answer models;
- fixed-size, recursive, and Markdown-aware chunking;
- Sentence Transformers embeddings using `all-MiniLM-L6-v2`;
- SQLAlchemy/pgvector persistence, Alembic migrations, and cosine search;
- semantic retrieval with top-k, document, and similarity filtering;
- source-labeled retrieval context; and
- OpenAI-backed structured answer generation with citation validation.

Not implemented yet:

- HTTP document upload;
- REST retrieval and question-answering endpoints;
- hybrid retrieval, reranking, and query rewriting; and
- authentication, observability, evaluation, and production deployment
  configuration.

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
| Answer generation | `openai` |
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
│   ├── core/config.py                        # Environment-backed settings
│   ├── domain/                               # Domain models and protocols
│   └── infrastructure/                       # Database, embeddings, and OpenAI adapters
├── examples/                                 # Demo input documents
├── scripts/                                  # Steps 4–7 runnable demos
├── migrations/                               # Alembic migrations
├── tests/                                    # Unit and integration tests
├── compose.yaml                              # Local PostgreSQL service
├── .env.example                              # Environment template
├── requirements.txt                          # Runtime dependencies
├── requirements-dev.txt                      # Runtime + test dependencies
└── EXPLANATION.md                            # Architecture and implementation notes
```

## Roadmap

1. Add HTTP document upload and production file validation.
2. Expose retrieval and answering through the API.
3. Add hybrid retrieval, reranking, and query rewriting.
4. Add retrieval and answer-quality evaluation.
5. Add authentication, observability, CI/CD, and deployment support.

See [EXPLANATION.md](EXPLANATION.md) for architecture and implementation
details.
