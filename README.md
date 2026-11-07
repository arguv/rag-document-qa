# RAG Document Q&A

A Python and FastAPI foundation for a Retrieval-Augmented Generation (RAG)
application. The repository supports document loading, chunking, embedding
generation, PostgreSQL/pgvector persistence, semantic retrieval, and
source-labeled context assembly. HTTP upload, retrieval APIs, and LLM answer
generation are still planned.

## Current status

Implemented:

- FastAPI application with a health endpoint and OpenAPI documentation;
- PDF, plain-text, and Markdown document loading;
- PDF page metadata preservation;
- immutable document and chunk domain models;
- fixed-size, recursive, and Markdown-aware chunking strategies;
- pluggable embedding providers;
- Sentence Transformers embeddings using `all-MiniLM-L6-v2`; and
- cosine-similarity ranking in the embedding demo;
- SQLAlchemy models and a repository for storing embedded chunks in pgvector;
- Alembic migrations for the vector-storage schema; and
- a PostgreSQL-backed vector search demo;
- a semantic retrieval service with top-k search, document filtering, and
  optional similarity thresholds; and
- structured retrieval results with source-labeled context ready for an LLM.

Not implemented yet:

- HTTP document upload;
- REST retrieval and question-answering endpoints;
- hybrid retrieval, reranking, and query rewriting;
- LLM-powered answers and source citations; and
- authentication, observability, evaluation, and production deployment
  configuration.

## How it works

The intended document pipeline is:

```text
Document -> Parse -> Chunk -> Embed -> Vector store
```

The retrieval pipeline implemented by the project is:

```text
Question -> Embed -> Retrieve -> Filter -> Context
```

The intended question-answering pipeline adds generation:

```text
Question -> Embed -> Retrieve -> Filter -> Context -> LLM -> Cited answer
```

The current code implements parsing, chunking, embedding, persistence,
database-level vector search, and application-level semantic retrieval locally.
The remaining HTTP and generation stages will build on these components.

## Technology stack

| Area | Technology |
| --- | --- |
| Language | Python 3.10+ |
| API | FastAPI and Uvicorn |
| Document parsing | `pypdf` |
| Embeddings | Sentence Transformers and PyTorch |
| Testing | Pytest |
| Storage | PostgreSQL and pgvector |
| Database access | SQLAlchemy and Alembic |
| Local services | Docker Compose |

## Getting started

### Prerequisites

- Python 3.10 or newer
- Git

### Installation

```bash
git clone https://github.com/arguv/rag-document-qa.git
cd rag-document-qa

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The first real embedding run may download the configured Sentence Transformers
model (`sentence-transformers/all-MiniLM-L6-v2`).

### Start PostgreSQL with pgvector

Copy the example environment file and start the database service:

```bash
cp .env.example .env
docker compose up -d db
python -m alembic upgrade head
```

The Compose service exposes PostgreSQL on `127.0.0.1:5433` and stores its data
in the named `rag_postgres_data` volume. The default development database is
`rag_documents`. Keep `TEST_DATABASE_URL` pointed at the separate
`rag_documents_test` database when running integration tests.

## Run the API

Start the development server from the project root:

```bash
python -m uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`.

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "rag-document-qa"
}
```

Interactive Swagger UI is available at
`http://127.0.0.1:8000/swagger`. The OpenAPI schema is available at
`http://127.0.0.1:8000/json.json`.

## Run the embedding demo

The demo loads `examples/embedding_demo.md`, splits it into Markdown-aware
chunks, embeds the chunks and a sample question, then prints chunks ordered by
cosine similarity:

```bash
python -m scripts.step_04_embeddings
```

The demo uses a 300-character chunk size with 40 characters of overlap. The
embedding provider can also be used independently:

```python
from app.infrastructure.document_loader import load_document
from app.infrastructure.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddingProvider,
)

document = load_document("examples/embedding_demo.md")
provider = SentenceTransformerEmbeddingProvider()
vectors = provider.encode([document.text])
```

## Retrieve relevant context

After storing the example document with the vector-storage demo, run the
semantic retrieval demo:

```bash
python -m scripts.step_06_retrieval
```

The demo embeds a question, searches PostgreSQL for the nearest chunks, limits
results to the selected document, applies a minimum cosine-similarity score,
and prints source-labeled context. The `RetrievalService` accepts `top_k`
values from 1 to 50 and can also be used with fake providers and repositories
in unit tests.

## Persist and search vectors

After PostgreSQL is running and the migration has been applied, run the
database-backed demo:

```bash
python -m scripts.step_05_vector_storage
```

It loads and chunks the example document, generates 384-dimensional embeddings,
saves the document and chunks in PostgreSQL, and searches them using pgvector
cosine distance. The repository stores the embedding model name with each
document so searches can be restricted to compatible vectors.

Supported document extensions are `.pdf`, `.txt`, and `.md`. Scanned PDFs
without an extractable text layer and encrypted PDFs are not supported yet.

## Run tests

```bash
python -m pytest -v
```

The test suite covers the health endpoint, document loading and error cases,
chunking behavior, metadata preservation, embedding validation, cosine
similarity, vector persistence, and semantic retrieval behavior. Retrieval
tests cover query embedding, top-k and document filtering, similarity
thresholds, context formatting, empty results, and invalid inputs.

## Project structure

```text
.
├── app/
│   ├── main.py                              # FastAPI application
│   ├── application/
│   │   ├── chunking.py                      # Chunking strategies
│   │   ├── embeddings.py                    # Embedding orchestration
│   │   └── retrieval.py                     # Semantic retrieval service
│   ├── domain/
│   │   ├── chunks.py                        # Chunk model
│   │   ├── documents.py                     # Parsed document models
│   │   ├── embeddings.py                     # Vectors and provider protocol
│   │   ├── retrieval.py                      # Retrieval result model
│   │   └── search.py                         # Search match model
│   └── infrastructure/
│       ├── database/                        # SQLAlchemy/pgvector persistence
│       ├── document_loader.py               # PDF, TXT, and Markdown loader
│       └── sentence_transformer_embeddings.py
├── examples/
│   └── embedding_demo.md                   # Demo input document
├── scripts/
│   ├── step_04_embeddings.py                # Local embedding demo
│   ├── step_05_vector_storage.py            # PostgreSQL/pgvector demo
│   └── step_06_retrieval.py                 # Semantic retrieval demo
├── migrations/                              # Alembic migrations
├── compose.yaml                             # Local PostgreSQL service
├── .env.example                             # Development DB settings
├── tests/                                   # Unit and API tests
├── requirements.txt                         # Runtime dependencies
├── requirements-dev.txt                     # Runtime + test dependencies
└── EXPLANATION.md                           # Step-by-step project notes
```

## Roadmap

1. Add HTTP document upload and production file validation.
2. Expose semantic retrieval through the API.
3. Add hybrid retrieval, reranking, and query rewriting.
4. Connect retrieved context to an LLM.
5. Return and validate source citations with generated answers.
6. Add retrieval and answer-quality evaluation.
7. Add authentication, observability, CI/CD, and production deployment support.

See [EXPLANATION.md](EXPLANATION.md) for architectural background and detailed
implementation notes.
