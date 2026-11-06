# RAG Document Q&A

A Python and FastAPI foundation for a Retrieval-Augmented Generation (RAG)
application. The repository currently supports document loading, chunking,
embedding generation, and cosine-similarity ranking. HTTP upload, persistence,
retrieval APIs, and LLM answer generation are still planned.

## Current status

Implemented:

- FastAPI application with a health endpoint and OpenAPI documentation;
- PDF, plain-text, and Markdown document loading;
- PDF page metadata preservation;
- immutable document and chunk domain models;
- fixed-size, recursive, and Markdown-aware chunking strategies;
- pluggable embedding providers;
- Sentence Transformers embeddings using `all-MiniLM-L6-v2`; and
- cosine-similarity ranking in the embedding demo.

Not implemented yet:

- HTTP document upload;
- persistent document or vector storage;
- semantic or hybrid retrieval endpoints;
- LLM-powered answers and source citations; and
- authentication, observability, and deployment configuration.

## How it works

The intended document pipeline is:

```text
Document -> Parse -> Chunk -> Embed -> Vector store
```

The intended question-answering pipeline is:

```text
Question -> Embed -> Retrieve -> Context -> LLM -> Cited answer
```

The current code implements parsing, chunking, and embedding locally. The
remaining retrieval and generation stages will build on these components.

## Technology stack

| Area | Technology |
| --- | --- |
| Language | Python 3.10+ |
| API | FastAPI and Uvicorn |
| Document parsing | `pypdf` |
| Embeddings | Sentence Transformers and PyTorch |
| Testing | Pytest |
| Planned storage | PostgreSQL and pgvector |

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

Supported document extensions are `.pdf`, `.txt`, and `.md`. Scanned PDFs
without an extractable text layer and encrypted PDFs are not supported yet.

## Run tests

```bash
python -m pytest -v
```

The test suite covers the health endpoint, document loading and error cases,
chunking behavior, metadata preservation, embedding validation, and cosine
similarity.

## Project structure

```text
.
├── app/
│   ├── main.py                              # FastAPI application
│   ├── application/
│   │   ├── chunking.py                      # Chunking strategies
│   │   └── embeddings.py                    # Embedding orchestration
│   ├── domain/
│   │   ├── chunks.py                        # Chunk model
│   │   ├── documents.py                     # Parsed document models
│   │   └── embeddings.py                    # Vectors and provider protocol
│   └── infrastructure/
│       ├── document_loader.py               # PDF, TXT, and Markdown loader
│       └── sentence_transformer_embeddings.py
├── examples/
│   └── embedding_demo.md                   # Demo input document
├── scripts/
│   └── step_04_embeddings.py                # End-to-end local demo
├── tests/                                   # Unit and API tests
├── requirements.txt                         # Runtime dependencies
├── requirements-dev.txt                     # Runtime + test dependencies
└── EXPLANATION.md                           # Step-by-step project notes
```

## Roadmap

1. Add HTTP document upload and production file validation.
2. Persist parsed documents, chunks, and embeddings.
3. Add semantic and hybrid retrieval.
4. Connect retrieved context to an LLM.
5. Return source citations with generated answers.
6. Add retrieval and answer-quality evaluation.
7. Add authentication, observability, Docker, CI/CD, and deployment support.

See [EXPLANATION.md](EXPLANATION.md) for architectural background and detailed
implementation notes.
