# RAG Document Q&A

A FastAPI service for building a Retrieval-Augmented Generation (RAG)
application. The service will eventually answer questions using information
retrieved from uploaded documents.

## Current status

The project is at the initial API-foundation stage. It currently includes:

- a FastAPI application in `app/main.py`;
- a `GET /health` endpoint;
- generated Swagger UI at `/swagger`;
- an OpenAPI JSON document at `/json.json`; and
- a pytest test for the health endpoint.

Document upload, parsing, chunking, embeddings, vector search, and LLM answer
generation are planned but are not implemented yet.

## What is RAG?

Retrieval-Augmented Generation combines two steps:

1. retrieve relevant information from an external knowledge base; and
2. provide that information to a language model so it can generate a grounded
   answer.

The intended document-ingestion pipeline is:

```text
Documents -> Parsing -> Chunking -> Embeddings -> Vector database
```

The intended question-answering pipeline is:

```text
Question -> Retrieval -> Context -> LLM -> Answer
```

These pipelines are separate so documents can be processed once and reused
for many questions.

## Requirements

- Python 3.10 or newer
- `pip`

Dependencies are listed in `requirements.txt`.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the API

Start the development server from the project root:

```bash
uvicorn app.main:app --reload
```

The service is available at `http://127.0.0.1:8000`.

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

Interactive API documentation is available at
`http://127.0.0.1:8000/swagger`.

## Run tests

```bash
pytest
```

The health test verifies both the HTTP 200 response and the response body.

## Project structure

```text
.
├── app/
│   ├── __init__.py
│   └── main.py          # FastAPI application and health endpoint
├── tests/
│   └── test_health.py   # API health check test
├── requirements.txt     # Python dependencies
└── EXPLANATION.md       # Project documentation
```

## Planned next steps

1. Add document upload and file-type validation.
2. Parse documents and split them into searchable chunks.
3. Generate embeddings and persist them in a vector database.
4. Retrieve relevant chunks for a user question.
5. Connect retrieval context to an LLM and return a cited answer.
6. Add tests for ingestion, retrieval quality, and answer generation.
