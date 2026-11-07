# RAG Document Q&A — Architecture and Implementation Notes

This project is a modular Python implementation of a Retrieval-Augmented
Generation (RAG) pipeline. It separates document parsing, chunking, embedding,
storage, retrieval, and answer generation so each stage can be tested and
replaced independently.

## Current status

The repository currently contains:

- a FastAPI application with health, readiness, document, search, question,
  Swagger, OpenAPI, and metrics endpoints;
- PDF, TXT, and Markdown document loading;
- immutable document and chunk domain models;
- fixed-size, recursive, and Markdown-aware chunking;
- Sentence Transformers embeddings;
- PostgreSQL/pgvector persistence and cosine-distance search;
- semantic retrieval with top-k, document, and similarity filtering;
- source-labeled retrieval context;
- an OpenAI Responses API answer generator using structured output; and
- an answer service that validates answerability and citations;
- optional hybrid retrieval, reranking, query decomposition, neighboring
  context, offline evaluation, and bounded agentic retry; and
- API boundary controls, Docker Compose deployment files, and CI.

HTTP upload, retrieval, question answering, document management and operational
endpoints are implemented in `app/api/`. Scripts remain available as demos.

## What is RAG?

RAG combines retrieval from an external knowledge base with generation by a
language model. The model receives relevant document passages instead of
having to rely only on its training data.

The document pipeline runs once per document:

```text
Document -> Parse -> Chunk -> Embed -> Vector store
```

The question pipeline runs for each question:

```text
Question -> Embed -> Retrieve -> Filter -> Context -> LLM -> Answer + citations
```

Separating these pipelines lets the application process documents once and
reuse their vectors for many questions.

## Layered architecture

### Domain layer

The domain package describes the problem space without depending on FastAPI,
PostgreSQL, or an external model provider:

- `documents.py` — parsed documents and page-aware sections;
- `chunks.py` — searchable chunks and chunk metadata;
- `embeddings.py` — embedding vectors and provider protocol;
- `search.py` — individual search matches;
- `retrieval.py` — retrieval results and formatted context; and
- `generation.py` — answer drafts, citations, and final answer results.

The models use frozen dataclasses where practical. This prevents later stages
from accidentally changing source data or retrieval metadata used for
citations.

### Application layer

The application package coordinates use cases:

- `chunking.py` turns parsed documents into searchable chunks;
- `embeddings.py` coordinates embedding providers;
- `retrieval.py` embeds a question, queries the repository, applies filters,
  and formats source-labeled context; and
- `answering.py` calls retrieval, invokes an answer generator only when there
  is evidence, and validates the generator's source numbers.

These services depend on domain protocols rather than concrete database or
LLM implementations, making them testable with fakes.

### Infrastructure layer

The infrastructure package contains adapters to external systems:

- `document_loader.py` reads local PDF, TXT, and Markdown files;
- `sentence_transformer_embeddings.py` uses the local
  `all-MiniLM-L6-v2` model;
- `database/` contains SQLAlchemy models, sessions, and the pgvector
  repository; and
- `openai_answer_generator.py` calls the OpenAI Responses API and parses a
  structured answer.

## Document ingestion

`load_document(path)` returns a common `ParsedDocument` regardless of file
format.

TXT and Markdown files are read as UTF-8 text. Markdown syntax is preserved so
the Markdown-aware chunker can use headings and retain heading metadata.

`pypdf` extracts PDF text page by page. Each non-empty page becomes a document
section with its original one-based page number. `page_count` records the total
number of pages and may be greater than the number of sections when a PDF has
blank or non-extractable pages.

Scanned image-only PDFs require OCR and are rejected when no text can be
extracted. Encrypted PDFs are not supported by the current loader.

`DocumentLoadError` gives callers one consistent exception type for missing,
unsupported, empty, unreadable, or invalid documents. The original exception
is retained as the cause when the parser provides one.

## Chunking

Chunking divides parsed text into units suitable for embedding and retrieval.
The implementation supports:

- fixed-size windows for predictable chunk lengths;
- recursive splitting that prefers paragraphs, newlines, sentence-like
  boundaries, spaces, and finally character boundaries; and
- Markdown-aware splitting that uses headings and preserves heading metadata.

Chunk size is measured in characters, not tokens. Overlap repeats trailing
content between adjacent chunks from the same source section. Overlap reduces
context loss at boundaries but increases processed text. PDF page numbers and
Markdown headings are carried into chunk metadata for source tracking.

## Embeddings and vector storage

The default provider uses Sentence Transformers with
`sentence-transformers/all-MiniLM-L6-v2`. It exposes its model name and vector
dimension and validates that generated vectors match the configured model.

The storage layer uses SQLAlchemy with Psycopg and the `pgvector` extension.
Each stored document records its embedding model. Searches use cosine distance
and can be restricted to matching models so incompatible vector spaces are not
mixed.

The schema is managed by Alembic:

```bash
python -m alembic upgrade head
```

The local PostgreSQL service is defined in `compose.yaml` and exposed on port
`5433`.

## Retrieval

`RetrievalService` performs these operations:

1. Validate the question and `top_k` range.
2. Embed the question with the configured provider.
3. Search PostgreSQL/pgvector for nearest chunks.
4. Optionally restrict matches to a document ID.
5. Optionally remove matches below a minimum cosine-similarity threshold.
6. Return structured matches and source-labeled context.

The context labels each match as `Source [n]`. Those numbers are used by the
answer generator to identify supporting citations.

## Answer generation and citation validation

`OpenAIAnswerGenerator` sends the question and retrieved context to the OpenAI
Responses API. It requests a structured response containing `answer`,
`is_answerable`, and `source_numbers`.

The system instructions treat retrieved text as untrusted document data, not
as instructions. The generator must answer only from supplied context and use
source numbers from the context labels.

`AnswerService` adds application safeguards:

- no LLM call is made when retrieval returns no matches;
- an unanswerable or blank answer becomes an `insufficient_context` result;
- at least one citation is required for an answer;
- every source number must refer to an actual retrieved match; and
- duplicate source numbers are removed while preserving order.

The final result contains the answer, status, and structured citations with
filename, chunk index, page number, and heading metadata.

## Configuration

Configuration is loaded from `.env` using `python-dotenv`:

```dotenv
DATABASE_URL=postgresql+psycopg://rag_user:rag_local_password@127.0.0.1:5433/rag_documents
TEST_DATABASE_URL=postgresql+psycopg://rag_user:rag_local_password@127.0.0.1:5433/rag_documents_test
OPENAI_API_KEY=replace_with_your_api_key
OPENAI_MODEL=gpt-6-luna
LLM_BACKEND=mock
APP_ENV=development
API_KEY=
MAX_UPLOAD_BYTES=10485760
MAX_DOCUMENT_CHARS=2000000
RATE_LIMIT_PER_MINUTE=60
MAX_CONCURRENT_REQUESTS=8
BM25_MAX_CHUNKS=20000
RERANKER_MODEL=
```

The database URL is required by the database session factory. `LLM_BACKEND=mock`
is the default and makes no provider calls. Set `LLM_BACKEND=openai` and provide
a valid key to use `OpenAIAnswerGenerator`; use a model name available in your
account. `API_KEY` protects data, readiness, and metrics routes when set, and
is mandatory when `APP_ENV=production`. The request, document, concurrency, and
rate settings are safety bounds rather than quality guarantees.

## Running the implementation

Install runtime and development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Start the database and apply migrations, then run the demos in sequence:

```bash
docker compose up -d db
python -m alembic upgrade head
python -m scripts.step_04_embeddings
python -m scripts.step_05_vector_storage
python -m scripts.step_06_retrieval
python -m scripts.step_07_answers
```

Run the API with `python -m uvicorn app.main:app --reload` and tests with
`python -m pytest -v`. The development requirements include Pytest and
ReportLab; ReportLab is used to create deterministic PDFs for loader tests and
is not needed by the runtime application.

## HTTP API behavior

`POST /documents` accepts PDF, TXT, and Markdown multipart uploads. `POST
/search` returns source-labeled retrieval matches and context; `POST
/questions` returns a cited answer. Both accept vector or hybrid search,
document scoping, cosine thresholds, optional reranking, multi-query retrieval,
neighbor expansion, and bounded agentic retry. `GET /ready` checks the database
without loading models. `GET /metrics` reports process-local request counts and
duration. Health, Swagger, and the OpenAPI schema are public; the other routes
are protected by `X-API-Key` when `API_KEY` is configured.

## Known limitations

Text documents expect UTF-8. Scanned PDFs require a separate OCR stage. Exact
BM25 is bounded by `BM25_MAX_CHUNKS`, and operational counters/rate limits are
process-local. HNSW is approximate and selective filters may reduce recall.
Authentication is one shared key over one shared corpus; document UUIDs are
filters, not tenant permissions. There is no chat memory, distributed job
queue, external-tool agent, automated factuality judge, or production load
certification.

## Steps 8–15 implementation

`app/api/routes.py` exposes ingestion and retrieval through FastAPI, reusing the existing document loader, chunker, embedding provider, pgvector repository and citation validator. Database sessions are request-scoped; successful ingestion/deletion commits, while failures close the session and roll back. Uploaded paths are discarded, temporary files are cleaned, and source filenames are restored on parsed document metadata. `app/api/dependencies.py` lazily caches the database engine, embedding model, optional reranker and generator. Search does not initialize an LLM client.

`SearchPipeline` obtains extra vector candidates, optionally decomposes compound questions, calculates exact BM25 against model/document-scoped stored chunks and combines ranks using RRF (constant 60). An optional cross-encoder reorders candidates. The final stage expands adjacent chunks, deduplicates evidence and trims the assembled source-labeled context to 12,000 characters. Fusion does not replace the cosine similarity field. A requested minimum cosine threshold is explicitly computed for lexical candidates. Context neighbors can have lower relevance because they provide surrounding text.

`AgenticAnswerService` is a compiled LangGraph state machine with conditional routing. Empty evidence triggers one broader retrieval with twice the top K (capped at 50) and no similarity threshold. The document scope remains unchanged. Generation runs once through `AnswerService.answer_retrieval`, so the same refusal and citation checks apply. This is a bounded adaptive retrieval workflow, not an autonomous general-purpose agent.

`application/evaluation.py` and `scripts/evaluate.py` compute deterministic retrieval and answer-overlap metrics from JSONL exports. They perform no API calls and do not establish semantic faithfulness. The example data is synthetic. Security and operational boundaries live in `api/security.py`: body caps apply before multipart parsing, authentication uses constant-time key comparison, request IDs are generated server-side, and logs omit questions, document content and provider payloads. Metrics and rate limits are bounded, process-local state.

Migration 0002 adds model filtering and cosine HNSW indexes. Database connections and statements have time limits. The Dockerfile runs as a non-root user; the optional Compose overlay gates startup on migrations. GitHub Actions provisions pgvector, migrates the isolated test database, runs tests and builds the container. Deployment configuration is implemented; successful remote deployment and production behavior still require environment-specific verification. See the README for all commands, knobs and limits.
