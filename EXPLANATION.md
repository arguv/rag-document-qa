# RAG Document Q&A — Architecture and Implementation Notes

This project is a modular Python implementation of a Retrieval-Augmented
Generation (RAG) pipeline. It separates document parsing, chunking, embedding,
storage, retrieval, and answer generation so each stage can be tested and
replaced independently.

## Current status

The repository currently contains:

- a FastAPI application with `GET /health`;
- PDF, TXT, and Markdown document loading;
- immutable document and chunk domain models;
- fixed-size, recursive, and Markdown-aware chunking;
- Sentence Transformers embeddings;
- PostgreSQL/pgvector persistence and cosine-distance search;
- semantic retrieval with top-k, document, and similarity filtering;
- source-labeled retrieval context;
- an OpenAI Responses API answer generator using structured output; and
- an answer service that validates answerability and citations.

HTTP upload, retrieval, and question-answering endpoints are intentionally not
implemented yet. The current workflows are exercised through `scripts/`.

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
```

The database URL is required by the database session factory. A valid OpenAI
key is required only by `OpenAIAnswerGenerator` and the step 7 demo. Use an
OpenAI model name available in your account.

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

## Known limitations and next steps

- The API does not yet expose upload, retrieval, or answer endpoints.
- Documents are loaded from local filesystem paths and text files expect UTF-8.
- Scanned PDFs need an OCR stage that is not present yet.
- File size, content sniffing, and upload security limits are not implemented.
- Retrieval is semantic-only; hybrid retrieval and reranking are future work.
- Retrieval and answer quality need a formal evaluation suite.
- Authentication, observability, CI/CD, and deployment configuration are
  future work.

The current design keeps these additions localized: HTTP adapters can call the
existing application services, while new retrievers or generators can satisfy
the existing domain protocols.
