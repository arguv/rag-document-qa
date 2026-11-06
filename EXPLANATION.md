# RAG Document Q&A

A FastAPI service for building a Retrieval-Augmented Generation (RAG)
application. The project currently provides the API foundation and a reusable
document-ingestion component for PDF, TXT, and Markdown files.

## Current status

The project currently includes:

- a FastAPI application in `app/main.py`;
- a `GET /health` endpoint;
- generated Swagger UI at `/swagger`;
- an OpenAPI JSON document at `/json.json`; and
- a pytest test for the health endpoint;
- immutable `DocumentSection` and `ParsedDocument` domain models;
- a document loader in `app/infrastructure/document_loader.py`;
- PDF, TXT, and Markdown text extraction;
- PDF page metadata preservation; and
- consistent `DocumentLoadError` handling for invalid or unsupported files.

HTTP document upload, chunking, embeddings, vector search, and LLM answer
generation are not implemented yet.

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
│   ├── main.py                    # FastAPI application and health endpoint
│   ├── domain/
│   │   └── documents.py           # Parsed document data models
│   └── infrastructure/
│       └── document_loader.py     # PDF, TXT, and Markdown loader
├── tests/
│   ├── test_health.py             # API health check test
│   └── test_document_loader.py    # Document ingestion tests
├── requirements.txt               # Runtime dependencies
├── requirements-dev.txt           # Test and development dependencies
└── EXPLANATION.md                 # Project documentation
```

## Planned next steps

1. Add HTTP document upload and production file validation.
2. Split parsed documents into searchable chunks.
3. Generate embeddings and persist them in a vector database.
4. Retrieve relevant chunks for a user question.
5. Connect retrieval context to an LLM and return a cited answer.
6. Add tests for chunking, retrieval quality, and answer generation.

---

# Step 2 — Document Ingestion and Parsing

## 1. Goal

The goal of Step 2 is to implement a reusable document loader that reads external files, extracts text, and preserves metadata.

Supported document formats:

- PDF (`.pdf`)
- Plain text (`.txt`)
- Markdown (`.md`)

The loader converts these formats into a common Python data structure called `ParsedDocument`.

This prepares our documents for the next stages of RAG: chunking, embedding generation, and retrieval.

## 2. What is Document Ingestion?

Document ingestion is the process of loading data from external sources into an application for processing.

In a RAG system, documents provide the external knowledge that an LLM will later use to answer questions.

The complete ingestion pipeline is:

**Documents → Parsing → Chunking → Embeddings → Vector Database**

In Step 2, we implement only the first two operations:

**Documents → Parsing**

Example:

Input: `company_policy.pdf`

Output:

- Extracted document text
- Original filename
- File format
- Total number of PDF pages
- Page number associated with each extracted section

We do not generate embeddings or store documents in a database yet.

## 3. What is Document Parsing?

Parsing is the process of extracting useful content from a file and converting it into a format the application can process.

Different document formats require different parsing techniques.

### TXT files

A TXT file contains plain text.

Python can read it directly using `Path.read_text()`.

Example:

`path.read_text(encoding="utf-8")`

The `encoding="utf-8"` parameter tells Python how to decode the file's bytes into text.

### Markdown files

Markdown files also contain plain text, but include formatting syntax such as headings, lists, and links.

Example:

`# RAG Introduction`

Our current loader reads Markdown as text and preserves its Markdown syntax.

It does not convert Markdown into HTML or extract its semantic structure.

### PDF files

PDF files have a more complex internal representation.

We use the `pypdf` library to extract text from individual PDF pages.

Important operations:

- `PdfReader(file)` — Opens and parses the PDF.
- `reader.pages` — Provides access to PDF pages.
- `page.extract_text()` — Attempts to extract text from a page.
- `len(reader.pages)` — Returns the total number of pages.

A limitation of `pypdf` is that it does not perform OCR (Optical Character Recognition).

Therefore, scanned PDFs containing only images cannot currently be processed successfully unless they also contain an extractable text layer.

## 4. Project Architecture

We introduced two important packages:

**Domain layer**

`app/domain/documents.py`

Contains the data models that represent parsed documents.

**Infrastructure layer**

`app/infrastructure/document_loader.py`

Contains the implementation responsible for reading and parsing physical files.

Project structure:

```text
app/
├── __init__.py
├── main.py
├── domain/
│   ├── __init__.py
│   └── documents.py
└── infrastructure/
    ├── __init__.py
    └── document_loader.py
```

### Why separate Domain and Infrastructure?

The domain layer describes what our application works with.

The infrastructure layer describes how the application interacts with external systems.

For example:

The domain knows that a document contains text, a filename, and metadata.

It does not need to know whether that text came from a PDF, a database, or a remote API.

This design follows the Separation of Concerns principle.

Benefits include:

- Easier unit testing
- Better maintainability
- Clear responsibilities
- Reduced coupling
- Easier replacement of external libraries

This is part of our gradual transition toward a modular, senior-level application architecture.

## 5. Document Data Models

We created two models using Python's `dataclasses` module.

### DocumentSection

Represents a section of extracted document content.

Fields:

- `text: str` — Extracted text
- `page_number: int | None` — Original PDF page number, when available

The page number is optional because TXT and Markdown files do not have PDF-style pages.

For PDF files, our initial implementation creates one section per page that contains extractable, non-empty text.

### ParsedDocument

Represents an entire parsed document.

Fields:

- `filename` — Original filename
- `file_type` — File extension without the dot
- `sections` — Tuple of extracted DocumentSection objects
- `page_count` — Total PDF page count, or None for TXT/Markdown

### Why use @dataclass?

The `@dataclass` decorator automatically generates useful methods for classes primarily intended to store data.

These include an initializer and a representation method.

This reduces boilerplate code.

### Why use frozen=True?

`@dataclass(frozen=True)` prevents normal reassignment of an object's fields after creation.

This is useful because parsed document results should not be accidentally modified during later processing.

It provides shallow immutability, not automatic deep immutability of every possible nested object.

### Why use a tuple?

`ParsedDocument.sections` is stored as a tuple instead of a list.

A tuple cannot have elements added, removed, or replaced after creation.

This works well with immutable document models.

### What is @property?

We defined a `text` property on `ParsedDocument`.

It joins text from all document sections using two newline characters.

This allows us to access:

`document.text`

instead of manually combining every section.

The original sections remain available separately so we can preserve page information during future chunking.

## 6. Document Loader Implementation

Main function:

`load_document(file_path: str | Path) -> ParsedDocument`

The function accepts a filesystem path and returns a structured document object.

### Processing flow

1. Convert the provided path into a `Path` object.
2. Check whether the file extension is supported.
3. Verify that the file exists and is a regular file.
4. Choose the appropriate parser.
5. Extract text and metadata.
6. Reject documents without extractable text.
7. Return a `ParsedDocument`.

### Why use pathlib.Path?

`pathlib` provides an object-oriented way to work with filesystem paths.

Useful operations include:

- `Path(file_path)` — Create a path object.
- `path.suffix` — Read the file extension.
- `path.name` — Get the filename.
- `path.is_file()` — Check whether it is a regular file.
- `path.read_text()` — Read text content.
- `path.open()` — Open the file.

Using `pathlib` generally produces clearer, cross-platform filesystem code.

### Why have separate parsing functions?

We use:

- `_load_text()` for TXT and Markdown.
- `_load_pdf()` for PDF documents.

Different formats require different parsing logic.

Separating these functions improves readability, testing, and maintainability.

The underscore prefix indicates that these functions are intended for internal module use. It does not make them strictly private in Python.

### Why return the same data type?

Both parsers ultimately provide data used to construct a `ParsedDocument`.

This means later RAG components do not need separate processing logic for every file format.

For example, the chunking component will be able to accept a parsed document without knowing which parser created it.

## 7. Metadata and Source Tracking

Metadata describes information about a document beyond its main textual content.

Our current metadata includes:

- Filename
- File type
- PDF page count
- Section page numbers

### Why is metadata important in RAG?

Imagine a user asks:

"What is the company's vacation policy?"

Our retrieval system may find relevant information on page 8 of `employee_handbook.pdf`.

Later, our RAG application can generate an answer with a reference such as:

**Source: employee_handbook.pdf, page 8**

Without preserving source metadata, generating accurate document references becomes more difficult.

### Important distinction

`page_count` represents the total number of pages in a PDF.

`len(document.sections)` represents the number of sections containing extracted text.

These values may differ when a PDF contains blank pages or pages without extractable text.

### Metadata in future steps

Later, we can extend metadata with:

- Document ID
- Chunk ID
- Section heading
- Content hash
- Upload timestamp
- Document version
- Access permissions

These fields will support retrieval filtering, document updates, and security.

## 8. Error Handling

We introduced a custom exception:

`DocumentLoadError(ValueError)`

It represents errors encountered while loading a document.

### Why use a custom exception?

Without custom exceptions, different parsing libraries may raise many unrelated errors.

Our application should have a consistent way to represent document-loading failures.

Examples include:

- Unsupported file format
- Missing file
- Empty document
- Invalid PDF
- Encrypted PDF
- Unreadable text file

### Exception chaining

We use the `raise ... from exc` syntax when appropriate.

This preserves the original exception as the cause of our custom exception.

It helps developers understand the underlying problem during debugging.

### Why reject empty documents?

A document without usable text cannot contribute meaningful information to our current text-based RAG pipeline.

Rejecting it early prevents unnecessary processing in later steps.

### Current validation limitations

Checking a file extension does not prove that the actual file content matches the declared type.

We also do not currently enforce file-size limits or resource limits.

These checks will become important when we implement HTTP uploads and production security.

## 9. Automated Testing

We use Pytest to verify document-loader behavior.

Test file:

`tests/test_document_loader.py`

### What is tmp_path?

`tmp_path` is a built-in Pytest fixture.

It provides a temporary directory for an individual test.

This allows us to create test files without modifying real user documents.

It also helps keep tests independent.

### Why use ReportLab?

ReportLab is used only for testing.

It generates real PDF documents containing known text.

Our tests then use `pypdf` to extract that text and verify the results.

This lets us test PDF parsing without depending on external PDF files.

### What do we test?

| Test | Purpose |
|---|---|
| test_load_txt | Verify TXT extraction |
| test_load_markdown | Verify Markdown extraction |
| test_load_pdf | Verify PDF extraction and page numbers |
| test_unsupported_format | Reject unsupported extensions |
| test_missing_file | Handle missing files |
| test_empty_document | Reject empty text |
| test_pdf_without_text | Reject PDFs without extractable text |

We also retain the existing FastAPI health test from Step 1.

### Run tests

`python -m pytest -v`

Actual verified result:

**8 passed in 0.60 seconds**

This confirms that all eight current tests passed in the development environment.

It does not mean every possible PDF format or production failure scenario has been tested.

## 10. Dependencies

We introduced two requirements files.

### requirements.txt

Contains direct application dependencies:

- FastAPI
- Uvicorn
- PyPDF
- HTTPX2

### requirements-dev.txt

References `requirements.txt` and adds:

- Pytest
- ReportLab

The line `-r requirements.txt` tells pip to include the main application requirements.

This separation helps keep development-only tooling distinct from the application's direct runtime dependencies.

We also verified the installed environment using:

`python -m pip check`

Result:

`No broken requirements found.`

## 11. What is __init__.py?

`__init__.py` is a special Python file that identifies a directory as a regular Python package.

In this step, we use it in:

- `app/`
- `app/domain/`
- `app/infrastructure/`

It makes package boundaries explicit.

For example:

`from app.domain.documents import ParsedDocument`

Since Python 3.3, implicit namespace packages can also work without `__init__.py`.

Therefore, this file is not always required for imports.

Our Tiny LLM project used module execution with `python -m`, which can work without `__init__.py` when modules are discoverable through Python's import path.

For our RAG project, we intentionally use explicit regular packages because we are building a modular backend application.

Empty `__init__.py` files are valid and should be committed to Git when they are part of the intended package structure.

## 12. Current Limitations

At the end of Step 2:

- Documents are loaded from local filesystem paths.
- TXT and Markdown parsing expects UTF-8.
- Scanned PDFs without extractable text are unsupported.
- Encrypted PDFs are unsupported.
- Markdown structure is not yet parsed into headings and sections.
- HTTP document-upload endpoints have not been implemented.
- File-size and content-validation limits are not implemented.
- Chunking is not yet implemented.
- Embeddings and vector storage are not yet implemented.

These limitations are intentional at this stage.

We are building and validating one component at a time.

## 13. Step 2 Final Result

We implemented a reusable document ingestion component with:

- Support for PDF, TXT, and Markdown.
- Structured domain models.
- Separation of domain and infrastructure logic.
- PDF page metadata preservation.
- Consistent error handling.
- Automated tests.

**All 8 tests passed successfully.**

Our document loader is now ready to provide extracted content to the future chunking component.

### Key Takeaway

Document ingestion is the foundation of RAG.

Before generating embeddings or performing semantic search, we must reliably extract text and preserve information about its source.

The quality and structure of ingested documents directly influence the quality of retrieval and the accuracy of future source citations.


---

# Step 3 — Document Chunking

## Goal

Implement multiple text chunking strategies and preserve
document metadata for future embedding and retrieval.

## 1. What is Chunking?

Chunking divides large documents into smaller text units.

Each chunk can later be converted into an embedding
and stored in a vector database.

Smaller chunks improve retrieval granularity, but
excessively small chunks may lose important context.

## 2. Why Do We Need Chunking?

Large documents can exceed embedding model input limits.

Using entire documents as retrieval units can also
include large amounts of irrelevant text.

Chunking helps isolate relevant passages.

## 3. Chunk Size and Overlap

Chunk size defines the maximum number of characters
in our current implementation.

Overlap repeats part of the text between consecutive
chunks from the same source section.

Example:

Text: ABCDEFGHIJKL
Size: 5
Overlap: 2

Chunks:
ABCDE
DEFGH
GHIJK
JKL

Overlap reduces boundary-related context loss but
increases the amount of processed text.

Character count is different from token count.

## 4. Fixed-size Chunking

Fixed-size chunking splits text into character windows.

Advantages:
- Simple implementation
- Predictable maximum size
- Low processing overhead

Disadvantages:
- Can split sentences
- Does not understand document structure

## 5. Recursive Chunking

Our recursive splitter tries separators in priority order:

1. Paragraph boundaries
2. Newlines
3. Sentence-like boundaries
4. Spaces
5. Character-length fallback

It recursively checks lower-priority boundaries when
a preferred separator cannot be used.

This is a lightweight boundary-aware approach, not
a complete linguistic sentence parser.

## 6. Document-aware Chunking

Document-aware chunking uses information about the
original document's structure.

For Markdown:
- Identify headings
- Split at heading boundaries
- Preserve heading metadata

For PDF:
- Process extracted pages independently
- Preserve original page numbers

For TXT:
- Apply boundary-aware splitting

## 7. Architecture

New files:

app/domain/chunks.py
app/application/chunking.py

The domain layer defines the DocumentChunk data model.

The application layer implements chunking algorithms.

Chunking depends on domain models but does not depend
on FastAPI or the PDF parsing library.

## 8. Strategy Design Pattern

Each chunker implements the same interface:

chunk(document) -> list[DocumentChunk]

Implementations:

- FixedSizeChunker
- RecursiveChunker
- DocumentAwareChunker

We use a Python Protocol to describe the interface.

This makes chunking strategies interchangeable.

## 9. Configuration Validation

ChunkingSettings stores:

- size
- overlap

Validation ensures:

- size > 0
- overlap >= 0
- overlap < size

This prevents invalid window calculations.

## 10. Metadata Preservation

Each DocumentChunk contains:

- text
- filename
- file_type
- page_number
- chunk_index
- strategy
- heading

Metadata allows future retrieval results to reference
their source documents.

Chunk indices are sequential within one chunking result.
They are not yet persistent database identifiers.

## 11. Testing

We test:

- Fixed-size splitting
- Overlapping chunks
- Recursive paragraph boundaries
- Character-boundary fallback
- Chunk-length limits
- Markdown headings
- PDF page metadata
- Invalid configuration
- Empty content

Step 3 adds 12 test cases.

Expected complete test suite:
20 passed.

## 12. Current Limitations

- Chunk sizes are measured in characters, not tokens.
- Boundary detection uses simple text rules.
- Markdown handling supports basic ATX headings.
- PDFs use extracted page boundaries.
- No semantic chunking is implemented yet.
- Chunks are not stored in a database.
- Embeddings are not generated.
- No retrieval evaluation has been performed.

These are deliberate scope limitations for this step.

## 13. Step 3 Result

We now have three interchangeable chunking strategies
that produce structured DocumentChunk objects.

The chunking engine works with ParsedDocument objects
created by the document loader from Step 2.

The next step introduces embedding generation and
vector similarity.

## Key Takeaway

Chunking is not simply splitting text.

The strategy affects which information stays together,
how much context is preserved, and what the retrieval
system can find later.

Chunking choices should ultimately be evaluated using
real questions and retrieval-quality measurements.


---

# Step 4 — Embedding Generation and Vector Similarity

## Goal

Convert document chunks and user questions into numerical
vectors and compare them using cosine similarity.

## 1. What is an embedding?

An embedding is a numerical representation of text.

Embedding models are trained to place semantically related
texts near one another in a multidimensional vector space.

Example:

"How do I reset my password?"

The embedding model returns a vector such as:

[0.12, -0.35, 0.81, ...]

The values are learned representations, not manually
assigned meanings.

## 2. What is an embedding model?

An embedding model converts input text into vectors.

We use:

sentence-transformers/all-MiniLM-L6-v2

It generates 384-dimensional embeddings.

The model runs locally using PyTorch and CPU inference.

The model has a maximum input length of 256 tokens.
Longer inputs may be truncated.

## 3. What are vector dimensions?

Dimensions are the number of numerical elements in
an embedding.

For our model:

1 text -> 384 floating-point numbers

Both document chunks and user questions must be embedded
using compatible embedding models.

Their vectors must have matching dimensions.

## 4. Why embed document chunks?

After chunking, documents still contain ordinary text.

Embedding generation converts those chunks into
numerical representations suitable for similarity search.

Pipeline:

Document -> Chunks -> Embeddings

The next step will store these embeddings in PostgreSQL
using pgvector.

## 5. What is cosine similarity?

Cosine similarity measures the directional similarity
between two numerical vectors.

Its mathematical range is from -1 to 1.

1 means the same direction.
0 means orthogonal vectors.
-1 means opposite directions.

A high similarity score does not guarantee that a chunk
correctly answers a question.

## 6. What is vector normalization?

Normalization adjusts a nonzero vector to unit length.

With unit-length vectors, their dot product equals
their cosine similarity.

We request normalized embeddings from
Sentence Transformers.

## 7. Architecture

New files:

app/domain/embeddings.py
app/application/embeddings.py
app/infrastructure/sentence_transformer_embeddings.py

The domain defines EmbeddedChunk and EmbeddingProvider.

The application layer coordinates vector generation
and implements similarity calculations.

The infrastructure layer integrates Sentence Transformers.

## 8. What is EmbeddingProvider?

EmbeddingProvider is a Python Protocol.

It defines the method:

encode(texts) -> list[Vector]

Our application can use different embedding providers
without depending directly on their libraries.

This supports Dependency Inversion and testability.

## 9. How does embed_chunks() work?

1. Receive DocumentChunk objects.
2. Validate that chunk text is not empty.
3. Send the text to the embedding provider.
4. Validate the returned vectors.
5. Create EmbeddedChunk objects.
6. Preserve the original chunk metadata.

## 10. Why validate vectors?

Embedding generation is an external dependency.

Our application checks:

- The returned vector count
- Consistent vector dimensions
- Nonempty vectors
- Finite numerical values

This helps detect malformed provider responses.

## 11. How does embed_query() work?

It converts a user question into an embedding.

The same model used for document chunks is used
for the query.

Later, retrieval will compare query embeddings
against stored document embeddings.

## 12. What is in-memory similarity search?

We can compare a question against document embeddings
without a database.

The process:

Question -> Embedding -> Cosine Similarity -> Ranking

This works for small examples.

A vector database becomes useful for persistent storage,
filtering, indexing, and larger document collections.

## 13. How do we test embeddings?

Unit tests use a fake embedding provider.

This avoids downloading a real model during automated
tests and produces predictable test results.

We separately run a manual integration example using
the real pretrained embedding model.

Step 4 adds 16 test cases.

Expected total project test count: 36.

## 14. Current limitations

- No persistent vector database
- No vector indexing
- No document upload API
- No LLM answer generation
- No source-aware retrieval service
- Character-based rather than token-aware chunking
- No embedding model version stored with vectors
- No formal semantic retrieval evaluation
- No automated real-model integration test

The local model also has input-length and language
limitations. It is primarily suitable for English text.

## 15. Step 4 Result

The project now supports:

- Local text embedding generation
- Query embedding generation
- Structured EmbeddedChunk objects
- Vector validation
- Cosine similarity
- Small in-memory similarity comparisons

## Key Takeaway

Embeddings transform text into vectors that allow
computers to compare semantic similarity.

Chunking determines which text is represented.

The embedding model determines how that text is
represented numerically.

The retrieval system will use those representations
to identify relevant information.

Next: Step 5 — PostgreSQL and pgvector.
