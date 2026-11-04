# RAG Document Q&A

A production-oriented Retrieval-Augmented Generation (RAG) application built with Python and FastAPI.

The project aims to provide accurate, context-aware answers to questions about uploaded documents by combining semantic retrieval with Large Language Models (LLMs).

**Project Status:** In development — Step 1: Foundation

## Overview

RAG combines information retrieval with generative AI. Instead of relying exclusively on an LLM's pretrained knowledge, the application retrieves relevant information from external documents and provides it as context when generating answers.

The application is being developed incrementally, from a basic document Q&A system to an advanced RAG architecture with production-oriented engineering practices.

## Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Backend Framework | FastAPI |
| Testing | Pytest |
| Database (planned) | PostgreSQL |
| Vector Search (planned) | pgvector |
| Embeddings (planned) | Sentence Transformers |
| LLM Integration (planned) | Configurable LLM API |
| Advanced Workflows (planned) | LangGraph |
| Containerization (planned) | Docker |

## Features

**Currently implemented**

- FastAPI application setup
- Health-check REST endpoint
- Automated health endpoint test
- Interactive API documentation

**Planned features**

- PDF, TXT, and Markdown document ingestion
- Multiple document chunking strategies
- Embedding generation
- Persistent vector storage
- Semantic and hybrid search
- Retrieval reranking
- LLM-powered question answering
- Source citations
- RAG quality evaluation
- Authentication and document access control
- Observability and monitoring
- Docker and CI/CD

## Getting Started

### Prerequisites

- Python 3.10 or newer
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/arguv/rag-document-qa.git
cd rag-document-qa
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Run the Application

```bash
python -m uvicorn app.main:app --reload --port 8001
```

The API will be available at:

`http://localhost:8001`

### API Documentation

With the current custom configuration:

- Swagger UI: `http://localhost:8001/swagger`
- OpenAPI schema: `http://localhost:8001/json.json`

### Health Check

```bash
curl http://localhost:8001/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "rag-document-qa"
}
```

## Running Tests

```bash
python -m pytest -v
```

The current test suite verifies the health endpoint.

Additional unit, integration, and RAG evaluation tests will be introduced during development.

## Project Structure

```text
rag-document-qa/
├── app/
│   ├── __init__.py
│   └── main.py
├── tests/
│   └── test_health.py
├── .gitignore
├── EXPLANATION.md
├── README.md
└── requirements.txt
```

The structure will evolve as the ingestion, chunking, embedding, retrieval, and generation components are implemented.

## Learning Documentation

For detailed explanations of implementation steps, architecture decisions, and technical concepts, see [EXPLANATION.md](EXPLANATION.md).

## Development Approach

The project follows these engineering principles:

- Modular and maintainable architecture
- Separation of concerns
- Testable components
- Clear API contracts
- Incremental development
- Production-oriented engineering practices without unnecessary complexity

## Roadmap

1. Project foundation
2. Document ingestion and parsing
3. Chunking strategies
4. Embeddings and vector similarity
5. PostgreSQL and pgvector
6. Semantic retrieval and basic RAG
7. LLM integration and citations
8. Hybrid search and reranking
9. Advanced retrieval techniques
10. RAG evaluation
11. Agentic RAG with LangGraph
12. Security, observability, and production engineering
13. Docker, CI/CD, and deployment

The roadmap may evolve as the project develops.