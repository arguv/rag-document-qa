
from fastapi import FastAPI

app = FastAPI(
    title="RAG Document Q&A API",
    description="Document question answering using RAG",
    version="0.1.0",
    docs_url="/swagger",
    redoc_url=None,
    openapi_url="/json.json",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "rag-document-qa",
    }
