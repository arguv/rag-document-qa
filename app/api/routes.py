import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID
from dataclasses import replace
from typing import Literal
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select, delete, func, text
from sqlalchemy.orm import Session
from app.api import dependencies as deps
from app.application.chunking import DocumentAwareChunker, ChunkingSettings
from app.application.embeddings import embed_chunks
from app.domain.generation import AnswerResult
from app.domain.retrieval import RetrievalResult
from app.infrastructure.database.models import StoredDocument, StoredChunk
from app.infrastructure.database.repository import PgvectorRepository
from app.infrastructure.document_loader import load_document, SUPPORTED_EXTENSIONS, DocumentLoadError

router = APIRouter(dependencies=[Depends(APIKeyHeader(name='X-API-Key', auto_error=False))])


class Question(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: UUID | None = None
    min_similarity: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    search_mode: Literal['vector', 'hybrid'] = 'vector'
    rerank: bool = False
    multi_query: bool = False
    neighbors: bool = False
    agentic: bool = False

    @field_validator('question')
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError('Question cannot be empty')
        return value.strip()

    def retrieval_options(self):
        return self.model_dump(include={'top_k', 'document_id', 'min_similarity'})


@router.post('/documents', status_code=201)
def ingest(file: UploadFile = File(...), db: Session = Depends(deps.session)):
    name = (file.filename or '').replace('\\', '/').rsplit('/', 1)[-1]
    if not name or len(name) > 255 or any(ord(c) < 32 or ord(c) == 127 for c in name) or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, 'Upload a PDF, TXT, or Markdown file with a filename of at most 255 characters')
    try:
        with TemporaryDirectory(prefix='rag-') as directory:
            # Never use the user-supplied path for temporary storage.
            path = Path(directory) / ('upload' + Path(name).suffix.lower())
            size = 0
            with path.open('wb') as output:
                while data := file.file.read(65536):
                    size += len(data)
                    if size > int(os.getenv('MAX_UPLOAD_BYTES', '10485760')):
                        raise HTTPException(413, 'Document too large')
                    output.write(data)
            document = replace(load_document(path), filename=name)
        if sum(len(s.text) for s in document.sections) > int(os.getenv('MAX_DOCUMENT_CHARS', '2000000')):
            raise HTTPException(413, 'Extracted document too large')
        chunks = DocumentAwareChunker(ChunkingSettings()).chunk(document)
        provider = deps.embeddings()
        embedded = embed_chunks(chunks, provider)
        identifier = PgvectorRepository(db).save_document(document, embedded, provider.model_name)
        db.commit()
        return {'id': identifier, 'filename': name, 'chunks': len(chunks)}
    except DocumentLoadError as exc:
        raise HTTPException(422, 'Document could not be parsed or has no extractable text') from exc
    finally:
        file.file.close()


@router.get('/documents')
def documents(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0), db: Session = Depends(deps.session)):
    rows = db.scalars(select(StoredDocument).order_by(StoredDocument.created_at.desc(), StoredDocument.id).limit(limit).offset(offset))
    return [{'id': d.id, 'filename': d.filename, 'file_type': d.file_type,
             'embedding_model': d.embedding_model, 'created_at': d.created_at} for d in rows]


@router.get('/documents/{document_id}')
def document(document_id: UUID, db: Session = Depends(deps.session)):
    stored = db.get(StoredDocument, document_id)
    if stored is None:
        raise HTTPException(404, 'Document not found')
    count = db.scalar(select(func.count()).select_from(StoredChunk).where(StoredChunk.document_id == document_id))
    return {'id': stored.id, 'filename': stored.filename, 'chunks': count}


@router.delete('/documents/{document_id}', status_code=204)
def remove_document(document_id: UUID, db: Session = Depends(deps.session)):
    result = db.execute(delete(StoredDocument).where(StoredDocument.id == document_id))
    if result.rowcount == 0:
        raise HTTPException(404, 'Document not found')
    db.commit()


def ensure_scope(db, options):
    if options.document_id and db.get(StoredDocument, options.document_id) is None:
        raise HTTPException(404, 'Document not found')


@router.post('/search', response_model=RetrievalResult)
def search(options: Question, db: Session = Depends(deps.session)):
    ensure_scope(db, options)
    retrieval, _ = deps.services(db, options, with_generator=False)
    return retrieval.retrieve(options.question, **options.retrieval_options())


@router.post('/questions', response_model=AnswerResult)
def answer(options: Question, db: Session = Depends(deps.session)):
    ensure_scope(db, options)
    retrieval, service = deps.services(db, options)
    if options.agentic:
        from app.application.agent import AgenticAnswerService
        service = AgenticAnswerService(retrieval, service)
    return service.answer(options.question, **options.retrieval_options())


@router.get('/ready')
def ready(db: Session = Depends(deps.session)):
    db.execute(text('SELECT 1 FROM documents LIMIT 1'))
    return {'status': 'ready', 'llm_backend': os.getenv('LLM_BACKEND', 'mock')}
