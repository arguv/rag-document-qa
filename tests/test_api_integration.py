"""Exercise the full HTTP-to-pgvector pipeline with deterministic local providers."""
import os
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session
from app.main import create_app
from app.api import dependencies as deps
from app.domain.generation import AnswerDraft


@pytest.mark.integration
def test_persistent_api_roundtrip(monkeypatch):
    load_dotenv()
    url = os.getenv('TEST_DATABASE_URL')
    if not url:
        pytest.skip('TEST_DATABASE_URL not configured')
    if make_url(url).database != 'rag_documents_test':
        raise RuntimeError('Unsafe test database configuration')
    engine = create_engine(url, connect_args={'connect_timeout': 5})
    app = create_app()

    def session():
        with Session(engine) as db:
            yield db

    class Embeddings:
        model_name = 'api-integration-test'

        def encode(self, texts):
            return [(1.,) + (0.,) * 383 for _ in texts]

    class Generator:
        def generate(self, question, context):
            return AnswerDraft('RAG retrieves document context.', True, (1,))

    app.dependency_overrides[deps.session] = session
    monkeypatch.setattr(deps, 'embeddings', lambda: Embeddings())
    monkeypatch.setattr(deps, 'generator', lambda: Generator())
    monkeypatch.setenv('APP_ENV', 'development')
    monkeypatch.delenv('API_KEY', raising=False)
    identifier = None
    try:
        with TestClient(app) as client:
            response = client.post('/documents', files={'file': ('pipeline.md', b'# RAG\nRAG retrieves document context.\n' * 30)})
            assert response.status_code == 201, response.text
            identifier = response.json()['id']
            assert client.get(f'/documents/{identifier}').json()['chunks'] > 1
            assert client.get('/ready').status_code == 200
            for extra in [{}, {'search_mode': 'hybrid'}, {'search_mode': 'hybrid', 'min_similarity': .5}, {'multi_query': True, 'neighbors': True}, {'agentic': True}]:
                payload = {'question': 'How does RAG retrieve context?', 'document_id': identifier, **extra}
                response = client.post('/questions', json=payload)
                assert response.status_code == 200, response.text
                assert response.json()['status'] == 'answered'
                assert all(c['document_id'] == identifier for c in response.json()['citations'])
            response = client.post('/search', json={'question': 'RAG', 'document_id': identifier, 'search_mode': 'hybrid'})
            assert response.status_code == 200, response.text
            assert response.json()['matches']
            assert client.delete(f'/documents/{identifier}').status_code == 204
            assert client.get(f'/documents/{identifier}').status_code == 404
            identifier = None
    finally:
        if identifier:
            from uuid import UUID
            from sqlalchemy import delete
            from app.infrastructure.database.models import StoredDocument
            with Session(engine) as db:
                db.execute(delete(StoredDocument).where(StoredDocument.id == UUID(identifier)))
                db.commit()
        engine.dispose()
