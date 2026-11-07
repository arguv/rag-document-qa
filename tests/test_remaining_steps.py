from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.api import dependencies as deps
from app.application.retrieval import build_context
from app.application.search_pipeline import bm25, fuse, SearchPipeline
from app.application.agent import AgenticAnswerService
from app.application.answering import AnswerService
from app.application.evaluation import retrieval_metrics, answer_metrics
from app.domain.generation import AnswerDraft
from app.domain.search import SearchMatch
from app.domain.retrieval import RetrievalResult


@pytest.fixture
def match():
    return SearchMatch(uuid4(), 'manual.txt', 'The retrieval engine uses BM25 and vectors.', None, None, 0, .8)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv('API_KEY', raising=False)
    monkeypatch.setenv('APP_ENV', 'development')
    monkeypatch.setenv('RATE_LIMIT_PER_MINUTE', '10000')
    db = Mock()
    app = create_app()
    app.dependency_overrides[deps.session] = lambda: db
    with TestClient(app) as client:
        yield client, db
    app.dependency_overrides.clear()


def test_api_validation_without_models(client):
    api, db = client
    for payload in [{'question': ' '}, {'question': 'x', 'top_k': 51}, {'question': 'x', 'document_id': 'bad'}, {'question': 'x', 'search_mode': 'typo'}]:
        assert api.post('/questions', json=payload).status_code == 422
    assert api.post('/documents', files={'file': ('a.exe', b'data')}).status_code == 415
    assert api.post('/documents', files={'file': ('a.txt', b'')}).status_code == 422
    assert api.post('/documents', files={'file': ('a.pdf', b'bad pdf')}).status_code == 422


def test_ingestion_and_transaction(client, monkeypatch):
    api, db = client
    provider = Mock(model_name='fake')
    provider.encode.side_effect = lambda texts: [(1.,) + (0.,) * 383 for _ in texts]
    monkeypatch.setattr(deps, 'embeddings', lambda: provider)
    repository = Mock()
    identifier = uuid4()
    repository.save_document.return_value = identifier
    monkeypatch.setattr('app.api.routes.PgvectorRepository', lambda session: repository)
    response = api.post('/documents', files={'file': ('../../guide.md', b'# Guide\nRAG retrieves evidence.')})
    assert response.status_code == 201
    assert response.json()['id'] == str(identifier)
    document, embedded, model = repository.save_document.call_args.args
    assert document.filename == 'guide.md'
    assert embedded[0].chunk.filename == 'guide.md'
    db.commit.assert_called_once()


def test_api_answers_and_search(client, monkeypatch, match):
    api, db = client
    result = RetrievalResult('What?', (match,), build_context([match]))
    retrieval = Mock()
    retrieval.retrieve.return_value = result
    generator = Mock()
    generator.generate.return_value = AnswerDraft('BM25 and vectors.', True, (1,))
    service = AnswerService(retrieval, generator)
    monkeypatch.setattr(deps, 'services', lambda *args, **kwargs: (retrieval, service))
    response = api.post('/questions', json={'question': 'What?'})
    assert response.status_code == 200
    assert response.json()['citations'][0]['document_id'] == str(match.document_id)
    assert api.post('/search', json={'question': 'What?'}).json()['matches'][0]['text'] == match.text
    assert api.post('/questions', json={'question': 'What?', 'agentic': True}).status_code == 200


def test_missing_document(client):
    api, db = client
    db.get.return_value = None
    assert api.get(f'/documents/{uuid4()}').status_code == 404
    assert api.post('/questions', json={'question': 'x', 'document_id': str(uuid4())}).status_code == 404
    db.execute.return_value.rowcount = 0
    assert api.delete(f'/documents/{uuid4()}').status_code == 404


def test_auth_body_limit_and_metrics(client, monkeypatch):
    api, _ = client
    monkeypatch.setenv('API_KEY', 'test-key')
    assert api.get('/documents').status_code == 401
    assert api.get('/health').status_code == 200
    assert api.get('/metrics', headers={'X-API-Key': 'test-key'}).status_code == 200
    response = api.post('/questions', content=b'x' * 65537, headers={'X-API-Key': 'test-key'})
    assert response.status_code == 413
    assert response.headers['x-request-id']


def test_rate_limit_and_production(client, monkeypatch):
    api, _ = client
    monkeypatch.setenv('RATE_LIMIT_PER_MINUTE', '1')
    assert api.get('/metrics').status_code == 200
    assert api.get('/metrics').status_code == 429
    monkeypatch.setenv('APP_ENV', 'production')
    assert api.get('/documents').status_code == 503


def test_bm25_and_fusion(match):
    unrelated = replace(match, chunk_index=1, text='Gardening flowers')
    assert bm25('BM25', [unrelated, match]) == [match]
    assert bm25('absent', [match]) == []
    assert fuse([match, unrelated], [unrelated])[0] == unrelated
    assert fuse([match, match]) == [match]


def test_hybrid_budget_and_scope(match):
    vector = Mock(model_name='fake')
    vector.retrieve.return_value = RetrievalResult('BM25', (), '')
    repository = Mock()
    repository.corpus.return_value = [match]
    repository.score_candidates.side_effect = lambda matches, vector: matches
    vector.embedding_provider.encode.return_value = [(1., 0.)]
    pipeline = SearchPipeline(vector, repository, hybrid=True, context_chars=180)
    result = pipeline.retrieve('BM25', document_id=match.document_id)
    assert result.has_results
    assert len(result.context) <= 180
    repository.corpus.assert_called_once_with('fake', match.document_id)


def test_rerank_multi_query_neighbors(match):
    other = replace(match, chunk_index=1)
    vector = Mock()
    vector.retrieve.return_value = RetrievalResult('x and y', (match, other), '')
    ranker = Mock()
    ranker.rerank.return_value = [other, match]
    repository = Mock()
    repository.expand_neighbors.return_value = [other, match]
    pipeline = SearchPipeline(vector, repository, reranker=ranker, multi_query=True, neighbors=True)
    result = pipeline.retrieve('x and y', top_k=1)
    assert vector.retrieve.call_count == 3
    assert result.matches[0] == other
    repository.expand_neighbors.assert_called_once()


def test_agent_bounded_retry_and_citations(match):
    retrieval = Mock()
    retrieval.retrieve.side_effect = [RetrievalResult('q', (), ''), RetrievalResult('q', (match,), build_context([match]))]
    generator = Mock()
    generator.generate.return_value = AnswerDraft('evidence', True, (99,))
    service = AgenticAnswerService(retrieval, AnswerService(retrieval, generator))
    result = service.answer('q', document_id=match.document_id, min_similarity=.9)
    assert result.status == 'insufficient_context'
    assert retrieval.retrieve.call_count == 2
    assert retrieval.retrieve.call_args.kwargs['document_id'] == match.document_id
    assert retrieval.retrieve.call_args.kwargs['min_similarity'] is None


def test_evaluation():
    scores = retrieval_metrics(['a', 'a', 'b'], ['b'], k=2)
    assert scores['recall_at_k'] == 1
    assert scores['precision_at_k'] == .5
    assert scores['mrr'] == .5
    assert 0 < scores['ndcg_at_k'] < 1
    assert answer_metrics('RAG retrieves documents.', 'RAG retrieves documents')['token_f1'] == 1
    assert retrieval_metrics([], [], 1)['recall_at_k'] == 0


def test_swagger_has_api_key_authorization(client, monkeypatch):
    api, _ = client
    monkeypatch.setenv('API_KEY', 'test-key')
    assert api.get('/swagger').status_code == 200
    schema = api.get('/json.json').json()
    assert schema['components']['securitySchemes']['APIKeyHeader']['name'] == 'X-API-Key'
    assert schema['paths']['/questions']['post']['security'] == [{'APIKeyHeader': []}]


def test_hybrid_threshold_scores_lexical_candidates(match):
    low = replace(match, chunk_index=1, similarity=.1)
    vector = Mock(model_name='fake')
    vector.embedding_provider.encode.return_value = [(1., 0.)]
    vector.retrieve.return_value = RetrievalResult('BM25', (match,), '')
    repository = Mock()
    repository.corpus.return_value = [match, low]
    repository.score_candidates.return_value = [match, low]
    result = SearchPipeline(vector, repository, hybrid=True).retrieve('BM25', min_similarity=.5)
    assert result.matches == (match,)
    assert result.matches[0].similarity == .8


def test_budget_too_small_returns_no_unlabeled_evidence(match):
    vector = Mock()
    vector.retrieve.return_value = RetrievalResult('q', (match,), '')
    result = SearchPipeline(vector, Mock(), context_chars=1).retrieve('q')
    assert result.matches == ()
    assert result.context == ''


def test_api_dependency_failure_is_sanitized(client, monkeypatch):
    api, _ = client
    def unavailable():
        raise RuntimeError('private-provider-payload')
    monkeypatch.setattr(deps, 'embeddings', unavailable)
    response = api.post('/questions', json={'question': 'q'})
    assert response.status_code == 503
    assert 'private-provider-payload' not in response.text
