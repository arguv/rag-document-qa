import os
from functools import lru_cache
from sqlalchemy.orm import Session
from app.infrastructure.database.session import create_database_engine
from app.infrastructure.database.repository import PgvectorRepository
from app.application.retrieval import RetrievalService
from app.application.answering import AnswerService
from app.application.search_pipeline import SearchPipeline, CrossEncoderReranker
from app.domain.generation import AnswerDraft


@lru_cache(maxsize=1)
def engine():
    return create_database_engine()


def session():
    with Session(engine()) as db:
        yield db


@lru_cache(maxsize=1)
def embeddings():
    from app.infrastructure.sentence_transformer_embeddings import SentenceTransformerEmbeddingProvider
    return SentenceTransformerEmbeddingProvider(os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))


@lru_cache(maxsize=1)
def reranker():
    name = os.getenv('RERANKER_MODEL', '')
    return CrossEncoderReranker(name) if name else None


class MockGenerator:
    def generate(self, question, context):
        # Deliberately unmistakable: this is a wiring demonstration, not an LLM answer.
        return AnswerDraft('Mock response: retrieved evidence is available. Configure LLM_BACKEND=openai for generated answers.', True, (1,))


@lru_cache(maxsize=1)
def generator():
    backend = os.getenv('LLM_BACKEND', 'mock')
    if backend == 'mock':
        return MockGenerator()
    if backend != 'openai':
        raise RuntimeError('LLM_BACKEND must be mock or openai')
    from app.infrastructure.openai_answer_generator import OpenAIAnswerGenerator
    return OpenAIAnswerGenerator(os.getenv('OPENAI_MODEL', 'gpt-6-luna'))


def services(db, options, *, with_generator=True):
    repository = PgvectorRepository(db)
    provider = embeddings()
    vector = RetrievalService(provider, repository, provider.model_name)
    retrieval = SearchPipeline(vector, repository, hybrid=options.search_mode == 'hybrid',
                               reranker=reranker() if options.rerank else None,
                               multi_query=options.multi_query, neighbors=options.neighbors)
    if options.rerank and retrieval.reranker is None:
        from fastapi import HTTPException
        raise HTTPException(422, 'Set RERANKER_MODEL to enable reranking')
    return retrieval, AnswerService(retrieval, generator()) if with_generator else None
