
import math
from uuid import UUID

import pytest

from app.application.retrieval import RetrievalService
from app.domain.search import SearchMatch


DOCUMENT_ID = UUID(
    "00000000-0000-0000-0000-000000000001"
)
OTHER_DOCUMENT_ID = UUID(
    "00000000-0000-0000-0000-000000000002"
)


class FakeEmbeddingProvider:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [
            (1.0, 0.0)
            for _ in texts
        ]


class FakeSearchRepository:
    def __init__(self, matches=None):
        self.matches = matches or []
        self.calls = []

    def search(
        self,
        query_vector,
        model_name,
        limit=5,
        document_id=None,
    ):
        self.calls.append({
            "query_vector": query_vector,
            "model_name": model_name,
            "limit": limit,
            "document_id": document_id,
        })

        matches = self.matches

        if document_id is not None:
            matches = [
                match
                for match in matches
                if match.document_id == document_id
            ]

        return matches[:limit]


def make_match(
    text="RAG processes documents",
    similarity=0.9,
    document_id=DOCUMENT_ID,
    page_number=2,
    heading="Document Processing",
):
    return SearchMatch(
        document_id=document_id,
        filename="guide.pdf",
        text=text,
        page_number=page_number,
        heading=heading,
        chunk_index=0,
        similarity=similarity,
    )


def make_service(matches=None):
    provider = FakeEmbeddingProvider()
    repository = FakeSearchRepository(matches)

    service = RetrievalService(
        embedding_provider=provider,
        repository=repository,
        model_name="test-model",
    )

    return service, provider, repository


def test_retrieval_generates_query_and_context():
    matches = [
        make_match("First result", 0.9),
        make_match("Second result", 0.7),
    ]

    service, provider, repository = make_service(matches)

    result = service.retrieve(
        "  How does RAG work?  ",
        top_k=2,
    )

    assert result.question == "How does RAG work?"
    assert result.has_results is True
    assert len(result.matches) == 2

    assert provider.calls == [
        ["How does RAG work?"]
    ]

    assert repository.calls[0] == {
        "query_vector": (1.0, 0.0),
        "model_name": "test-model",
        "limit": 2,
        "document_id": None,
    }

    assert "Source [1]" in result.context
    assert "Document ID:" in result.context
    assert "File: guide.pdf" in result.context
    assert "Page: 2" in result.context
    assert "Heading: Document Processing" in result.context

    assert result.context.index(
        "First result"
    ) < result.context.index("Second result")


def test_min_similarity_filters_results():
    service, _, _ = make_service([
        make_match("Relevant", 0.85),
        make_match("Irrelevant", 0.10),
    ])

    result = service.retrieve(
        "What is RAG?",
        min_similarity=0.5,
    )

    assert len(result.matches) == 1
    assert result.matches[0].text == "Relevant"
    assert "Irrelevant" not in result.context


def test_no_matches_returns_empty_context():
    service, _, _ = make_service([])

    result = service.retrieve("Unknown question")

    assert result.has_results is False
    assert result.matches == ()
    assert result.context == ""


def test_document_filter_is_forwarded():
    service, _, repository = make_service([
        make_match("Document A", 0.9),
        make_match(
            "Document B",
            0.8,
            document_id=OTHER_DOCUMENT_ID,
        ),
    ])

    result = service.retrieve(
        "Question",
        document_id=OTHER_DOCUMENT_ID,
    )

    assert len(result.matches) == 1
    assert result.matches[0].text == "Document B"

    assert (
        repository.calls[0]["document_id"]
        == OTHER_DOCUMENT_ID
    )


def test_empty_question_is_rejected():
    service, provider, repository = make_service()

    with pytest.raises(ValueError, match="Question"):
        service.retrieve("   ")

    assert provider.calls == []
    assert repository.calls == []


@pytest.mark.parametrize(
    "top_k",
    [0, -1, 51],
)
def test_invalid_top_k_is_rejected(top_k):
    service, _, _ = make_service()

    with pytest.raises(ValueError, match="top_k"):
        service.retrieve(
            "Question",
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "threshold",
    [-1.1, 1.1, math.nan],
)
def test_invalid_similarity_threshold(threshold):
    service, _, _ = make_service()

    with pytest.raises(
        ValueError,
        match="min_similarity",
    ):
        service.retrieve(
            "Question",
            min_similarity=threshold,
        )


def test_context_without_optional_metadata():
    service, _, _ = make_service([
        make_match(
            text="Plain text content",
            page_number=None,
            heading=None,
        )
    ])

    result = service.retrieve("Question")

    assert "File: guide.pdf" in result.context
    assert "Plain text content" in result.context
    assert "\nPage:" not in result.context
    assert "\nHeading:" not in result.context
