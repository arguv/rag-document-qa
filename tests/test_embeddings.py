
import math

import pytest

from app.application.embeddings import (
    cosine_similarity,
    embed_chunks,
    embed_query,
)
from app.domain.chunks import DocumentChunk


class FakeEmbeddingProvider:
    def __init__(self, vectors):
        self.vectors = vectors
        self.received = []

    def encode(self, texts):
        self.received.append(list(texts))
        return self.vectors


def make_chunk(
    text="RAG retrieves information",
    index=0,
):
    return DocumentChunk(
        text=text,
        filename="guide.pdf",
        file_type="pdf",
        page_number=3,
        chunk_index=index,
        strategy="recursive",
    )


def test_embed_chunks_preserves_metadata_and_order():
    chunks = [
        make_chunk("First", 0),
        make_chunk("Second", 1),
    ]

    provider = FakeEmbeddingProvider([
        (1.0, 0.0),
        (0.0, 1.0),
    ])

    result = embed_chunks(chunks, provider)

    assert provider.received == [["First", "Second"]]
    assert [item.chunk for item in result] == chunks

    assert [item.vector for item in result] == [
        (1.0, 0.0),
        (0.0, 1.0),
    ]

    assert result[0].chunk.page_number == 3


def test_empty_chunks_do_not_call_provider():
    provider = FakeEmbeddingProvider([])

    assert embed_chunks([], provider) == []
    assert provider.received == []


def test_empty_chunk_text_is_rejected():
    with pytest.raises(
        ValueError, match="empty chunk"
    ):
        embed_chunks(
            [make_chunk("  ")],
            FakeEmbeddingProvider([(1.0, 0.0)]),
        )


def test_wrong_embedding_count_is_rejected():
    with pytest.raises(ValueError, match="count"):
        embed_chunks(
            [make_chunk()],
            FakeEmbeddingProvider([]),
        )


def test_inconsistent_embedding_dimensions_are_rejected():
    with pytest.raises(ValueError, match="dimensions"):
        embed_chunks(
            [
                make_chunk("First"),
                make_chunk("Second", 1),
            ],
            FakeEmbeddingProvider([
                (1.0, 0.0),
                (1.0,),
            ]),
        )


def test_empty_embedding_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        embed_chunks(
            [make_chunk()],
            FakeEmbeddingProvider([()]),
        )


def test_nonfinite_embedding_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        embed_chunks(
            [make_chunk()],
            FakeEmbeddingProvider([
                (float("nan"), 0.0),
            ]),
        )


def test_embed_query_returns_vector():
    provider = FakeEmbeddingProvider([
        (0.0, 1.0),
    ])

    result = embed_query("What is RAG?", provider)

    assert result == (0.0, 1.0)
    assert provider.received == [["What is RAG?"]]


def test_empty_query_is_rejected():
    with pytest.raises(
        ValueError, match="Query cannot be empty"
    ):
        embed_query(
            "  ",
            FakeEmbeddingProvider([]),
        )


def test_cosine_similarity_identical():
    assert cosine_similarity(
        [1.0, 2.0],
        [1.0, 2.0],
    ) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    assert cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    ) == pytest.approx(0.0)


def test_cosine_similarity_opposite():
    assert cosine_similarity(
        [1.0, 0.0],
        [-1.0, 0.0],
    ) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "left,right",
    [
        ([], []),
        ([1.0], [1.0, 2.0]),
        ([0.0, 0.0], [1.0, 0.0]),
        ([math.inf], [1.0]),
    ],
)
def test_cosine_similarity_invalid_vectors(left, right):
    with pytest.raises(ValueError):
        cosine_similarity(left, right)
