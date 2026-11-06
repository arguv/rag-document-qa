
import math
from typing import Sequence

from app.domain.chunks import DocumentChunk
from app.domain.embeddings import (
    EmbeddedChunk,
    EmbeddingProvider,
    Vector,
)


def _validate_vectors(
    vectors: list[Vector],
    expected_count: int,
) -> list[Vector]:
    if len(vectors) != expected_count:
        raise ValueError(
            "Embedding count does not match input count"
        )

    if not vectors:
        return []

    dimension = len(vectors[0])

    if dimension == 0:
        raise ValueError("Embeddings cannot be empty")

    for vector in vectors:
        if len(vector) != dimension:
            raise ValueError(
                "Embedding dimensions must match"
            )

        if not all(math.isfinite(value) for value in vector):
            raise ValueError(
                "Embeddings must contain finite numbers"
            )

    return vectors


def embed_chunks(
    chunks: Sequence[DocumentChunk],
    provider: EmbeddingProvider,
) -> list[EmbeddedChunk]:
    if not chunks:
        return []

    if any(not chunk.text.strip() for chunk in chunks):
        raise ValueError("Cannot embed an empty chunk")

    texts = [chunk.text for chunk in chunks]

    vectors = _validate_vectors(
        provider.encode(texts),
        len(chunks),
    )

    return [
        EmbeddedChunk(chunk=chunk, vector=vector)
        for chunk, vector in zip(chunks, vectors)
    ]


def embed_query(
    query: str,
    provider: EmbeddingProvider,
) -> Vector:
    if not query.strip():
        raise ValueError("Query cannot be empty")

    return _validate_vectors(
        provider.encode([query]),
        1,
    )[0]


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if not left or len(left) != len(right):
        raise ValueError(
            "Vectors must be nonempty and have equal dimensions"
        )

    if not all(
        math.isfinite(value)
        for value in (*left, *right)
    ):
        raise ValueError(
            "Vectors must contain finite numbers"
        )

    dot_product = math.fsum(
        a * b for a, b in zip(left, right)
    )

    left_norm = math.sqrt(
        math.fsum(a * a for a in left)
    )

    right_norm = math.sqrt(
        math.fsum(b * b for b in right)
    )

    if left_norm == 0 or right_norm == 0:
        raise ValueError(
            "Cannot compare zero-length vectors"
        )

    return dot_product / (left_norm * right_norm)
