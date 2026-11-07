
import math
from typing import Sequence
from uuid import UUID

from app.application.embeddings import embed_query
from app.domain.embeddings import EmbeddingProvider
from app.domain.retrieval import (
    ChunkSearchRepository,
    RetrievalResult,
)
from app.domain.search import SearchMatch


def build_context(
    matches: Sequence[SearchMatch],
) -> str:
    """Format retrieved chunks with source references."""

    sections = []

    for index, match in enumerate(matches, start=1):
        source = (
            f"Source [{index}]\n"
            f"Document ID: {match.document_id}\n"
            f"File: {match.filename}"
        )

        if match.page_number is not None:
            source += f"\nPage: {match.page_number}"

        if match.heading:
            source += f"\nHeading: {match.heading}"

        sections.append(
            f"{source}\n\n{match.text}"
        )

    return "\n\n---\n\n".join(sections)


class RetrievalService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        repository: ChunkSearchRepository,
        model_name: str,
    ) -> None:
        if not model_name.strip():
            raise ValueError(
                "Embedding model name is required"
            )

        self.embedding_provider = embedding_provider
        self.repository = repository
        self.model_name = model_name

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        document_id: UUID | None = None,
        min_similarity: float | None = None,
    ) -> RetrievalResult:
        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty")

        if not 1 <= top_k <= 50:
            raise ValueError(
                "top_k must be between 1 and 50"
            )

        if min_similarity is not None:
            if (
                not math.isfinite(min_similarity)
                or not -1 <= min_similarity <= 1
            ):
                raise ValueError(
                    "min_similarity must be between -1 and 1"
                )

        # Generate the question embedding.
        query_vector = embed_query(
            question,
            self.embedding_provider,
        )

        # Search the persistent vector database.
        candidates = self.repository.search(
            query_vector=query_vector,
            model_name=self.model_name,
            limit=top_k,
            document_id=document_id,
        )

        # Optionally filter low-similarity results.
        matches = tuple(
            match
            for match in candidates
            if (
                min_similarity is None
                or match.similarity >= min_similarity
            )
        )

        # Prepare source-labeled context.
        context = build_context(matches)

        return RetrievalResult(
            question=question,
            matches=matches,
            context=context,
        )
