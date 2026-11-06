
import math
from typing import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.documents import ParsedDocument
from app.domain.embeddings import EmbeddedChunk, Vector
from app.domain.search import SearchMatch
from app.infrastructure.database.models import (
    StoredChunk,
    StoredDocument,
)


EMBEDDING_DIMENSIONS = 384


def _validate_vector(vector: Sequence[float]) -> None:
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSIONS} dimensions"
        )

    if not all(math.isfinite(value) for value in vector):
        raise ValueError(
            "Vector must contain finite numbers"
        )

    if not any(value != 0 for value in vector):
        raise ValueError("Zero vectors are not supported")


class PgvectorRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def save_document(
        self,
        document: ParsedDocument,
        chunks: Sequence[EmbeddedChunk],
        model_name: str,
    ) -> UUID:
        if not model_name.strip():
            raise ValueError("Embedding model name is required")

        if not chunks:
            raise ValueError("Cannot store a document without chunks")

        for item in chunks:
            _validate_vector(item.vector)

            if (
                item.chunk.filename != document.filename
                or item.chunk.file_type != document.file_type
            ):
                raise ValueError(
                    "Chunk source does not match document"
                )

        stored_document = StoredDocument(
            filename=document.filename,
            file_type=document.file_type,
            embedding_model=model_name,
        )

        self.session.add(stored_document)
        self.session.flush()

        records = [
            StoredChunk(
                document_id=stored_document.id,
                chunk_index=item.chunk.chunk_index,
                text=item.chunk.text,
                page_number=item.chunk.page_number,
                heading=item.chunk.heading,
                strategy=item.chunk.strategy,
                embedding=list(item.vector),
            )
            for item in chunks
        ]

        self.session.add_all(records)
        self.session.flush()

        return stored_document.id

    def search(
        self,
        query_vector: Vector,
        model_name: str,
        limit: int = 5,
        document_id: UUID | None = None,
    ) -> list[SearchMatch]:
        _validate_vector(query_vector)

        if not model_name.strip():
            raise ValueError("Embedding model name is required")

        if limit <= 0:
            raise ValueError("limit must be positive")

        distance = StoredChunk.embedding.cosine_distance(
            list(query_vector)
        )

        statement = (
            select(
                StoredChunk.document_id,
                StoredDocument.filename,
                StoredChunk.text,
                StoredChunk.page_number,
                StoredChunk.heading,
                StoredChunk.chunk_index,
                (1 - distance).label("similarity"),
            )
            .join(
                StoredDocument,
                StoredDocument.id == StoredChunk.document_id,
            )
            .where(
                StoredDocument.embedding_model == model_name
            )
        )

        if document_id is not None:
            statement = statement.where(
                StoredChunk.document_id == document_id
            )

        statement = (
            statement
            .order_by(distance, StoredChunk.id)
            .limit(limit)
        )

        rows = self.session.execute(statement).all()

        return [
            SearchMatch(
                document_id=row.document_id,
                filename=row.filename,
                text=row.text,
                page_number=row.page_number,
                heading=row.heading,
                chunk_index=row.chunk_index,
                similarity=float(row.similarity),
            )
            for row in rows
        ]
