
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

    def corpus(self, model_name, document_id=None):
        """Exact BM25 corpus. Fail explicitly rather than silently truncate rankings."""
        import os
        statement = select(StoredChunk, StoredDocument.filename).join(StoredDocument).where(
            StoredDocument.embedding_model == model_name)
        if document_id is not None:
            statement = statement.where(StoredChunk.document_id == document_id)
        cap = int(os.getenv('BM25_MAX_CHUNKS', '20000'))
        rows = self.session.execute(statement.order_by(StoredChunk.id).limit(cap + 1)).all()
        if len(rows) > cap:
            raise RuntimeError('BM25 corpus limit exceeded; use document filtering or a dedicated search backend')
        return [SearchMatch(chunk.document_id, filename, chunk.text, chunk.page_number,
                            chunk.heading, chunk.chunk_index, 0.0) for chunk, filename in rows]

    def score_candidates(self, matches, query_vector):
        from dataclasses import replace
        from app.application.embeddings import cosine_similarity
        if not matches:
            return []
        from sqlalchemy import tuple_
        keys = [(item.document_id, item.chunk_index) for item in matches]
        rows = self.session.execute(select(StoredChunk).where(
            tuple_(StoredChunk.document_id, StoredChunk.chunk_index).in_(keys))).scalars()
        vectors = {(row.document_id, row.chunk_index): row.embedding for row in rows}
        return [replace(item, similarity=cosine_similarity(query_vector, vectors[(item.document_id, item.chunk_index)])) for item in matches]

    def expand_neighbors(self, matches, model_name):
        from sqlalchemy import or_, and_
        clauses = [and_(StoredChunk.document_id == m.document_id,
                        StoredChunk.chunk_index.between(max(0, m.chunk_index - 1), m.chunk_index + 1)) for m in matches]
        rows = self.session.execute(select(StoredChunk, StoredDocument.filename).join(StoredDocument).where(
            StoredDocument.embedding_model == model_name, or_(*clauses))).all()
        by_key = {(chunk.document_id, chunk.chunk_index): SearchMatch(chunk.document_id, filename, chunk.text,
                   chunk.page_number, chunk.heading, chunk.chunk_index, 0.0) for chunk, filename in rows}
        expanded, seen = [], set()
        for match in matches:
            # Evidence hits precede their neighbors when the context budget is tight.
            for key in [(match.document_id, match.chunk_index), (match.document_id, match.chunk_index - 1),
                        (match.document_id, match.chunk_index + 1)]:
                if key in by_key and key not in seen:
                    expanded.append(match if key == (match.document_id, match.chunk_index) else by_key[key])
                    seen.add(key)
        return expanded
