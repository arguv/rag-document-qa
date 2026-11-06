
import os

import pytest

from dotenv import load_dotenv
from sqlalchemy import create_engine, delete
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.domain.chunks import DocumentChunk
from app.domain.documents import (
    DocumentSection,
    ParsedDocument,
)
from app.domain.embeddings import EmbeddedChunk
from app.infrastructure.database.models import StoredDocument
from app.infrastructure.database.repository import (
    PgvectorRepository,
)


@pytest.mark.integration
def test_store_and_search_vectors():
    load_dotenv()

    database_url = os.getenv("TEST_DATABASE_URL")

    if not database_url:
        pytest.skip("TEST_DATABASE_URL not configured")

    # Prevent accidental writes to the development database.
    if make_url(database_url).database != "rag_documents_test":
        raise RuntimeError("Unsafe test database configuration")

    engine = create_engine(database_url)

    first_vector = (1.0,) + (0.0,) * 383
    second_vector = (0.0, 1.0) + (0.0,) * 382

    document = ParsedDocument(
        filename="integration_test.txt",
        file_type="txt",
        sections=(
            DocumentSection(
                text="First section. Second section."
            ),
        ),
    )

    chunks = [
        EmbeddedChunk(
            chunk=DocumentChunk(
                text="First section",
                filename=document.filename,
                file_type="txt",
                page_number=None,
                chunk_index=0,
                strategy="fixed",
            ),
            vector=first_vector,
        ),
        EmbeddedChunk(
            chunk=DocumentChunk(
                text="Second section",
                filename=document.filename,
                file_type="txt",
                page_number=None,
                chunk_index=1,
                strategy="fixed",
            ),
            vector=second_vector,
        ),
    ]

    document_id = None

    try:
        with Session(engine) as session:
            repository = PgvectorRepository(session)

            with session.begin():
                document_id = repository.save_document(
                    document,
                    chunks,
                    model_name="test-embedding-model",
                )

        # A new session proves the data was committed.
        with Session(engine) as session:
            repository = PgvectorRepository(session)

            matches = repository.search(
                query_vector=first_vector,
                model_name="test-embedding-model",
                document_id=document_id,
                limit=2,
            )

            assert len(matches) == 2
            assert matches[0].text == "First section"
            assert matches[0].similarity == pytest.approx(1.0)
            assert matches[0].filename == "integration_test.txt"

    finally:
        if document_id is not None:
            with Session(engine) as session:
                session.execute(
                    delete(StoredDocument).where(
                        StoredDocument.id == document_id
                    )
                )
                session.commit()

        engine.dispose()
