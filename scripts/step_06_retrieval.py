
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.application.retrieval import RetrievalService
from app.infrastructure.database.models import StoredDocument
from app.infrastructure.database.repository import (
    PgvectorRepository,
)
from app.infrastructure.database.session import (
    create_database_engine,
)
from app.infrastructure.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddingProvider,
)


def main() -> None:
    question = "How does RAG process documents?"

    provider = SentenceTransformerEmbeddingProvider()

    engine = create_database_engine()

    try:
        with Session(engine) as session:
            # Find the sample document saved in Step 5.
            statement = (
                select(StoredDocument.id)
                .where(
                    StoredDocument.filename
                    == "embedding_demo.md",
                    StoredDocument.embedding_model
                    == provider.model_name,
                )
                .order_by(
                    StoredDocument.created_at.desc(),
                    StoredDocument.id.desc(),
                )
                .limit(1)
            )

            document_id = session.scalar(statement)

            if document_id is None:
                raise SystemExit(
                    "Sample document not found. "
                    "Run Step 5 storage demonstration first."
                )

            repository = PgvectorRepository(session)

            service = RetrievalService(
                embedding_provider=provider,
                repository=repository,
                model_name=provider.model_name,
            )

            result = service.retrieve(
                question=question,
                top_k=3,
                document_id=document_id,
                min_similarity=0.20,
            )

            print("Question:", result.question)
            print("Document ID:", document_id)
            print("Retrieved chunks:", len(result.matches))

            for index, match in enumerate(
                result.matches,
                start=1,
            ):
                print(
                    f"\nRank: {index}"
                    f"\nHeading: {match.heading}"
                    f"\nSimilarity: {match.similarity:.4f}"
                )

            print("\nPrepared LLM context:")
            print(result.context or "(No relevant context)")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
