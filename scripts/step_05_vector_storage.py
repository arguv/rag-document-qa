
from sqlalchemy.orm import Session

from app.application.chunking import (
    ChunkingSettings,
    DocumentAwareChunker,
)
from app.application.embeddings import (
    embed_chunks,
    embed_query,
)
from app.infrastructure.database.repository import (
    PgvectorRepository,
)
from app.infrastructure.database.session import (
    create_database_engine,
)
from app.infrastructure.document_loader import load_document
from app.infrastructure.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddingProvider,
)


def main() -> None:
    document = load_document(
        "examples/embedding_demo.md"
    )

    chunker = DocumentAwareChunker(
        ChunkingSettings(
            size=300,
            overlap=40,
        )
    )

    chunks = chunker.chunk(document)

    provider = SentenceTransformerEmbeddingProvider()

    embedded_chunks = embed_chunks(
        chunks,
        provider,
    )

    question = "How does RAG process documents?"

    query_vector = embed_query(
        question,
        provider,
    )

    engine = create_database_engine()

    try:
        with Session(engine) as session:
            repository = PgvectorRepository(session)

            # Save the document in one transaction.
            with session.begin():
                document_id = repository.save_document(
                    document=document,
                    chunks=embedded_chunks,
                    model_name=provider.model_name,
                )

            # Read it back using vector similarity.
            matches = repository.search(
                query_vector=query_vector,
                model_name=provider.model_name,
                limit=3,
                document_id=document_id,
            )

            print("Stored document ID:", document_id)
            print("Stored chunks:", len(embedded_chunks))
            print("Question:", question)
            print("\nDatabase search results:")

            for match in matches:
                print(
                    f"\nChunk: {match.chunk_index}"
                    f"\nHeading: {match.heading}"
                    f"\nSimilarity: {match.similarity:.4f}"
                    f"\nText: {match.text}"
                )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
