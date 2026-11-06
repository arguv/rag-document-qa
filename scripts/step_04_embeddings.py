
from app.application.chunking import (
    ChunkingSettings,
    DocumentAwareChunker,
)
from app.application.embeddings import (
    cosine_similarity,
    embed_chunks,
    embed_query,
)
from app.infrastructure.document_loader import load_document
from app.infrastructure.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddingProvider,
)


def main() -> None:
    # Step 2: Load document
    document = load_document(
        "examples/embedding_demo.md"
    )

    # Step 3: Create document chunks
    chunker = DocumentAwareChunker(
        ChunkingSettings(
            size=300,
            overlap=40,
        )
    )

    chunks = chunker.chunk(document)

    # Step 4: Generate embeddings
    provider = SentenceTransformerEmbeddingProvider()

    embedded_chunks = embed_chunks(
        chunks,
        provider,
    )

    question = "How does RAG process documents?"

    question_vector = embed_query(
        question,
        provider,
    )

    # Compare the question with all document chunks
    ranked = sorted(
        embedded_chunks,
        key=lambda item: cosine_similarity(
            question_vector,
            item.vector,
        ),
        reverse=True,
    )

    print("Document:", document.filename)
    print("Total chunks:", len(chunks))
    print(
        "Embedding dimensions:",
        len(question_vector),
    )

    print("\nQuestion:", question)

    for item in ranked:
        score = cosine_similarity(
            question_vector,
            item.vector,
        )

        print(
            f"\nHeading: {item.chunk.heading}"
            f"\nSimilarity: {score:.4f}"
            f"\nText: {item.chunk.text}"
        )


if __name__ == "__main__":
    main()
