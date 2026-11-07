
import os

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.application.answering import AnswerService
from app.application.retrieval import RetrievalService
from app.infrastructure.database.models import StoredDocument
from app.infrastructure.database.repository import PgvectorRepository
from app.infrastructure.database.session import (
    create_database_engine,
)
from app.infrastructure.openai_answer_generator import (
    OpenAIAnswerGenerator,
)
from app.infrastructure.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddingProvider,
)


def main() -> None:
    load_dotenv()

    question = "How does RAG process documents?"

    embedding_provider = (
        SentenceTransformerEmbeddingProvider()
    )

    generator = OpenAIAnswerGenerator(
        model=os.getenv("OPENAI_MODEL", "gpt-6-luna")
    )

    engine = create_database_engine()

    try:
        with Session(engine) as session:
            # Find the document saved in Step 5.
            statement = (
                select(StoredDocument.id)
                .where(
                    StoredDocument.filename
                    == "embedding_demo.md",
                    StoredDocument.embedding_model
                    == embedding_provider.model_name,
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
                    "Run the Step 5 storage demo first."
                )

            repository = PgvectorRepository(session)

            retrieval_service = RetrievalService(
                embedding_provider=embedding_provider,
                repository=repository,
                model_name=embedding_provider.model_name,
            )

            answer_service = AnswerService(
                retrieval_service=retrieval_service,
                generator=generator,
            )

            result = answer_service.answer(
                question=question,
                top_k=3,
                document_id=document_id,
                min_similarity=0.20,
            )

            print("Question:", result.question)
            print("Status:", result.status)
            print("\nAnswer:")
            print(result.answer)

            print("\nCitations:")

            for citation in result.citations:
                print(
                    f"[{citation.source_number}] "
                    f"{citation.filename}"
                    f" | Page: {citation.page_number}"
                    f" | Heading: {citation.heading}"
                )

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
