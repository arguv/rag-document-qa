
from uuid import UUID

from app.application.retrieval import RetrievalService
from app.domain.generation import (
    AnswerCitation,
    AnswerGenerator,
    AnswerResult,
)


INSUFFICIENT_CONTEXT_MESSAGE = (
    "I couldn't find enough information in the "
    "available documents to answer that question."
)


class AnswerService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        generator: AnswerGenerator,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.generator = generator

    def answer(
        self,
        question: str,
        top_k: int = 5,
        document_id: UUID | None = None,
        min_similarity: float | None = None,
    ) -> AnswerResult:

        # Step 6: Retrieve relevant document chunks.
        retrieval = self.retrieval_service.retrieve(
            question=question,
            top_k=top_k,
            document_id=document_id,
            min_similarity=min_similarity,
        )

        # No evidence: do not call the LLM.
        if not retrieval.has_results:
            return self._insufficient(retrieval.question)

        # Step 7: Generate an answer using retrieved context.
        draft = self.generator.generate(
            question=retrieval.question,
            context=retrieval.context,
        )

        if not draft.is_answerable or not draft.answer.strip():
            return self._insufficient(retrieval.question)

        # Require at least one supporting source.
        if not draft.source_numbers:
            return self._insufficient(retrieval.question)

        max_source = len(retrieval.matches)

        # Validate all source references from the LLM.
        for number in draft.source_numbers:
            if type(number) is not int or not 1 <= number <= max_source:
                return self._insufficient(retrieval.question)

        # Remove duplicate citations while retaining order.
        unique_numbers = tuple(
            dict.fromkeys(draft.source_numbers)
        )

        citations = []

        for number in unique_numbers:
            match = retrieval.matches[number - 1]

            citations.append(
                AnswerCitation(
                    source_number=number,
                    document_id=match.document_id,
                    filename=match.filename,
                    chunk_index=match.chunk_index,
                    page_number=match.page_number,
                    heading=match.heading,
                )
            )

        return AnswerResult(
            question=retrieval.question,
            answer=draft.answer.strip(),
            status="answered",
            citations=tuple(citations),
        )

    @staticmethod
    def _insufficient(question: str) -> AnswerResult:
        return AnswerResult(
            question=question,
            answer=INSUFFICIENT_CONTEXT_MESSAGE,
            status="insufficient_context",
            citations=(),
        )
