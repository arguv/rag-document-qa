
from uuid import UUID

import pytest

from app.application.answering import (
    AnswerService,
    INSUFFICIENT_CONTEXT_MESSAGE,
)
from app.application.retrieval import build_context
from app.domain.generation import AnswerDraft
from app.domain.retrieval import RetrievalResult
from app.domain.search import SearchMatch


DOCUMENT_ID = UUID(
    "00000000-0000-0000-0000-000000000001"
)


def make_match(
    text="RAG splits documents into chunks.",
    filename="guide.pdf",
    page_number=3,
    heading="RAG",
    chunk_index=0,
) -> SearchMatch:
    return SearchMatch(
        document_id=DOCUMENT_ID,
        filename=filename,
        text=text,
        page_number=page_number,
        heading=heading,
        chunk_index=chunk_index,
        similarity=0.9,
    )


class FakeRetrievalService:
    def __init__(self, matches):
        self.matches = matches
        self.calls = []

    def retrieve(
        self,
        question,
        top_k=5,
        document_id=None,
        min_similarity=None,
    ):
        self.calls.append({
            "question": question,
            "top_k": top_k,
            "document_id": document_id,
            "min_similarity": min_similarity,
        })

        matches = [
            match for match in self.matches
            if (
                document_id is None
                or match.document_id == document_id
            )
            and (
                min_similarity is None
                or match.similarity >= min_similarity
            )
        ][:top_k]

        return RetrievalResult(
            question=question.strip(),
            matches=tuple(matches),
            context=build_context(matches),
        )


class FakeAnswerGenerator:
    def __init__(self, draft):
        self.draft = draft
        self.calls = []

    def generate(self, question, context):
        self.calls.append((question, context))
        return self.draft


def make_service(matches, draft):
    retrieval = FakeRetrievalService(matches)
    generator = FakeAnswerGenerator(draft)

    return AnswerService(retrieval, generator), retrieval, generator


def test_answer_with_valid_citation():
    draft = AnswerDraft(
        answer="RAG splits documents into chunks.",
        is_answerable=True,
        source_numbers=(1,),
    )

    service, _, generator = make_service(
        [make_match()],
        draft,
    )

    result = service.answer("How does RAG work?")

    assert result.status == "answered"
    assert result.answer == draft.answer
    assert len(result.citations) == 1
    assert result.citations[0].source_number == 1
    assert result.citations[0].document_id == DOCUMENT_ID
    assert result.citations[0].filename == "guide.pdf"
    assert result.citations[0].page_number == 3
    assert result.citations[0].heading == "RAG"

    assert len(generator.calls) == 1
    assert "Source [1]" in generator.calls[0][1]


def test_duplicate_citations_are_removed():
    matches = [
        make_match("First", chunk_index=0),
        make_match("Second", chunk_index=1),
    ]

    draft = AnswerDraft(
        answer="Combined answer.",
        is_answerable=True,
        source_numbers=(2, 1, 2),
    )

    service, _, _ = make_service(matches, draft)

    result = service.answer("Question")

    assert [
        citation.source_number
        for citation in result.citations
    ] == [2, 1]


def test_no_retrieval_results_skips_llm():
    draft = AnswerDraft(
        answer="Should not be used",
        is_answerable=True,
        source_numbers=(1,),
    )

    service, _, generator = make_service([], draft)

    result = service.answer("Unknown question")

    assert result.status == "insufficient_context"
    assert result.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert result.citations == ()
    assert generator.calls == []


def test_llm_reports_insufficient_evidence():
    draft = AnswerDraft(
        answer="",
        is_answerable=False,
        source_numbers=(),
    )

    service, _, _ = make_service(
        [make_match()],
        draft,
    )

    result = service.answer("Unsupported question")

    assert result.status == "insufficient_context"
    assert result.citations == ()


def test_blank_answer_is_rejected():
    draft = AnswerDraft(
        answer="   ",
        is_answerable=True,
        source_numbers=(1,),
    )

    service, _, _ = make_service(
        [make_match()],
        draft,
    )

    result = service.answer("Question")

    assert result.status == "insufficient_context"


def test_missing_citations_are_rejected():
    draft = AnswerDraft(
        answer="An answer without evidence.",
        is_answerable=True,
        source_numbers=(),
    )

    service, _, _ = make_service(
        [make_match()],
        draft,
    )

    result = service.answer("Question")

    assert result.status == "insufficient_context"


@pytest.mark.parametrize(
    "invalid_number",
    [0, 3, -1, True],
)
def test_invalid_citation_numbers_are_rejected(
    invalid_number,
):
    draft = AnswerDraft(
        answer="Unsupported citation.",
        is_answerable=True,
        source_numbers=(invalid_number,),
    )

    service, _, _ = make_service(
        [make_match()],
        draft,
    )

    result = service.answer("Question")

    assert result.status == "insufficient_context"
    assert result.citations == ()


def test_retrieval_parameters_are_forwarded():
    draft = AnswerDraft(
        answer="A grounded answer.",
        is_answerable=True,
        source_numbers=(1,),
    )

    service, retrieval, _ = make_service(
        [make_match()],
        draft,
    )

    service.answer(
        question="Question",
        top_k=3,
        document_id=DOCUMENT_ID,
        min_similarity=0.5,
    )

    assert retrieval.calls == [{
        "question": "Question",
        "top_k": 3,
        "document_id": DOCUMENT_ID,
        "min_similarity": 0.5,
    }]


def test_llm_failure_is_not_treated_as_missing_evidence():
    class FailingGenerator:
        def generate(self, question, context):
            raise RuntimeError("LLM unavailable")

    retrieval = FakeRetrievalService([make_match()])

    service = AnswerService(
        retrieval_service=retrieval,
        generator=FailingGenerator(),
    )

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        service.answer("Question")
