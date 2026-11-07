
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID


@dataclass(frozen=True)
class AnswerDraft:
    answer: str
    is_answerable: bool
    source_numbers: tuple[int, ...]


class AnswerGenerator(Protocol):
    def generate(
        self,
        question: str,
        context: str,
    ) -> AnswerDraft:
        ...


@dataclass(frozen=True)
class AnswerCitation:
    source_number: int
    document_id: UUID
    filename: str
    chunk_index: int
    page_number: int | None
    heading: str | None


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    status: Literal["answered", "insufficient_context"]
    citations: tuple[AnswerCitation, ...]
