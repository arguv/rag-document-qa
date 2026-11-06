
from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class SearchMatch:
    document_id: UUID
    filename: str
    text: str
    page_number: int | None
    heading: str | None
    chunk_index: int
    similarity: float
