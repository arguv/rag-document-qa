
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from app.domain.embeddings import Vector
from app.domain.search import SearchMatch


class ChunkSearchRepository(Protocol):
    def search(
        self,
        query_vector: Vector,
        model_name: str,
        limit: int = 5,
        document_id: UUID | None = None,
    ) -> list[SearchMatch]:
        ...


@dataclass(frozen=True)
class RetrievalResult:
    question: str
    matches: tuple[SearchMatch, ...]
    context: str

    @property
    def has_results(self) -> bool:
        return bool(self.matches)
