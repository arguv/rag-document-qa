
from dataclasses import dataclass
from typing import Protocol, Sequence

from app.domain.chunks import DocumentChunk


Vector = tuple[float, ...]


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: DocumentChunk
    vector: Vector


class EmbeddingProvider(Protocol):
    def encode(self, texts: Sequence[str]) -> list[Vector]:
        ...
