
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    filename: str
    file_type: str
    page_number: int | None
    chunk_index: int
    strategy: str
    heading: str | None = None
