
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentSection:
    text: str
    page_number: int | None = None


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    file_type: str
    sections: tuple[DocumentSection, ...]
    page_count: int | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(
            section.text for section in self.sections
        )
