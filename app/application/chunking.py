
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from app.domain.chunks import DocumentChunk
from app.domain.documents import ParsedDocument


@dataclass(frozen=True)
class ChunkingSettings:
    size: int = 500
    overlap: int = 50

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be greater than zero")

        if not 0 <= self.overlap < self.size:
            raise ValueError(
                "overlap must be >= 0 and smaller than size"
            )


class Chunker(Protocol):
    def chunk(
        self, document: ParsedDocument
    ) -> list[DocumentChunk]: ...


class FixedSizeChunker:
    def __init__(self, settings: ChunkingSettings) -> None:
        self.settings = settings

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        return _make_chunks(
            document,
            "fixed",
            _document_units(document),
            _split_fixed,
            self.settings,
        )


class RecursiveChunker:
    def __init__(self, settings: ChunkingSettings) -> None:
        self.settings = settings

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        return _make_chunks(
            document,
            "recursive",
            _document_units(document),
            _split_recursive,
            self.settings,
        )


class DocumentAwareChunker:
    def __init__(self, settings: ChunkingSettings) -> None:
        self.settings = settings

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        def units() -> Iterable[
            tuple[str, int | None, str | None]
        ]:
            for section in document.sections:
                if document.file_type == "md":
                    for text, heading in _markdown_sections(
                        section.text
                    ):
                        yield text, section.page_number, heading
                else:
                    yield section.text, section.page_number, None

        return _make_chunks(
            document,
            "document_aware",
            units(),
            _split_recursive,
            self.settings,
        )


def _document_units(
    document: ParsedDocument,
) -> Iterable[tuple[str, int | None, str | None]]:
    for section in document.sections:
        yield section.text, section.page_number, None


def _make_chunks(
    document: ParsedDocument,
    strategy: str,
    units: Iterable[tuple[str, int | None, str | None]],
    splitter: Callable[
        [str, ChunkingSettings], Iterable[str]
    ],
    settings: ChunkingSettings,
) -> list[DocumentChunk]:
    result = []

    for unit_text, page_number, heading in units:
        for text in splitter(unit_text, settings):
            result.append(
                DocumentChunk(
                    text=text,
                    filename=document.filename,
                    file_type=document.file_type,
                    page_number=page_number,
                    chunk_index=len(result),
                    strategy=strategy,
                    heading=heading,
                )
            )

    return result


def _split_fixed(
    text: str,
    settings: ChunkingSettings,
) -> Iterable[str]:
    start = 0

    while start < len(text):
        end = min(start + settings.size, len(text))

        piece = text[start:end].strip()
        if piece:
            yield piece

        if end == len(text):
            break

        start = end - settings.overlap


def _split_recursive(
    text: str,
    settings: ChunkingSettings,
) -> Iterable[str]:
    start = 0

    while start < len(text):
        max_end = min(start + settings.size, len(text))
        end = max_end

        if max_end < len(text):
            end = _choose_boundary(
                text,
                start,
                max_end,
                settings,
                ("\n\n", "\n", ". ", " "),
            )

        piece = text[start:end].strip()
        if piece:
            yield piece

        if end == len(text):
            break

        start = end - settings.overlap


def _choose_boundary(
    text: str,
    start: int,
    max_end: int,
    settings: ChunkingSettings,
    separators: tuple[str, ...],
) -> int:
    if not separators:
        return max_end

    minimum = start + max(
        settings.overlap + 1,
        settings.size // 2,
    )

    separator = separators[0]
    position = text.rfind(separator, minimum, max_end)

    if position >= 0:
        end = position + len(separator)

        if end > start + settings.overlap:
            return end

    return _choose_boundary(
        text,
        start,
        max_end,
        settings,
        separators[1:],
    )


def _markdown_sections(
    text: str,
) -> Iterable[tuple[str, str | None]]:
    lines: list[str] = []
    heading: str | None = None

    for line in text.splitlines(keepends=True):
        match = re.match(
            r"^ {0,3}#{1,6}[ \t]+(.+?)\s*$",
            line,
        )

        if match:
            if lines:
                yield "".join(lines), heading

            lines = [line]
            heading = match.group(1).strip()
        else:
            lines.append(line)

    if lines:
        yield "".join(lines), heading
