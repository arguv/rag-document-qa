
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.domain.documents import (
    DocumentSection,
    ParsedDocument,
)


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class DocumentLoadError(ValueError):
    """Raised when a document cannot be loaded."""


def load_document(file_path: str | Path) -> ParsedDocument:
    path = Path(file_path)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported document type: {path.suffix}"
        )

    if not path.is_file():
        raise DocumentLoadError(
            f"File not found: {path}"
        )

    if path.suffix.lower() == ".pdf":
        sections, page_count = _load_pdf(path)
    else:
        sections = _load_text(path)
        page_count = None

    if not sections:
        raise DocumentLoadError(
            f"No extractable text found in: {path.name}"
        )

    return ParsedDocument(
        filename=path.name,
        file_type=path.suffix.lower().lstrip("."),
        sections=tuple(sections),
        page_count=page_count,
    )


def _load_text(path: Path) -> list[DocumentSection]:
    try:
        content = path.read_text(
            encoding="utf-8"
        ).strip()
    except (OSError, UnicodeError) as exc:
        raise DocumentLoadError(
            f"Unable to read text document: {path.name}"
        ) from exc

    if not content:
        return []

    return [DocumentSection(text=content)]


def _load_pdf(
    path: Path,
) -> tuple[list[DocumentSection], int]:
    sections = []

    try:
        with path.open("rb") as file:
            reader = PdfReader(file)

            if reader.is_encrypted:
                raise DocumentLoadError(
                    "Encrypted PDFs are not supported yet."
                )

            page_count = len(reader.pages)

            for page_number, page in enumerate(
                reader.pages, start=1
            ):
                text = (page.extract_text() or "").strip()

                if text:
                    sections.append(
                        DocumentSection(
                            text=text,
                            page_number=page_number,
                        )
                    )

    except (PdfReadError, OSError) as exc:
        raise DocumentLoadError(
            f"Unable to parse PDF: {path.name}"
        ) from exc

    return sections, page_count
