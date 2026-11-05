
import pytest

from app.application.chunking import (
    ChunkingSettings,
    DocumentAwareChunker,
    FixedSizeChunker,
    RecursiveChunker,
)
from app.domain.documents import (
    DocumentSection,
    ParsedDocument,
)


def make_document(
    text: str,
    file_type: str = "txt",
    page_number: int | None = None,
) -> ParsedDocument:
    return ParsedDocument(
        filename=f"guide.{file_type}",
        file_type=file_type,
        sections=(
            DocumentSection(
                text=text,
                page_number=page_number,
            ),
        ),
    )


def test_fixed_size_chunks():
    chunks = FixedSizeChunker(
        ChunkingSettings(size=5, overlap=0)
    ).chunk(make_document("ABCDEFGHIJKL"))

    assert [c.text for c in chunks] == [
        "ABCDE", "FGHIJ", "KL"
    ]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_fixed_size_overlap():
    chunks = FixedSizeChunker(
        ChunkingSettings(size=5, overlap=2)
    ).chunk(make_document("ABCDEFGHIJKL"))

    assert [c.text for c in chunks] == [
        "ABCDE", "DEFGH", "GHIJK", "JKL"
    ]


def test_recursive_preserves_paragraphs_when_possible():
    chunks = RecursiveChunker(
        ChunkingSettings(size=20, overlap=0)
    ).chunk(
        make_document(
            "Alpha paragraph.\n\nBeta paragraph."
        )
    )

    assert [c.text for c in chunks] == [
        "Alpha paragraph.",
        "Beta paragraph.",
    ]


def test_recursive_falls_back_to_character_boundaries():
    chunks = RecursiveChunker(
        ChunkingSettings(size=6, overlap=0)
    ).chunk(make_document("abcdefghijk"))

    assert [c.text for c in chunks] == [
        "abcdef", "ghijk"
    ]


def test_recursive_with_overlap_respects_limit():
    chunks = RecursiveChunker(
        ChunkingSettings(size=6, overlap=2)
    ).chunk(make_document("abcdefghijk"))

    assert [c.text for c in chunks] == [
        "abcdef", "efghij", "ijk"
    ]
    assert all(len(c.text) <= 6 for c in chunks)


def test_document_aware_uses_markdown_headings():
    chunks = DocumentAwareChunker(
        ChunkingSettings(size=100, overlap=0)
    ).chunk(
        make_document(
            "# First\nAAA\n\n## Second\nBBB",
            file_type="md",
        )
    )

    assert [c.heading for c in chunks] == [
        "First", "Second"
    ]
    assert chunks[0].text.startswith("# First")
    assert chunks[1].text.startswith("## Second")


def test_pdf_page_metadata_never_crosses_pages():
    document = ParsedDocument(
        filename="manual.pdf",
        file_type="pdf",
        sections=(
            DocumentSection(
                text="AAAAAA", page_number=1
            ),
            DocumentSection(
                text="BBBBBB", page_number=3
            ),
        ),
        page_count=3,
    )

    for chunker in (
        FixedSizeChunker(
            ChunkingSettings(size=4, overlap=1)
        ),
        RecursiveChunker(
            ChunkingSettings(size=4, overlap=1)
        ),
        DocumentAwareChunker(
            ChunkingSettings(size=4, overlap=1)
        ),
    ):
        chunks = chunker.chunk(document)

        assert [c.page_number for c in chunks] == [
            1, 1, 3, 3
        ]
        assert [c.chunk_index for c in chunks] == [
            0, 1, 2, 3
        ]
        assert all(
            c.filename == "manual.pdf" for c in chunks
        )
        assert all(
            set(c.text) in ({"A"}, {"B"})
            for c in chunks
        )


@pytest.mark.parametrize(
    "size,overlap",
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_settings(size, overlap):
    with pytest.raises(ValueError):
        ChunkingSettings(size=size, overlap=overlap)


def test_empty_document_returns_no_chunks():
    chunks = RecursiveChunker(
        ChunkingSettings()
    ).chunk(make_document("   "))

    assert chunks == []
