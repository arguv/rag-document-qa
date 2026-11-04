
import pytest
from reportlab.pdfgen import canvas

from app.infrastructure.document_loader import (
    DocumentLoadError,
    load_document,
)


def test_load_txt(tmp_path):
    file = tmp_path / "notes.txt"
    file.write_text(
        "RAG uses external knowledge.",
        encoding="utf-8",
    )

    document = load_document(file)

    assert document.filename == "notes.txt"
    assert document.file_type == "txt"
    assert document.text == "RAG uses external knowledge."
    assert document.page_count is None
    assert document.sections[0].page_number is None


def test_load_markdown(tmp_path):
    file = tmp_path / "guide.md"
    file.write_text(
        "# RAG Guide\n\nDocument ingestion.",
        encoding="utf-8",
    )

    document = load_document(file)

    assert document.file_type == "md"
    assert "# RAG Guide" in document.text
    assert "Document ingestion." in document.text


def test_load_pdf(tmp_path):
    file = tmp_path / "guide.pdf"

    pdf = canvas.Canvas(str(file))
    pdf.drawString(72, 750, "First PDF page")
    pdf.showPage()
    pdf.drawString(72, 750, "Second PDF page")
    pdf.showPage()
    pdf.save()

    document = load_document(file)

    assert document.file_type == "pdf"
    assert document.page_count == 2
    assert len(document.sections) == 2

    assert document.sections[0].page_number == 1
    assert "First PDF page" in document.sections[0].text

    assert document.sections[1].page_number == 2
    assert "Second PDF page" in document.sections[1].text


def test_unsupported_format(tmp_path):
    file = tmp_path / "image.jpg"
    file.write_bytes(b"fake-image")

    with pytest.raises(
        DocumentLoadError,
        match="Unsupported document type",
    ):
        load_document(file)


def test_missing_file(tmp_path):
    file = tmp_path / "missing.pdf"

    with pytest.raises(
        DocumentLoadError,
        match="File not found",
    ):
        load_document(file)


def test_empty_document(tmp_path):
    file = tmp_path / "empty.txt"
    file.write_text("", encoding="utf-8")

    with pytest.raises(
        DocumentLoadError,
        match="No extractable text",
    ):
        load_document(file)


def test_pdf_without_text(tmp_path):
    file = tmp_path / "blank.pdf"

    pdf = canvas.Canvas(str(file))
    pdf.showPage()
    pdf.save()

    with pytest.raises(
        DocumentLoadError,
        match="No extractable text",
    ):
        load_document(file)
