from pathlib import Path

from PyPDF2 import PdfReader, PdfWriter

from combine_manifest_pdf import combine_pdfs


def _create_dummy_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as f:
        writer.write(f)


def test_combine_pdfs(tmp_path: Path) -> None:
    pdf1 = tmp_path / "a.pdf"
    pdf2 = tmp_path / "b.pdf"
    _create_dummy_pdf(pdf1)
    _create_dummy_pdf(pdf2)

    output = tmp_path / "combined.pdf"
    result = combine_pdfs(output, [pdf1, pdf2])

    assert result == output
    assert output.exists()

    reader = PdfReader(str(output))
    assert len(reader.pages) == 2
