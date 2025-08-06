"""Combine multiple manifest PDFs into one file.

This module exposes a :func:`combine_pdfs` function and can also be
executed as a script to merge input PDFs into a single output file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PyPDF2 import PdfReader, PdfWriter


def combine_pdfs(output_path: str | Path, input_paths: Iterable[str | Path]) -> Path:
    """Combine multiple PDF files into a single PDF.

    Parameters
    ----------
    output_path:
        Destination file path for the merged PDF.
    input_paths:
        Iterable of paths to the PDF files to merge in order.

    Returns
    -------
    Path
        The path to the created PDF file.
    """
    writer = PdfWriter()
    for pdf_path in input_paths:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            writer.add_page(page)
    output_path = Path(output_path)
    with output_path.open("wb") as f:
        writer.write(f)
    return output_path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Path to the combined PDF to create")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more input PDF files to merge",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    combine_pdfs(args.output, args.inputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
