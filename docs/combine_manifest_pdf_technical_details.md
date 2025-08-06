# Combine Manifest PDF – Technical Details

The `combine_manifest_pdf` module provides a minimal implementation for
concatenating PDF files. It relies on the [`PyPDF2`](https://pypi.org/project/PyPDF2/) library
which handles PDF parsing and writing.

## Module Structure
- `combine_manifest_pdf/__init__.py` – re-exports the main `combine_pdfs` function.
- `combine_manifest_pdf/main.py` – implementation and simple command-line
  interface.

## Implementation Notes
`combine_pdfs` iterates over each page of the input PDFs using
`PyPDF2.PdfReader` and appends them to a `PdfWriter` instance. The combined
pages are written to the output path provided by the caller. No attempt is made
to optimise or validate the input files beyond what `PyPDF2` performs.

The `main` function exposes a CLI that accepts an output file followed by one or
more input PDFs. This allows the module to be executed with `python -m
combine_manifest_pdf.main` without requiring additional wrappers.
