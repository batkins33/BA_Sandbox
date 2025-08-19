# Combine Manifest PDF – User Guide

This utility merges multiple shipment manifest PDFs into a single document. It
can be used either as a library function or run as a small command-line tool.

## Requirements
- Python 3.9+
- [PyPDF2](https://pypi.org/project/PyPDF2/)

Install the dependency with:
```bash
pip install PyPDF2
```

## Command-Line Usage
Merge two or more PDFs into a new file:
```bash
python -m combine_manifest_pdf.main combined.pdf manifest1.pdf manifest2.pdf
```
The first argument is the output PDF path followed by the input files in the
desired order.

## Library Usage
```python
from combine_manifest_pdf import combine_pdfs

combine_pdfs("combined.pdf", ["manifest1.pdf", "manifest2.pdf"])
```
The function returns the path to the created file.
