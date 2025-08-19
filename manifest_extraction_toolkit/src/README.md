# Combine Manifest PDFs

This Python script batch-processes PDF files located in a structured directory of dated subfolders. It merges all PDF files in each dated folder into a single, combined PDF with bookmarks for each file and generates an Excel log.

## Features

- Recursively scans a source directory for dated subfolders
- Merges all PDF files in each folder by creation date (oldest to newest)
- Adds PDF bookmarks for each file
- Names output using folder date and total page count
- Saves combined files to a YYYY-MM structured output path
- Generates `.xlsx` log files listing file names, page counts, and start pages

## Dependencies

- Python 3.10 (recommended via Conda)
- `pypdf`
- `pandas`
- `openpyxl`

Install via pip:

```bash
pip install -r requirements.txt

