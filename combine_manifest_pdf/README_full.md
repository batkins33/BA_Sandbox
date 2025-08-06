# Manifest Processing Pipeline

This project provides a series of scripts to automate the processing, merging, OCR extraction, organization, and summarization of manifest PDF files. It is tailored for operations such as construction trucking or demolition where batch processing of ticket/manifest PDFs is common.

## Workflow Overview

1. **Merge PDF Manifests by Day**

   - Script: `combine_manifest.py`
   - Merges all PDFs in each dated folder into a single PDF with bookmarks (one per source file).
   - Generates an Excel log listing files, their page counts, and starting page numbers.
   - Output is stored by month in a structured directory.

2. **Extract Manifest Numbers (OCR)**

   - Script: `extract_manifest_numbers.py`
   - For every combined PDF, converts each page to an image and applies OCR (Tesseract) to extract 8-digit manifest numbers.
   - Outputs a log with filename, page number, and manifest number found.

3. **Organize Log Files**

   - Script: `MoveLogs.py`
   - Moves each Excel log generated in the month folder to a `/logs` subfolder for improved organization and later summarization.

4. **Summarize Page Counts**

   - Script: `summarize_pages.py`
   - Reads all log files, annotates with date/month/week columns, and creates a summary pivot table of total page counts by month, week, and day.
   - Saves a summary Excel report at the root of the `Combined` directory.

---

## Dependencies

- Python 3.10+
- `pypdf`, `pdf2image`, `pytesseract`, `pandas`, `openpyxl`
- Tesseract OCR (install separately if not in PATH)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Directory Structure Example

```
MANIFEST/
├── ToBeProcessed/
│   ├── 05.14.2024/
│   │   ├── A.pdf
│   │   ├── B.pdf
│   └── ...
├── Combined/
│   ├── 2024-05/
│   │   ├── 24-105_05.14.2024_manifest_combined_8.pdf
│   │   ├── 24-105_05.14.2024_manifest_combined_8.xlsx
│   │   └── logs/
│   │       └── 24-105_05.14.2024_manifest_combined_8.xlsx
│   └── combined_manifest_summary.xlsx
```

---

## Usage

### 1. Merge PDFs & Create Logs

```bash
python combine_manifest.py
```

- Output: One combined PDF and one `.xlsx` log per dated folder.

### 2. Extract Manifest Numbers from PDFs (OCR)

```bash
python extract_manifest_numbers.py
```

- Output: `manifest_number_log.xlsx` in `/Combined`.

### 3. Organize Log Files

```bash
python MoveLogs.py
```

- Output: Moves all log files into `logs/` subfolders by month.

### 4. Summarize Page Counts

```bash
python summarize_pages.py
```

- Output: `combined_manifest_summary.xlsx` summarizing page counts by day/week/month.

---

## Notes & Best Practices

- **File Naming:** Date folders must follow `MM.DD.YYYY` format for the merge script to detect them.
- **Tesseract:** If not in your system PATH, set `pytesseract.pytesseract.tesseract_cmd` in `extract_manifest_numbers.py`.
- **OCR Errors:** Check manifest number log for missing numbers—OCR may sometimes fail; manual correction may be needed.
- **Log File Patterns:** Only files matching `24-105_MM.DD.YYYY_manifest_combined_*.xlsx` are summarized.

---

## Author

Created by Brian Atkins for Lindamood Demolition operations automation.

