# Receipt Analyzer Technical Details

This document describes the implementation of the receipt processing utility located in `analyzer_projects/Lindamood_Ticket_Analyzer_v1`.

## Module Layout
- **`receipt_processing/main.py`** – Application entry point. Configures directories, defines OCR helpers and the watcher logic.
- **`receipt_processing/utils.py`** – Helper functions and the `ReceiptFields` dataclass used to store extracted values.
- **`tests/test_receipt_utils.py`** – Unit tests that validate the field extraction logic.

## Processing Flow
1. **`extract_text_pages`** returns OCR text for each page of an image or PDF.
2. **`extract_text`** flattens those results into a single list of lines.
3. **`extract_fields`** scans those lines with regular expressions, populating a `ReceiptFields` object. Vendors are categorized using `CATEGORY_MAP`.
4. **`process_receipt_pages`** iterates through PDF pages, auto-crops image files, extracts fields for each page, renames the file to `YYYYMMDD_VENDOR.ext`, moves it into a category folder and returns the extracted `ReceiptFields` list along with the final path. The helper `process_receipt` wraps the same logic for single-page receipts.
5. **`ReceiptFileHandler`** watches the input directory for new files and records each page's fields to an Excel workbook using `pandas`.
6. **`run_batch`** performs initial processing of any files already present before the watcher starts.

## Development Notes
- Tests can be executed from the repository root with:
  ```bash
  PYTHONPATH=analyzer_projects/Lindamood_Ticket_Analyzer_v1 pytest
  ```
- The OCR model is loaded lazily in `_get_ocr_model()` to reduce startup cost.
- Errors encountered during file handling are printed but do not interrupt the watcher loop.

