# Receipt Analyzer Technical Details

This document describes the implementation of the receipt processing utility located in `analyzer_projects/Lindamood_Ticket_Analyzer_v1`.

## Module Layout
- **`receipt_processing/main.py`** – Application entry point. Configures directories, defines OCR helpers and the watcher logic.
- **`receipt_processing/utils.py`** – Helper functions and the `ReceiptFields` dataclass used to store extracted values.
- **`tests/test_receipt_utils.py`** – Unit tests that validate the field extraction logic.

## Processing Flow
1. **`extract_text`** uses the DocTR OCR model to read text lines from an image or PDF.
2. **`extract_fields`** scans those lines with regular expressions, populating a `ReceiptFields` object. Vendors are categorized using `CATEGORY_MAP`.
3. **`process_receipt`** moves the original file into a category folder and returns the extracted fields.
4. **`ReceiptFileHandler`** watches the input directory for new files and records each processed receipt to an Excel workbook using `pandas`.
5. **`run_batch`** performs initial processing of any files already present before the watcher starts.

## Development Notes
- Tests can be executed from the repository root with:
  ```bash
  PYTHONPATH=analyzer_projects/Lindamood_Ticket_Analyzer_v1 pytest
  ```
- The OCR model is loaded lazily in `_get_ocr_model()` to reduce startup cost.
- Errors encountered during file handling are printed but do not interrupt the watcher loop.

