# BA Sandbox

This repository contains small experiments with document analysis. The most complete example is the **receipt analyzer** located under `analyzer_projects/Lindamood_Ticket_Analyzer_v1`.

The receipt analyzer monitors a directory for new receipt images or PDFs, extracts information using OCR and simple regular expressions, then organizes the files into category folders while keeping a log in an Excel workbook.

The `WM_Invoice_Parser` directory holds earlier invoice extraction tests and is not integrated with the receipt analyzer.

## Development Notes
- Unit tests cover the receipt analyzer utilities. Run them with:
  ```bash
  PYTHONPATH=analyzer_projects/Lindamood_Ticket_Analyzer_v1 pytest
  ```
- Further information is available in `docs/receipt_analyzer_user_guide.md` and `docs/receipt_analyzer_technical_details.md`.

