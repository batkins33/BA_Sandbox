# BA Sandbox

This repository contains small experiments with document analysis. The most complete example is the **receipt analyzer** located under `analyzer_projects/Lindamood_Ticket_Analyzer_v1`.

The receipt analyzer monitors a directory for new receipt images or PDFs, extracts information using OCR and simple regular expressions, then organizes the files into category folders while keeping a log in an Excel workbook. Each receipt's line items are exported to a separate **LineItems** sheet in the same workbook. Image receipts are cropped before OCR and renamed based on the detected vendor and date. Cropping can be disabled by setting ``AUTO_CROP_ENABLED = False`` in ``receipt_processing/main.py`` if it causes issues with your photos.

The `WM_Invoice_Parser` directory holds earlier invoice extraction tests and is not integrated with the receipt analyzer.

## Development Notes
- Unit tests cover the receipt analyzer utilities. Run them with:
  ```bash
  PYTHONPATH=analyzer_projects/Lindamood_Ticket_Analyzer_v1 pytest
  ```
- Multi-page PDFs can be parsed page-by-page using the new `process_receipt_pages` helper.
- Further information is available in `docs/receipt_analyzer_user_guide.md` and `docs/receipt_analyzer_technical_details.md`.

## Streamlit Review UI
Run a small web UI to review and correct logged receipts:

```bash
streamlit run streamlit_app.py
```

The app loads `receipt_log.xlsx`, supports inline edits and bulk category
corrections by vendor, and can upload new receipt files into the intake
folder.

## ML-Based Categorization
Train a simple scikit-learn model from the existing log and use it for future
categorization:

```bash
python -m receipt_processing.ml_categorizer <path/to/receipt_log.xlsx>
```

This creates `receipt_category_model.joblib`. The main pipeline will use it
automatically when present.

