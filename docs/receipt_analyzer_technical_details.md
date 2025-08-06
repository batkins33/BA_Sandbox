# Receipt Analyzer Technical Details

This document describes the implementation of the receipt processing utility located in `analyzer_projects/Lindamood_Ticket_Analyzer_v1`.

## Module Layout
- **`receipt_processing/main.py`** – Application entry point. Configures directories, defines OCR helpers and the watcher logic.
- **`receipt_processing/utils.py`** – Helper functions and the `ReceiptFields` dataclass used to store extracted values.
- **`tests/test_receipt_utils.py`** – Unit tests that validate the field extraction logic.

## Processing Flow
1. **`extract_text_pages`** returns OCR text and word bounding boxes for each page of an image or PDF.
2. **`extract_text`** flattens those results into a single list of lines.
3. **`extract_fields`** scans those lines with regular expressions, populating a `ReceiptFields` object. Vendors are categorized using `CATEGORY_MAP` as defined in `config.yaml`. Missing totals or subtotals are computed from the other values when possible. Line items are marked as taxable when the receipt uses flags such as `T` or `A`; if no flags are present the function infers the taxable subset by matching item totals to the overall tax using the local sales tax rate.
4. **`process_receipt_pages`** iterates through PDF pages, auto-crops image files, deskews them and extracts fields for each page, renames the file to `VENDOR_YYYYMMDD_PROCESSEDTIMESTAMP.ext`, moves it into a category folder and returns the extracted `ReceiptFields` list along with the final path. Cropping uses a safe bounding box approach with a margin so white receipts on light backgrounds are not clipped. Skew is detected with Hough line analysis and the image is rotated accordingly; when a four-corner contour is detected a perspective transform flattens angled photos. The helper `process_receipt` wraps the same logic for single-page receipts. Cropping can be disabled via the ``auto_crop_enabled`` option in `config.yaml` if needed. For each field listed in `extraction_rules.yaml`, a region-of-interest JPEG is also generated using the field name as the suffix (e.g. `_TicketNum.jpg`, `_Manifest.jpg`).
5. **`ReceiptFileHandler`** watches the input directory for new files and records each page's fields to an Excel workbook using `pandas`. The main sheet stores receipt-level data ordered as `date`, `vendor`, `subtotal`, `tax`, `total`, `category`, `payment_method`, `card_last4`, `filename`, `processed_time`, `Receipt_Img`, `Total_Img`, `CardLast4_Img` (hyperlinks to field crops). A second sheet named `LineItems` captures one row per line item with `receipt_id`, `date`, `vendor`, `item_description`, `item_price`, `quantity`, `taxable`, `category` and a hyperlink to the image.
6. **`run_batch`** performs initial processing of any files already present before the watcher starts.

## Development Notes
- Tests can be executed from the repository root with:
  ```bash
  PYTHONPATH=analyzer_projects/Lindamood_Ticket_Analyzer_v1 pytest
  ```
- The OCR model is loaded lazily in `_get_ocr_model()` to reduce startup cost.
- Errors encountered during file handling are printed but do not interrupt the watcher loop.

