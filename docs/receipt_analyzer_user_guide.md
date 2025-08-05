# Receipt Analyzer User Guide

The Lindamood Ticket Analyzer processes receipt images or PDFs using optical character recognition (OCR). It extracts key fields and organizes the receipts into categorized folders while logging their details to an Excel file.

## Requirements
- Python 3.9+
- Packages: `doctr`, `watchdog`, `pandas`, `openpyxl`

## Setup
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install doctr watchdog pandas openpyxl
   ```
3. Ensure the paths defined in `receipt_processing/main.py` exist on your system:
   - `INPUT_DIR` – folder for new receipts (default `C:/Receipts/input`)
   - `OUTPUT_DIR` – folder for processed receipts (default `C:/Receipts/processed`)
   - `LOG_FILE` – Excel workbook that stores the processing log.

## Running the App
Run the module directly:
```bash
python -m receipt_processing.main
```
The script processes any receipts already in the input directory and then watches for new files. Supported formats are `.jpg`, `.jpeg`, `.png`, and `.pdf`.

When a receipt is processed it will be moved into `OUTPUT_DIR/<category>` and a row will be appended to `LOG_FILE`. Each line item is also written to a `LineItems` sheet in the same workbook with its description, price, quantity, tax flag, category, and link to the receipt image. Image files are automatically cropped to the bounding box of all visible content with a small margin to avoid trimming edges, then rotated upright based on EXIF data or aspect ratio before OCR. After text extraction the file is renamed to `VENDOR_YYYYMMDD_PROCESSEDTIMESTAMP.ext` (e.g. `Shell_20240501_20240731_130256.jpg`). If the transaction date cannot be parsed the vendor name and processed timestamp are used. The log columns are ordered as `date`, `vendor`, `subtotal`, `tax`, `total`, `category`, `payment_method`, `card_last4`, `filename`, `processed_time`, `Receipt_Img`. If cropping yields poor OCR results you can disable it by setting ``AUTO_CROP_ENABLED = False`` in `receipt_processing/main.py`.

Stop the watcher with `Ctrl+C`.

## Customization
- Modify `CATEGORY_MAP` in `receipt_processing/utils.py` to change how vendor keywords map to categories.
- Update the `INPUT_DIR`, `OUTPUT_DIR` and `LOG_FILE` constants in `receipt_processing/main.py` to point to other locations.
- Use `process_receipt_pages` if you need to extract fields from each page of a multi-page PDF.

