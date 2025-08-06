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
3. Review `config.yaml` and update any paths for your system:
   - `input_dir` – folder for new receipts (default `C:/Receipts/input`)
   - `output_dir` – folder for processed receipts (default `C:/Receipts/processed`)
   - `log_file` – Excel workbook that stores the processing log.

## Running the App
Run the module directly:
```bash
python -m receipt_processing.main
```
The script processes any receipts already in the input directory and then watches for new files. Supported formats are `.jpg`, `.jpeg`, `.png`, and `.pdf`.

When a receipt is processed it will be moved into `OUTPUT_DIR/<category>` and a row will be appended to `LOG_FILE`. Each line item is also written to a `LineItems` sheet in the same workbook with its description, price, quantity, taxable flag, category, and link to the receipt image. Image files are automatically cropped to the bounding box of all visible content with a small margin to avoid trimming edges, then deskewed using Hough-line angle detection and a perspective transform when a four-corner contour is found. As a final safeguard images are rotated upright based on EXIF data or aspect ratio before OCR. After text extraction the file is renamed to `VENDOR_YYYYMMDD_PROCESSEDTIMESTAMP.ext` (e.g. `Shell_20240501_20240731_130256.jpg`). If the transaction date cannot be parsed the vendor name and processed timestamp are used. The log columns are ordered as `date`, `vendor`, `subtotal`, `tax`, `total`, `category`, `payment_method`, `card_last4`, `filename`, `processed_time`, `Receipt_Img`, `Total_Img`, `CardLast4_Img`. If cropping yields poor OCR results you can disable it by setting ``auto_crop_enabled: false`` in `config.yaml`.  In addition, every field listed in `extraction_rules.yaml` is cropped to a small JPEG and saved in a `FieldImages` folder alongside the receipt using the field description as the filename suffix (for example `_TicketNum.jpg` or `_Manifest.jpg`).

Stop the watcher with `Ctrl+C`.

## Customization
- Adjust the `category_map`, tax rate, thresholds, and folder locations in `config.yaml`.
- Use `process_receipt_pages` if you need to extract fields from each page of a multi-page PDF.

