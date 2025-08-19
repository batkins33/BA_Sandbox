# Receipt Processing Toolkit

A comprehensive OCR-based receipt processing system with automated categorization, line item extraction, and web-based review interface.

## Features

- **OCR Receipt Processing**: Extract vendor, date, amounts, payment methods from receipt images/PDFs
- **Automated Categorization**: ML-based and rule-based vendor categorization
- **Line Item Extraction**: Parse individual items with prices and quantities
- **Streamlit Web UI**: Review and correct processed receipts
- **Excel Logging**: Structured output with confidence scoring
- **Duplicate Detection**: Prevent processing the same receipt twice

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure paths in config.yaml
# Run main processing
python -m receipt_processing.main

# Launch web UI for review
streamlit run streamlit_app.py
```

## Configuration

Edit `config.yaml` to set:
- Input/output directories
- OCR settings (cropping, orientation)
- Tax rates and categorization rules
- Integration endpoints

## Testing

```bash
python -m pytest tests/ -v
```

## Project Structure

- `receipt_processing/` - Core processing modules
- `tests/` - Unit tests
- `streamlit_app.py` - Web review interface
- `config.yaml` - Configuration settings
- `vendor_categories.csv` - Vendor categorization rules