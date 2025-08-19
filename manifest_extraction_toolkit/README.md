# Manifest Extraction Toolkit

Advanced OCR pipeline for extracting structured fields from manifest PDFs using anchor-based positioning and ROI extraction.

## Features

- **Anchor-Based OCR**: Precise field extraction using label positioning
- **Transporter Fields**: Extract company names, EPA IDs, state IDs, phone numbers
- **Handwritten Support**: Process license plates, DOT IDs, truck numbers
- **PDF Combining**: Merge multiple manifest PDFs
- **Field Validation**: Normalize phone numbers, validate manifest numbers
- **Batch Processing**: Process entire directories of PDFs

## Quick Start

```bash
# Install dependencies
pip install -r src/requirements.txt

# Extract all fields from PDFs
python src/extract_manifest_fields.py /path/to/pdfs --out results.xlsx

# Extract manifest numbers only
python src/extract_manifest_numbers.py

# Combine PDFs
python src/main.py output.pdf input1.pdf input2.pdf
```

## Output Format

Excel file with columns:
- `source_file`, `page`, `manifest_number`
- `t1_company`, `t1_us_epa_id`, `t1_state_id`, `t1_phone`
- `t2_company`, `t2_us_epa_id`, `t2_state_id`, `t2_phone`
- `license_plate`, `dot_id`, `truck_number`

## Configuration

- ROI positions adjustable in `FieldSpec` definitions
- Handwritten processing uses heavy binarization
- Anchor search is case-insensitive and OCR-tolerant

## Testing

```bash
python -m pytest tests/ -v
```

## Project Structure

- `src/` - Core extraction modules
- `tests/` - Unit tests
- `docs/` - Technical documentation