# BA Sandbox Reorganization Plan

## Current Issues
- Mixed project files in root directory
- Duplicate manifest extraction code
- Tests scattered across projects
- No clear separation of concerns

## Recommended Structure

```
u:\dev\projects\
├── receipt_processing_toolkit/          # Standalone project
│   ├── receipt_processing/
│   ├── tests/
│   │   ├── test_receipt_utils.py
│   │   ├── test_receipt_extraction_dataset.py
│   │   └── test_duplicate_detection.py
│   ├── streamlit_app.py
│   ├── config.yaml
│   ├── vendor_categories.csv
│   ├── requirements.txt
│   └── README.md
│
├── manifest_extraction_toolkit/         # Standalone project
│   ├── src/
│   │   ├── extract_manifest_fields.py
│   │   ├── extract_manifest_numbers.py
│   │   ├── combine_manifest.py
│   │   └── main.py
│   ├── tests/
│   │   ├── test_manifest_extraction.py
│   │   └── test_combine_manifest_pdf.py
│   ├── docs/
│   ├── requirements.txt
│   └── README.md
│
└── document_processing_shared/          # Shared utilities (optional)
    ├── ocr_utils.py                     # Common OCR functions
    ├── pdf_utils.py                     # Common PDF operations
    └── field_extractors.py              # Shared extraction patterns
```

## Resource Sharing Analysis

### Should be Grouped (High Efficiency):
- **Receipt + Manifest**: Both use OCR, PDF processing, field extraction
- **Shared Dependencies**: pytesseract, PyMuPDF, Pillow, pandas

### Should Stay Separate:
- **WM Invoice Parser**: Different domain, experimental status
- **Different Use Cases**: Receipt (retail) vs Manifest (logistics) vs Invoice (B2B)

## Migration Commands

### 1. Create Receipt Processing Project
```bash
mkdir u:\dev\projects\receipt_processing_toolkit
xcopy receipt_processing u:\dev\projects\receipt_processing_toolkit\receipt_processing /E /I
copy streamlit_app.py u:\dev\projects\receipt_processing_toolkit\
copy config.yaml u:\dev\projects\receipt_processing_toolkit\
copy vendor_categories.csv u:\dev\projects\receipt_processing_toolkit\
mkdir u:\dev\projects\receipt_processing_toolkit\tests
copy tests\test_receipt_*.py u:\dev\projects\receipt_processing_toolkit\tests\
copy tests\test_duplicate_detection.py u:\dev\projects\receipt_processing_toolkit\tests\
copy tests\conftest.py u:\dev\projects\receipt_processing_toolkit\tests\
```

### 2. Create Manifest Extraction Project  
```bash
mkdir u:\dev\projects\manifest_extraction_toolkit
mkdir u:\dev\projects\manifest_extraction_toolkit\src
xcopy combine_manifest_pdf u:\dev\projects\manifest_extraction_toolkit\src /E /I
mkdir u:\dev\projects\manifest_extraction_toolkit\tests
copy tests\test_combine_manifest_pdf.py u:\dev\projects\manifest_extraction_toolkit\tests\
mkdir u:\dev\projects\manifest_extraction_toolkit\docs
copy docs\combine_manifest_pdf_*.md u:\dev\projects\manifest_extraction_toolkit\docs\
```

### 3. Create Shared Library (Optional)
```bash
mkdir u:\dev\projects\document_processing_shared
# Extract common OCR/PDF utilities from both projects
```

## Benefits of This Structure

### Separate Projects:
- **Independent deployment**: Each can be packaged/distributed separately
- **Clear ownership**: Different teams can own different projects
- **Focused testing**: Tests are co-located with relevant code
- **Easier maintenance**: Smaller, focused codebases

### Shared Library Option:
- **Code reuse**: Common OCR patterns, PDF utilities
- **Consistency**: Standardized field extraction approaches
- **Efficiency**: Single place for OCR optimizations

## Recommendation: Hybrid Approach

1. **Keep projects separate** for deployment/ownership
2. **Create shared utilities package** for common functionality
3. **Use shared package as dependency** in both projects

This provides maximum flexibility while avoiding code duplication.