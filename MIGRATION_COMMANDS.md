# Migration Commands

The BA Sandbox has been reorganized into separate project toolkits. Here are the commands to migrate each to standalone projects.

## Current Organized Structure

```
BA_Sandbox/
├── receipt_processing_toolkit/     # Complete receipt processing system
├── manifest_extraction_toolkit/    # Complete manifest extraction system  
├── experimental/                   # WM Invoice Parser (experimental)
├── docs/                          # Remaining shared docs
└── [legacy directories]           # Original structure (can be removed)
```

## Migration Commands

### 1. Receipt Processing Toolkit → Standalone Project

```bash
# Create new project
mkdir u:\dev\projects\receipt_processing_toolkit
cd u:\dev\projects\receipt_processing_toolkit

# Copy organized files
xcopy u:\dev\projects\BA_Sandbox\receipt_processing_toolkit . /E /I

# Initialize git (optional)
git init
git add .
git commit -m "Initial commit: Receipt processing toolkit"

# Install and test
pip install -r requirements.txt
python -m pytest tests/ -v
```

### 2. Manifest Extraction Toolkit → Standalone Project

```bash
# Create new project  
mkdir u:\dev\projects\manifest_extraction_toolkit
cd u:\dev\projects\manifest_extraction_toolkit

# Copy organized files
xcopy u:\dev\projects\BA_Sandbox\manifest_extraction_toolkit . /E /I

# Initialize git (optional)
git init
git add .
git commit -m "Initial commit: Manifest extraction toolkit"

# Install and test
pip install -r src/requirements.txt
python -m pytest tests/ -v
```

### 3. WM Invoice Parser → Experimental Project (Optional)

```bash
# Create experimental project
mkdir u:\dev\projects\wm_invoice_parser_experimental
cd u:\dev\projects\wm_invoice_parser_experimental

# Copy experimental files
xcopy u:\dev\projects\BA_Sandbox\experimental\WM_Invoice_Parser . /E /I

# Note: This includes a large venv - consider recreating dependencies instead
```

## Post-Migration Cleanup

After successful migration, you can clean up the original BA_Sandbox:

```bash
cd u:\dev\projects\BA_Sandbox

# Remove duplicated directories (keep originals as backup until confirmed)
# rmdir /s combine_manifest_pdf
# rmdir /s receipt_processing  
# rmdir /s WM_Invoice_Parser
# rmdir /s manifest_extraction
# rmdir /s tests
# rmdir /s docs  # Now distributed to respective toolkits

# Keep: README.md, shared_requirements.txt, and migration docs
```

## Verification

Test each migrated project:

```bash
# Receipt Processing
cd u:\dev\projects\receipt_processing_toolkit
pip install -r requirements.txt
python -c "from receipt_processing import extract_fields; print('Receipt processing OK')"
streamlit run streamlit_app.py --help

# Manifest Extraction  
cd u:\dev\projects\manifest_extraction_toolkit
pip install -r src/requirements.txt
python src/extract_manifest_fields.py --help
python -c "import sys; sys.path.insert(0, 'src'); from extract_manifest_fields import clean_manifest; print('Manifest extraction OK')"
```

## Benefits of This Structure

- **Independent deployment**: Each toolkit can be packaged separately
- **Clear ownership**: Different teams can own different toolkits  
- **Focused testing**: Tests are co-located with relevant code
- **Easier maintenance**: Smaller, focused codebases
- **Shared learning**: Common patterns can be extracted to shared utilities later