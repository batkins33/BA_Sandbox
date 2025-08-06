import os
import re
from pathlib import Path
import pandas as pd
from pdf2image import convert_from_path
import pytesseract

# === CONFIGURATION ===
COMBINED_DIR = Path(r"C:\Users\brian.atkins\OneDrive - Lindamood Demolition\Documents - Truck Tickets\General\24-105-PHMS New Pediatric Campus\MANIFEST\Combined\2025-06")
OUTPUT_FILE = COMBINED_DIR / "manifest_number_log.xlsx"

# Optional: Set this if Tesseract isn't in PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_manifest_number(text):
    match = re.search(r"\b\d{8}\b", text)
    return match.group(0) if match else None

results = []

for pdf_path in COMBINED_DIR.rglob("*.pdf"):
    print(f"[PROCESSING] {pdf_path.name}")
    try:
        images = convert_from_path(str(pdf_path), dpi=300)
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img)
            manifest = extract_manifest_number(text)
            results.append({
                "File": pdf_path.name,
                "Page": i + 1,
                "Manifest Number": manifest
            })
    except Exception as e:
        print(f"[ERROR] {pdf_path}: {e}")

df = pd.DataFrame(results)
df.to_excel(OUTPUT_FILE, index=False)
print(f"[DONE] Manifest numbers logged to: {OUTPUT_FILE}")
