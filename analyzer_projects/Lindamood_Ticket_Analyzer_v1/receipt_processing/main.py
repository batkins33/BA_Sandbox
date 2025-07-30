"""Receipt Processing App using DocTR (OCR)."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import CATEGORY_MAP, ReceiptFields, extract_fields


# --- Configuration ---
INPUT_DIR = Path("C:/Receipts/input")
OUTPUT_DIR = Path("C:/Receipts/processed")
LOG_FILE = Path("C:/Receipts/receipt_log.xlsx")


# --- Lazy OCR initialization ---
def _get_ocr_model():
    from doctr.models import ocr_predictor
    return ocr_predictor(
        det_arch="db_resnet18",
        reco_arch="crnn_mobilenet_v3_small",
        pretrained=True,
    )


# --- Utility: OCR processing ---
def extract_text(filepath: Path) -> Iterable[str]:
    """Run OCR on an image or PDF and return a list of text lines."""
    from doctr.io import DocumentFile

    if filepath.suffix.lower() == ".pdf":
        doc = DocumentFile.from_pdf(str(filepath))
    else:
        doc = DocumentFile.from_images([str(filepath)])

    model = _get_ocr_model()
    result = model(doc)
    export = result.export()
    lines: list[str] = []
    for page in export["pages"]:
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                text = "".join(word["value"] for word in line.get("words", []))
                lines.append(text)
    return lines


# --- Main receipt processor ---
def process_receipt(filepath: Path) -> ReceiptFields:
    print(f"Processing {filepath.name}...")
    lines = extract_text(filepath)
    fields = extract_fields(lines, CATEGORY_MAP)

    # Move file to output folder
    category_folder = OUTPUT_DIR / fields.category
    category_folder.mkdir(parents=True, exist_ok=True)
    new_path = category_folder / filepath.name
    shutil.move(str(filepath), str(new_path))

    return fields


def run_batch() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    for file in INPUT_DIR.glob("*"):
        if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".pdf"}:
            try:
                fields = process_receipt(file)
                record = {
                    "filename": file.name,
                    "vendor": fields.vendor,
                    "date": fields.date,
                    "total": fields.total,
                    "category": fields.category,
                    "processed_time": datetime.now().isoformat(),
                }
                records.append(record)
            except Exception as e:  # pragma: no cover - runtime protection
                print(f"Error: {file.name} - {e}")

    if records:
        df = pd.DataFrame(records)
        if LOG_FILE.exists():
            existing = pd.read_excel(LOG_FILE)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_excel(LOG_FILE, index=False)


if __name__ == "__main__":  # pragma: no cover - script entry
    run_batch()
