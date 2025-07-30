"""Receipt Processing App using DocTR (OCR)."""

from __future__ import annotations

import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import pandas as pd

from utils import CATEGORY_MAP, ReceiptFields, extract_fields


# --- Configuration ---
INPUT_DIR = Path(r"C:\Users\brian.atkins\Dropbox\Personal\Receipts\input")
OUTPUT_DIR = Path(r"C:\Users\brian.atkins\Dropbox\Personal\Receipts\processed")
LOG_FILE = Path(r"C:\Users\brian.atkins\Dropbox\Personal\Receipts\receipt_log.xlsx")



# --- Lazy OCR initialization ---
def _get_ocr_model():
    from doctr.models import ocr_predictor
    return ocr_predictor(
        det_arch="db_resnet34",  # <-- Use db_resnet34 (commonly available)
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


class ReceiptFileHandler(FileSystemEventHandler):
    """Handle new files dropped into the input directory."""

    def on_created(self, event):
        """Process new receipt files as they appear."""
        if event.is_directory:
            return

        filepath = Path(event.src_path)
        if filepath.suffix.lower() not in {".jpg", ".jpeg", ".png", ".pdf"}:
            return

        try:
            fields = process_receipt(filepath)
            record = {
                "filename": filepath.name,
                "vendor": fields.vendor,
                "date": fields.date,
                "total": fields.total,
                "category": fields.category,
                "processed_time": datetime.now().isoformat(),
            }
            df = pd.DataFrame([record])
            if LOG_FILE.exists():
                existing = pd.read_excel(LOG_FILE)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_excel(LOG_FILE, index=False)
        except Exception as e:  # pragma: no cover - runtime protection
            print(f"Error processing {filepath.name}: {e}")


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
    # Optional initial batch in case files already exist when starting
    run_batch()

    event_handler = ReceiptFileHandler()
    observer = Observer()
    observer.schedule(event_handler, str(INPUT_DIR), recursive=False)
    observer.start()
    print("Watching for new receipt files...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
