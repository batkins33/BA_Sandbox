"""Receipt Processing App using DocTR (OCR)."""

from __future__ import annotations

import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import pandas as pd

from utils import CATEGORY_MAP, ReceiptFields, extract_fields, load_vendor_categories

try:  # Optional dependency for image processing
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional
    cv2 = None
    np = None


# --- Configuration ---
INPUT_DIR = Path(r"C:\Users\brian.atkins\Dropbox\Personal\Receipts\input")
OUTPUT_DIR = Path(r"G:\My Drive\receipts\processed")
LOG_FILE = Path(r"G:\My Drive\receipts\receipt_log.xlsx")

# Enable or disable automatic image cropping prior to OCR.  Set to ``False``
# if the cropping logic negatively impacts OCR accuracy for your photos.
AUTO_CROP_ENABLED = True
AUTO_ORIENT_ENABLED = True

# Optional vendor categorization via CSV mapping
USE_VENDOR_CSV = True
VENDOR_CSV_PATH = Path("vendor_categories.csv")
VENDOR_MAP = load_vendor_categories(VENDOR_CSV_PATH) if USE_VENDOR_CSV and VENDOR_CSV_PATH.exists() else None



# --- Lazy OCR initialization ---
def _get_ocr_model():
    from doctr.models import ocr_predictor
    return ocr_predictor(
        det_arch="db_resnet34",  # <-- Use db_resnet34 (commonly available)
        reco_arch="crnn_mobilenet_v3_small",
        pretrained=True,
    )



# --- Utility: OCR processing ---
def extract_text_pages(filepath: Path) -> List[List[str]]:
    """Return OCR text for each page of an image or PDF."""
    from doctr.io import DocumentFile

    if filepath.suffix.lower() == ".pdf":
        doc = DocumentFile.from_pdf(str(filepath))
    else:
        doc = DocumentFile.from_images([str(filepath)])

    model = _get_ocr_model()
    result = model(doc)
    export = result.export()
    pages: List[List[str]] = []
    for page in export["pages"]:
        lines: List[str] = []
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                text = "".join(word["value"] for word in line.get("words", []))
                lines.append(text)
        pages.append(lines)
    return pages


def extract_text(filepath: Path) -> Iterable[str]:
    """Run OCR on an image or PDF and return all text lines."""
    pages = extract_text_pages(filepath)
    lines: list[str] = []
    for page in pages:
        lines.extend(page)
    return lines


def auto_crop_image(image_path: Path) -> Path:
    """Attempt to detect the document boundaries and overwrite the image with a
    cropped version. Returns the path that should be used for OCR."""
    if cv2 is None or np is None:
        return image_path

    img = cv2.imread(str(image_path))
    if img is None:
        return image_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    screen = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen = approx
            break

    if screen is None:
        return image_path

    pts = screen.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH))

    cv2.imwrite(str(image_path), warped)
    return image_path


def correct_orientation(image_path: Path) -> Path:
    """Rotate the image if its width is greater than its height."""
    if cv2 is None:
        return image_path

    img = cv2.imread(str(image_path))
    if img is None:
        return image_path

    h, w = img.shape[:2]
    if w > h:
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(str(image_path), rotated)
    return image_path


def preprocess_image(image_path: Path) -> Path:
    """Apply optional cropping and orientation correction."""
    if AUTO_CROP_ENABLED:
        image_path = auto_crop_image(image_path)
    if AUTO_ORIENT_ENABLED:
        image_path = correct_orientation(image_path)
    return image_path


def _parse_date(date_str: str) -> str | None:
    """Return YYYYMMDD or None if parsing fails."""
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y%m%d")
        except ValueError:
            pass
    return None


def rename_receipt_file(filepath: Path, vendor: str, date_str: str) -> Path:
    """Rename the file as ``Vendor_TransactionDate_ProcessedTimestamp``.

    If the transaction date cannot be parsed, it is omitted. The processed
    timestamp is always appended using ``YYYYMMDD_HHMMSS`` format.
    """

    date_fmt = _parse_date(date_str)
    safe_vendor = re.sub(r"[^A-Za-z0-9]+", "_", vendor).strip("_") or "receipt"
    processed_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if date_fmt:
        new_name = f"{safe_vendor}_{date_fmt}_{processed_ts}{filepath.suffix.lower()}"
    else:
        new_name = f"{safe_vendor}_{processed_ts}{filepath.suffix.lower()}"

    new_path = filepath.with_name(new_name)
    filepath.rename(new_path)
    return new_path


# --- Main receipt processor ---
def process_receipt(filepath: Path) -> ReceiptFields:
    print(f"Processing {filepath.name}...")

    if filepath.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        filepath = preprocess_image(filepath)

    lines = extract_text(filepath)
    fields = extract_fields(lines, CATEGORY_MAP, VENDOR_MAP)

    filepath = rename_receipt_file(filepath, fields.vendor, fields.date)

    category_folder = OUTPUT_DIR / fields.category
    category_folder.mkdir(parents=True, exist_ok=True)
    new_path = category_folder / filepath.name
    shutil.move(str(filepath), str(new_path))

    return fields


def process_receipt_pages(filepath: Path) -> tuple[List[ReceiptFields], Path]:
    """Process a multi-page PDF and return fields for each page."""

    if filepath.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        filepath = preprocess_image(filepath)

    pages = extract_text_pages(filepath)
    fields_list: List[ReceiptFields] = []
    for lines in pages:
        fields_list.append(extract_fields(lines, CATEGORY_MAP, VENDOR_MAP))

    vendor = fields_list[0].vendor if fields_list else "receipt"
    date_val = fields_list[0].date if fields_list else ""
    filepath = rename_receipt_file(filepath, vendor, date_val)

    if fields_list:
        category_folder = OUTPUT_DIR / fields_list[0].category
    else:
        category_folder = OUTPUT_DIR / "unknown"
    category_folder.mkdir(parents=True, exist_ok=True)
    new_path = category_folder / filepath.name
    shutil.move(str(filepath), str(new_path))

    return fields_list, new_path


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
            page_fields, final_path = process_receipt_pages(filepath)
            records: list[dict[str, str]] = []
            for fields in page_fields:
                record = {
                    "date": fields.date,
                    "vendor": fields.vendor,
                    "subtotal": fields.subtotal,
                    "tax": fields.tax,
                    "total": fields.total,
                    "category": fields.category,
                    "payment_method": fields.payment_method,
                    "card_last4": fields.card_last4,
                    "filename": final_path.name,
                    "processed_time": datetime.now().isoformat(),
                }
                records.append(record)

            if records:
                df = pd.DataFrame(records)[
                    [
                        "date",
                        "vendor",
                        "subtotal",
                        "tax",
                        "total",
                        "category",
                        "payment_method",
                        "card_last4",
                        "filename",
                        "processed_time",
                    ]
                ]
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
                page_fields, final_path = process_receipt_pages(file)
                for fields in page_fields:
                    record = {
                        "date": fields.date,
                        "vendor": fields.vendor,
                        "subtotal": fields.subtotal,
                        "tax": fields.tax,
                        "total": fields.total,
                        "category": fields.category,
                        "payment_method": fields.payment_method,
                        "card_last4": fields.card_last4,
                        "filename": final_path.name,
                        "processed_time": datetime.now().isoformat(),
                    }
                    records.append(record)
            except Exception as e:  # pragma: no cover - runtime protection
                print(f"Error: {file.name} - {e}")

    if records:
        df = pd.DataFrame(records)[
            [
                "date",
                "vendor",
                "subtotal",
                "tax",
                "total",
                "category",
                "payment_method",
                "card_last4",
                "filename",
                "processed_time",
            ]
        ]
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
