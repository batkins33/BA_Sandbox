"""Receipt Processing App using DocTR (OCR)."""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import pandas as pd

from integrations import (
    push_to_firefly,
    push_to_google_sheets,
    push_to_sharepoint,
)
from ml_categorizer import load_model, predict_category
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
LINE_ITEMS_SHEET = "LineItems"

RECEIPT_COLUMNS = [
    "receipt_id",
    "date",
    "vendor",
    "subtotal",
    "tax",
    "total",
    "category",
    "payment_method",
    "card_last4",
    "line_items",
    "filename",
    "processed_time",
    "Receipt_Img",
]

LINE_ITEM_COLUMNS = [
    "receipt_id",
    "date",
    "vendor",
    "item_description",
    "item_price",
    "quantity",
    "taxable",
    "category",
    "image_link",
]

# Enable or disable automatic image cropping prior to OCR.  Set to ``False``
# if the cropping logic negatively impacts OCR accuracy for your photos.
AUTO_CROP_ENABLED = True
AUTO_ORIENT_ENABLED = True

# Optional vendor categorization via CSV mapping
USE_VENDOR_CSV = True
VENDOR_CSV_PATH = Path("vendor_categories.csv")
VENDOR_MAP = load_vendor_categories(VENDOR_CSV_PATH) if USE_VENDOR_CSV and VENDOR_CSV_PATH.exists() else None
ML_MODEL = load_model()



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


def auto_crop_receipt_image(img: "np.ndarray") -> "np.ndarray":
    """Return a tightly cropped copy of ``img``.

    The previous implementation attempted a perspective transform based on the
    largest contour which could easily cut off light edges of white receipts.
    This helper instead computes a bounding box around *all* content using a
    binary mask and retains a small margin. If the detected region is too small
    the original image is returned untouched.
    """

    # Convert to grayscale and threshold to highlight receipt edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )

    # Connect nearby regions so the bounding box covers the whole document
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    morph = cv2.dilate(morph, kernel, iterations=1)

    coords = cv2.findNonZero(morph)
    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)

    # Add a small margin around the detected region
    pad = 10
    x, y = max(0, x - pad), max(0, y - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    h = min(img.shape[0] - y, h + 2 * pad)

    # Sanity check to avoid over‑cropping.  If the proposed crop is less than
    # half the width or height of the original image we keep the original.
    orig_h, orig_w = img.shape[:2]
    if w < orig_w * 0.5 or h < orig_h * 0.5:
        return img

    return img[y : y + h, x : x + w]


def auto_crop_image(image_path: Path) -> Path:
    """Load ``image_path`` and overwrite it with a safely cropped version."""
    if cv2 is None or np is None:
        return image_path

    img = cv2.imread(str(image_path))
    if img is None:
        return image_path

    cropped = auto_crop_receipt_image(img)
    cv2.imwrite(str(image_path), cropped)
    return image_path


def correct_orientation(image_path: Path) -> Path:
    """Ensure the image is upright using EXIF metadata and simple heuristics."""
    if cv2 is None:
        return image_path

    # First honour any EXIF orientation data if Pillow is available
    try:  # pragma: no cover - optional dependency
        from PIL import Image, ImageOps

        with Image.open(image_path) as pil_img:
            fixed = ImageOps.exif_transpose(pil_img)
            fixed.save(image_path)
    except Exception:
        pass

    img = cv2.imread(str(image_path))
    if img is None:
        return image_path

    h, w = img.shape[:2]
    # If still in landscape orientation, rotate to portrait as a fallback
    if w > h:
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(str(image_path), rotated)
    return image_path


def deskew_image(image_path: Path) -> Path:
    """Detect and correct minor skew in the receipt image."""
    if cv2 is None or np is None:
        return image_path

    img = cv2.imread(str(image_path))
    if img is None:
        return image_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return image_path

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.1:
        return image_path

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    cv2.imwrite(str(image_path), rotated)
    return image_path


def enhance_image_for_ocr(image_path: Path) -> Path:
    """Denoise, sharpen, and apply adaptive thresholding to boost text contrast."""
    if cv2 is None or np is None:
        return image_path

    img = cv2.imread(str(image_path))
    if img is None:
        return image_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)
    thresh = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )
    cv2.imwrite(str(image_path), thresh)
    return image_path


def preprocess_image(image_path: Path) -> Path:
    """Apply cropping, orientation, deskewing, and contrast enhancement."""
    if AUTO_CROP_ENABLED:
        image_path = auto_crop_image(image_path)
    if AUTO_ORIENT_ENABLED:
        image_path = correct_orientation(image_path)
    image_path = deskew_image(image_path)
    image_path = enhance_image_for_ocr(image_path)
    return image_path


def _parse_date(date_str: str) -> str | None:
    """Return YYYYMMDD or None if parsing fails."""
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y%m%d")
        except ValueError:
            pass
    return None


def _image_hyperlink(path: Path | None) -> str:
    """Return an Excel hyperlink formula for ``path``."""
    if path is None:
        return ""
    return f'=HYPERLINK("{path}", "Image")'


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
    if ML_MODEL is not None:
        pred, conf = predict_category(
            fields.vendor, "\n".join(fields.lines), fields.total, model=ML_MODEL
        )
        if pred and conf >= 0.5:
            fields.category = pred

    filepath = rename_receipt_file(filepath, fields.vendor, fields.date)

    category_folder = OUTPUT_DIR / fields.category
    category_folder.mkdir(parents=True, exist_ok=True)
    new_path = category_folder / filepath.name
    shutil.move(str(filepath), str(new_path))

    receipt_data = asdict(fields)
    push_to_firefly(receipt_data)
    push_to_google_sheets(receipt_data)
    push_to_sharepoint(receipt_data)

    return fields


def process_receipt_pages(filepath: Path) -> tuple[List[ReceiptFields], Path]:
    """Process a multi-page PDF and return fields for each page."""

    if filepath.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        filepath = preprocess_image(filepath)

    pages = extract_text_pages(filepath)
    fields_list: List[ReceiptFields] = []
    for lines in pages:
        fields = extract_fields(lines, CATEGORY_MAP, VENDOR_MAP)
        if ML_MODEL is not None:
            pred, conf = predict_category(
                fields.vendor, "\n".join(fields.lines), fields.total, model=ML_MODEL
            )
            if pred and conf >= 0.5:
                fields.category = pred
        fields_list.append(fields)

    # Determine destination folder based on first page's category
    if fields_list:
        category_folder = OUTPUT_DIR / fields_list[0].category
    else:
        category_folder = OUTPUT_DIR / "unknown"
    category_folder.mkdir(parents=True, exist_ok=True)

    # Multi-page PDF handling: save each page as an image and move original
    is_multi_pdf = filepath.suffix.lower() == ".pdf" and len(pages) > 1
    if is_multi_pdf:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_pdf_name = f"{filepath.stem}_{timestamp}{filepath.suffix.lower()}"
        new_path = category_folder / new_pdf_name
        shutil.move(str(filepath), str(new_path))

        images_folder = category_folder / "ReceiptImage"
        images_folder.mkdir(parents=True, exist_ok=True)

        try:  # Extract pages as images
            from doctr.io import DocumentFile
            doc = DocumentFile.from_pdf(str(new_path))
            page_images = doc.pages
        except Exception:
            page_images = []

        for img, fields in zip(page_images, fields_list):
            safe_vendor = re.sub(r"[^A-Za-z0-9]+", "_", fields.vendor).strip("_") or "receipt"
            date_fmt = _parse_date(fields.date) or "unknown"
            total_val = fields.total if fields.total is not None else 0
            total_str = f"{total_val:.2f}".replace(".", "_")
            img_name = f"{safe_vendor}_{date_fmt}_{total_str}.jpg"
            img_path = images_folder / img_name
            try:
                from PIL import Image

                Image.fromarray(img).save(img_path)
            except Exception:
                pass
            fields.image_path = img_path
    else:
        vendor = fields_list[0].vendor if fields_list else "receipt"
        date_val = fields_list[0].date if fields_list else ""
        filepath = rename_receipt_file(filepath, vendor, date_val)
        new_path = category_folder / filepath.name
        shutil.move(str(filepath), str(new_path))
        if fields_list:
            fields_list[0].image_path = new_path

    for f in fields_list:
        receipt_data = asdict(f)
        push_to_firefly(receipt_data)
        push_to_google_sheets(receipt_data)
        push_to_sharepoint(receipt_data)

    return fields_list, new_path


def _append_to_log(
    receipt_rows: List[dict[str, object]],
    item_rows: List[dict[str, object]],
) -> None:
    """Append new receipt and line-item rows to the Excel log."""
    if LOG_FILE.exists():
        with pd.ExcelFile(LOG_FILE) as xls:
            receipts_df = pd.read_excel(xls, sheet_name=0)
            if LINE_ITEMS_SHEET in xls.sheet_names:
                items_df = pd.read_excel(xls, sheet_name=LINE_ITEMS_SHEET)
            else:
                items_df = pd.DataFrame(columns=LINE_ITEM_COLUMNS)
    else:
        receipts_df = pd.DataFrame(columns=RECEIPT_COLUMNS)
        items_df = pd.DataFrame(columns=LINE_ITEM_COLUMNS)

    if receipt_rows:
        new_receipts = pd.DataFrame(receipt_rows)[RECEIPT_COLUMNS]
        receipts_df = pd.concat([receipts_df, new_receipts], ignore_index=True)

    if item_rows:
        new_items = pd.DataFrame(item_rows)[LINE_ITEM_COLUMNS]
        items_df = pd.concat([items_df, new_items], ignore_index=True)

    with pd.ExcelWriter(LOG_FILE, engine="openpyxl", mode="w") as writer:
        receipts_df.to_excel(writer, index=False, sheet_name="Sheet1")
        items_df.to_excel(writer, index=False, sheet_name=LINE_ITEMS_SHEET)


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
            item_records: list[dict[str, object]] = []
            for idx, fields in enumerate(page_fields):
                receipt_id = final_path.stem if len(page_fields) == 1 else f"{final_path.stem}_{idx+1}"
                record = {
                    "receipt_id": receipt_id,
                    "date": fields.date,
                    "vendor": fields.vendor,
                    "subtotal": fields.subtotal,
                    "tax": fields.tax,
                    "total": fields.total,
                    "category": fields.category,
                    "payment_method": fields.payment_method,
                    "card_last4": fields.card_last4,
                    "line_items": json.dumps(fields.line_items) if fields.line_items else "",
                    "filename": final_path.name,
                    "processed_time": datetime.now().isoformat(),
                    "Receipt_Img": _image_hyperlink(fields.image_path),
                }
                records.append(record)

                for item in fields.line_items:
                    item_records.append(
                        {
                            "receipt_id": receipt_id,
                            "date": fields.date,
                            "vendor": fields.vendor,
                            "item_description": item.get("item_description"),
                            "item_price": item.get("price"),
                            "quantity": item.get("quantity"),
                            "taxable": item.get("tax", False),
                            "category": item.get("category"),
                            "image_link": _image_hyperlink(fields.image_path),
                        }
                    )

            if records or item_records:
                _append_to_log(records, item_records)
        except Exception as e:  # pragma: no cover - runtime protection
            print(f"Error processing {filepath.name}: {e}")


def run_batch() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    item_records: list[dict[str, object]] = []
    for file in INPUT_DIR.glob("*"):
        if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".pdf"}:
            try:
                page_fields, final_path = process_receipt_pages(file)
                for idx, fields in enumerate(page_fields):
                    receipt_id = final_path.stem if len(page_fields) == 1 else f"{final_path.stem}_{idx+1}"
                    record = {
                        "receipt_id": receipt_id,
                        "date": fields.date,
                        "vendor": fields.vendor,
                        "subtotal": fields.subtotal,
                        "tax": fields.tax,
                        "total": fields.total,
                        "category": fields.category,
                        "payment_method": fields.payment_method,
                        "card_last4": fields.card_last4,
                        "line_items": json.dumps(fields.line_items) if fields.line_items else "",
                        "filename": final_path.name,
                        "processed_time": datetime.now().isoformat(),
                        "Receipt_Img": _image_hyperlink(fields.image_path),
                    }
                    records.append(record)

                    for item in fields.line_items:
                        item_records.append(
                            {
                                "receipt_id": receipt_id,
                                "date": fields.date,
                                "vendor": fields.vendor,
                                "item_description": item.get("item_description"),
                                "item_price": item.get("price"),
                                "quantity": item.get("quantity"),
                                "taxable": item.get("tax", False),
                                "category": item.get("category"),
                                "image_link": _image_hyperlink(fields.image_path),
                            }
                        )
            except Exception as e:  # pragma: no cover - runtime protection
                print(f"Error: {file.name} - {e}")

    if records or item_records:
        _append_to_log(records, item_records)


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
