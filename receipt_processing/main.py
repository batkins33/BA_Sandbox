"""Receipt Processing App using DocTR (OCR)."""

from __future__ import annotations

import json
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from multiprocessing import Lock
from pathlib import Path
from typing import Iterable, List

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import pandas as pd

from config import CONFIG
from integrations import (
    push_to_firefly,
    push_to_google_sheets,
    push_to_sharepoint,
)
from ml_categorizer import load_model, predict_category
from utils import (
    CATEGORY_MAP,
    ReceiptFields,
    compute_confidence_score,
    extract_fields,
    load_vendor_categories,
    compute_receipt_signature,
)

try:  # Optional dependency for image processing
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional
    cv2 = None
    np = None


# --- Configuration ---
INPUT_DIR = CONFIG["input_dir"]
OUTPUT_DIR = CONFIG["output_dir"]
LOG_FILE = CONFIG["log_file"]
LINE_ITEMS_SHEET = CONFIG["line_items_sheet"]

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
    "confidence_score",
    "line_items",
    "filename",
    "processed_time",
    "signature",
    "Receipt_Img",
    "Total_Img",
    "CardLast4_Img",
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
AUTO_CROP_ENABLED = CONFIG["auto_crop_enabled"]
AUTO_ORIENT_ENABLED = CONFIG["auto_orient_enabled"]

# Optional vendor categorization via CSV mapping
USE_VENDOR_CSV = CONFIG["use_vendor_csv"]
VENDOR_CSV_PATH = CONFIG.get("vendor_csv_path")
VENDOR_MAP = (
    load_vendor_categories(VENDOR_CSV_PATH)
    if USE_VENDOR_CSV and VENDOR_CSV_PATH and VENDOR_CSV_PATH.exists()
    else None
)
ML_MODEL = load_model()

LOW_CONFIDENCE_THRESHOLD = CONFIG["low_confidence_threshold"]
LOW_CONFIDENCE_LOG = CONFIG["low_confidence_log"]
LOG_LOCK = Lock()
CROP_PAD = CONFIG["cropping_pad"]



# --- Lazy OCR initialization ---
def _get_ocr_model():
    from doctr.models import ocr_predictor
    return ocr_predictor(
        det_arch="db_resnet34",  # <-- Use db_resnet34 (commonly available)
        reco_arch="crnn_mobilenet_v3_small",
        pretrained=True,
    )



# --- Utility: OCR processing ---
def extract_text_pages(
    filepath: Path,
) -> tuple[List[List[str]], List[List[tuple[str, tuple[float, float, float, float]]]]]:
    """Return OCR text and word boxes for each page of an image or PDF."""
    from doctr.io import DocumentFile

    if filepath.suffix.lower() == ".pdf":
        doc = DocumentFile.from_pdf(str(filepath))
    else:
        doc = DocumentFile.from_images([str(filepath)])

    model = _get_ocr_model()
    result = model(doc)
    export = result.export()
    pages: List[List[str]] = []
    boxes: List[List[tuple[str, tuple[float, float, float, float]]]] = []
    for page in export["pages"]:
        lines: List[str] = []
        word_boxes: List[tuple[str, tuple[float, float, float, float]]] = []
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                words = line.get("words", [])
                text = "".join(w["value"] for w in words)
                lines.append(text)
                for w in words:
                    geom = w.get("geometry", [0, 0, 1, 1])
                    word_boxes.append((w["value"], tuple(geom)))
        pages.append(lines)
        boxes.append(word_boxes)
    return pages, boxes


def extract_text(filepath: Path) -> Iterable[str]:
    """Run OCR on an image or PDF and return all text lines."""
    pages, _ = extract_text_pages(filepath)
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
    pad = CROP_PAD
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
    """Detect skew and correct perspective before OCR.

    The function first estimates the dominant rotation angle using
    ``cv2.HoughLines`` on Canny edges.  The image is rotated to make text
    horizontal.  It then searches for a large four-point contour and, if
    found, performs a perspective transform to deskew angled photos.
    """

    if cv2 is None or np is None:
        return image_path

    img = cv2.imread(str(image_path))
    if img is None:
        return image_path

    # --- Step 1: rotation via Hough line angle ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 200)
    angle = 0.0
    if lines is not None and len(lines) > 0:
        angles = [(theta * 180.0 / np.pi) - 90 for rho, theta in lines[:, 0]]
        angle = float(np.median(angles))
        if abs(angle) > 0.1:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(
                img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

    # --- Step 2: perspective correction using contour ---
    def _order_points(pts: "np.ndarray") -> "np.ndarray":
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        if cnts:
            biggest = max(cnts, key=cv2.contourArea)
            peri = cv2.arcLength(biggest, True)
            approx = cv2.approxPolyDP(biggest, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype("float32")
                rect = _order_points(pts)
                (tl, tr, br, bl) = rect
                widthA = np.linalg.norm(br - bl)
                widthB = np.linalg.norm(tr - tl)
                maxW = int(max(widthA, widthB))
                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)
                maxH = int(max(heightA, heightB))
                dst = np.array(
                    [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
                    dtype="float32",
                )
                M = cv2.getPerspectiveTransform(rect, dst)
                img = cv2.warpPerspective(img, M, (maxW, maxH))
    except Exception:
        pass

    cv2.imwrite(str(image_path), img)
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


def _save_field_crop(
    image_path: Path, box: tuple[float, float, float, float], out_path: Path
) -> None:
    """Crop ``image_path`` to ``box`` and save to ``out_path``."""
    from PIL import Image

    img = Image.open(image_path)
    width, height = img.size
    x1, y1, x2, y2 = box
    left = max(int(x1 * width) - 5, 0)
    top = max(int(y1 * height) - 5, 0)
    right = min(int(x2 * width) + 5, width)
    bottom = min(int(y2 * height) + 5, height)
    crop = img.crop((left, top, right, bottom))
    crop.save(out_path)


def _find_field_box(
    words: list[tuple[str, tuple[float, float, float, float]]],
    patterns: list[str],
) -> tuple[float, float, float, float] | None:
    """Return bounding box for the first word matching ``patterns``."""
    for idx, (text, box) in enumerate(words):
        if any(re.search(pat, text, re.I) for pat in patterns):
            x1, y1, x2, y2 = box
            if idx + 1 < len(words):
                nxt_text, nxt_box = words[idx + 1]
                if re.search(r"\d", nxt_text):
                    x2 = max(x2, nxt_box[2])
                    y1 = min(y1, nxt_box[1])
                    y2 = max(y2, nxt_box[3])
            return x1, y1, x2, y2
    return None


def _find_card_box(
    words: list[tuple[str, tuple[float, float, float, float]]], last4: str | None
) -> tuple[float, float, float, float] | None:
    """Return bounding box for the ``last4`` digits."""
    if not last4:
        return None
    pat = re.compile(re.escape(last4))
    for text, box in words:
        if pat.search(text):
            return box
    return None


def crop_field_images(
    image_path: Path,
    words: list[tuple[str, tuple[float, float, float, float]]],
    fields: ReceiptFields,
    folder: Path,
) -> dict[str, Path]:
    """Crop key fields from ``image_path`` based on OCR ``words``."""
    folder.mkdir(parents=True, exist_ok=True)
    crops: dict[str, Path] = {}

    total_box = _find_field_box(words, [r"total", r"amount due", r"balance due"])
    if total_box:
        out = folder / f"{image_path.stem}_total.jpg"
        _save_field_crop(image_path, total_box, out)
        crops["Total"] = out

    card_box = _find_card_box(words, fields.card_last4)
    if card_box:
        out = folder / f"{image_path.stem}_cardlast4.jpg"
        _save_field_crop(image_path, card_box, out)
        crops["CardLast4"] = out

    return crops


def _log_low_confidence(path: Path, score: float, page: int | None = None) -> None:
    """Log receipts with low confidence scores for later review."""
    details = path.name
    if page is not None:
        details = f"{details} [page {page}]"
    message = f"{datetime.now().isoformat()} - {details} score={score:.2f}"
    print(f"[LOW CONFIDENCE] {message}")
    with LOG_LOCK:
        with open(LOW_CONFIDENCE_LOG, "a", encoding="utf-8") as fh:
            fh.write(message + "\n")


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

    fields.confidence_score = compute_confidence_score(fields)

    filepath = rename_receipt_file(filepath, fields.vendor, fields.date)

    category_folder = OUTPUT_DIR / fields.category
    category_folder.mkdir(parents=True, exist_ok=True)
    new_path = category_folder / filepath.name
    shutil.move(str(filepath), str(new_path))

    fields.image_path = new_path
    if fields.confidence_score < LOW_CONFIDENCE_THRESHOLD:
        _log_low_confidence(new_path, fields.confidence_score)

    receipt_data = asdict(fields)
    push_to_firefly(receipt_data)
    push_to_google_sheets(receipt_data)
    push_to_sharepoint(receipt_data)

    return fields


def process_receipt_pages(filepath: Path) -> tuple[List[ReceiptFields], Path]:
    """Process a multi-page PDF and return fields for each page."""

    if filepath.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        filepath = preprocess_image(filepath)

    pages, word_boxes = extract_text_pages(filepath)
    fields_list: List[ReceiptFields] = []
    for lines in pages:
        fields = extract_fields(lines, CATEGORY_MAP, VENDOR_MAP)
        if ML_MODEL is not None:
            pred, conf = predict_category(
                fields.vendor, "\n".join(fields.lines), fields.total, model=ML_MODEL
            )
            if pred and conf >= 0.5:
                fields.category = pred
        fields.confidence_score = compute_confidence_score(fields)
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
        field_folder = category_folder / "FieldImages"
        field_folder.mkdir(parents=True, exist_ok=True)

        try:  # Extract pages as images
            from doctr.io import DocumentFile
            doc = DocumentFile.from_pdf(str(new_path))
            page_images = doc.pages
        except Exception:
            page_images = []

        for img, fields, boxes in zip(page_images, fields_list, word_boxes):
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
            fields.field_images = crop_field_images(img_path, boxes, fields, field_folder)
    else:
        vendor = fields_list[0].vendor if fields_list else "receipt"
        date_val = fields_list[0].date if fields_list else ""
        filepath = rename_receipt_file(filepath, vendor, date_val)
        new_path = category_folder / filepath.name
        shutil.move(str(filepath), str(new_path))
        if fields_list:
            fields_list[0].image_path = new_path
            field_folder = category_folder / "FieldImages"
            field_folder.mkdir(parents=True, exist_ok=True)
            fields_list[0].field_images = crop_field_images(
                new_path, word_boxes[0] if word_boxes else [], fields_list[0], field_folder
            )

    for idx, f in enumerate(fields_list, start=1):
        if f.confidence_score < LOW_CONFIDENCE_THRESHOLD:
            _log_low_confidence(new_path, f.confidence_score, page=idx if len(fields_list) > 1 else None)
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
    with LOG_LOCK:
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

        kept_ids: set[str] = set()
        if receipt_rows:
            if "confidence_score" not in receipts_df.columns:
                receipts_df["confidence_score"] = None
            for col in ("Total_Img", "CardLast4_Img", "signature"):
                if col not in receipts_df.columns:
                    receipts_df[col] = ""
            receipts_df["signature"] = receipts_df.apply(
                lambda r: compute_receipt_signature(
                    str(r.get("vendor", "")), str(r.get("date", "")), r.get("total")
                ),
                axis=1,
            )
            existing = set(receipts_df["signature"].dropna().astype(str))
            filtered: list[dict[str, object]] = []
            for row in receipt_rows:
                sig = compute_receipt_signature(
                    str(row.get("vendor", "")), str(row.get("date", "")), row.get("total")
                )
                if sig in existing:
                    continue
                row["signature"] = sig
                existing.add(sig)
                kept_ids.add(str(row.get("receipt_id")))
                filtered.append(row)
            if filtered:
                new_receipts = pd.DataFrame(filtered)[RECEIPT_COLUMNS]
                receipts_df = pd.concat([receipts_df, new_receipts], ignore_index=True)

        if item_rows and kept_ids:
            item_rows = [r for r in item_rows if str(r.get("receipt_id")) in kept_ids]
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
                    "confidence_score": fields.confidence_score,
                    "line_items": json.dumps(fields.line_items) if fields.line_items else "",
                    "filename": final_path.name,
                    "processed_time": datetime.now().isoformat(),
                    "signature": compute_receipt_signature(
                        fields.vendor, fields.date, fields.total
                    ),
                    "Receipt_Img": _image_hyperlink(fields.image_path),
                    "Total_Img": _image_hyperlink(
                        fields.field_images.get("Total") if fields.field_images else None
                    ),
                    "CardLast4_Img": _image_hyperlink(
                        fields.field_images.get("CardLast4") if fields.field_images else None
                    ),
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
                            "taxable": item.get("taxable", False),
                            "category": item.get("category"),
                            "image_link": _image_hyperlink(fields.image_path),
                        }
                    )

            if records or item_records:
                _append_to_log(records, item_records)
        except Exception as e:  # pragma: no cover - runtime protection
            print(f"Error processing {filepath.name}: {e}")


def run_batch(parallel: bool = True) -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    item_records: list[dict[str, object]] = []

    def _collect(page_fields: List[ReceiptFields], final_path: Path) -> None:
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
                "confidence_score": fields.confidence_score,
                "line_items": json.dumps(fields.line_items) if fields.line_items else "",
                "filename": final_path.name,
                "processed_time": datetime.now().isoformat(),
                "signature": compute_receipt_signature(
                    fields.vendor, fields.date, fields.total
                ),
                "Receipt_Img": _image_hyperlink(fields.image_path),
                "Total_Img": _image_hyperlink(
                    fields.field_images.get("Total") if fields.field_images else None
                ),
                "CardLast4_Img": _image_hyperlink(
                    fields.field_images.get("CardLast4") if fields.field_images else None
                ),
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
                        "taxable": item.get("taxable", False),
                        "category": item.get("category"),
                        "image_link": _image_hyperlink(fields.image_path),
                    }
                )

    files = [
        f
        for f in INPUT_DIR.glob("*")
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".pdf"}
    ]

    if parallel and files:
        with ProcessPoolExecutor() as executor:
            future_to_file = {
                executor.submit(process_receipt_pages, f): f for f in files
            }
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    page_fields, final_path = future.result()
                    _collect(page_fields, final_path)
                except Exception as e:  # pragma: no cover - runtime protection
                    print(f"Error: {file.name} - {e}")
    else:
        for file in files:
            try:
                page_fields, final_path = process_receipt_pages(file)
                _collect(page_fields, final_path)
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
