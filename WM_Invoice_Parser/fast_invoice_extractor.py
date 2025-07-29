
import argparse
from pathlib import Path
import logging
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import re
from doctr.models import ocr_predictor
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure simple logging to show progress information
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# --- USER SETTINGS ---
INPUT_PATH = Path(r"C:\Users\brian.atkins\OneDrive - Lindamood Demolition\24-105 PHMS NPC - Documents\PM\Invoices\invoices\A_P_Invoice_0060771-0399-9_10162024.pdf")
OUTPUT_PATH = Path(r"C:\Users\brian.atkins\OneDrive - Lindamood Demolition\24-105 PHMS NPC - Documents\PM\Invoices\invoices\output.xlsx")
PAGE_WORKERS = 16                               # Number of threads per PDF for OCR

# --- REGEX PATTERN ---
REGEX_PATTERN = re.compile(
    r"Vehicle[#:\s]*\s*(?P<vehicle>\S+).*?"
    r"Profile\s*#\s*(?P<profile>\S+).*?"
    r"Generator\s*(?P<generator>.*?)\s*Manifest[#:\s]*"
    r"(?P<manifest>\S+).*?"
    r"Date[:\s]*(?P<date>\d{2}/\d{2}/\d{2}).*?"
    r"ticket\s*(?:number)?\s*[:#]?\s*(?P<ticket>\d+).*?"
    r"Qty[:\s]*(?P<qty>[0-9.]+).*?"
    r"UoM[:\s]*(?P<uom>\w+).*?"
    r"Rate[:\s]*(?P<rate>[0-9.]+).*?"
    r"Ticket\s+Total[:\s]*(?P<ticket_total>[0-9.]+)",
    re.IGNORECASE | re.DOTALL,
)


def extract_lines(export: dict) -> str:
    """Flatten DocTR export into newline-separated text."""
    lines = []
    if isinstance(export, dict) and "blocks" in export:
        for block in export["blocks"]:
            for line in block.get("lines", []):
                words = [w.get("value", "") for w in line.get("words", [])]
                line_text = " ".join(words).strip()
                if line_text:
                    lines.append(line_text)
    return "\n".join(lines)

def ocr_page(model, idx, img):
    result = model([img])
    page = result.pages[0]
    return idx, extract_lines(page.export())


def process_pdf(path: Path, model, page_workers: int = 4):
    logging.info("Processing %s", path)
    doc = fitz.open(str(path))
    pages = []
    for page_idx, page in enumerate(doc):
        pix = page.get_pixmap(dpi=250, colorspace=fitz.csRGB)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[..., :3]
        pages.append(img)
    logging.info("%d page(s) loaded from %s", len(pages), path)

    texts = [None] * len(pages)
    with ThreadPoolExecutor(max_workers=page_workers) as executor:
        futures = [executor.submit(ocr_page, model, idx, img) for idx, img in enumerate(pages)]
        for fut in as_completed(futures):
            idx, text = fut.result()
            texts[idx] = text
            logging.info("OCR completed for page %d of %s", idx + 1, path)

    records = []
    for page_num, page_text in enumerate(texts, start=1):
        for match in REGEX_PATTERN.finditer(page_text):
            groups = match.groupdict()
            records.append({
                "File": path.name,
                "Page": page_num,
                "Vehicle": groups.get("vehicle", ""),
                "Date": groups.get("date", ""),
                "Ticket#": groups.get("ticket", ""),
                "Qty": groups.get("qty", ""),
                "Rate": groups.get("rate", ""),
                "Profile#": groups.get("profile", ""),
                "Generator": groups.get("generator", "").strip(),
                "Manifest#": groups.get("manifest", ""),
                "TicketTotal": groups.get("ticket_total", ""),
            })
    return records


def main():
    parser = argparse.ArgumentParser(description="Extract invoice loads using DocTR OCR")
    parser.add_argument("input", help="PDF file or directory of PDF files")
    parser.add_argument("output", help="Path to output Excel file")
    parser.add_argument("--page-workers", type=int, default=4, help="Threads per PDF for OCR")
    args = parser.parse_args()

    input_path = Path(args.input)
    model = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)

    all_records = []
    if input_path.is_file():
        all_records.extend(process_pdf(input_path, model, args.page_workers))
    else:
        pdfs = sorted(p for p in input_path.glob("*.pdf") if p.is_file())
        for pdf in pdfs:
            all_records.extend(process_pdf(pdf, model, args.page_workers))

    df = pd.DataFrame(all_records)
    df.to_excel(args.output, index=False)

def main():
    logging.info("Loading OCR model...")
    model = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)

    all_records = []
    if INPUT_PATH.is_file():
        all_records.extend(process_pdf(INPUT_PATH, model, PAGE_WORKERS))
    elif INPUT_PATH.is_dir():
        pdfs = sorted(p for p in INPUT_PATH.glob("*.pdf") if p.is_file())
        for pdf in pdfs:
            all_records.extend(process_pdf(pdf, model, PAGE_WORKERS))
    else:
        logging.error("INPUT_PATH not found: %s", INPUT_PATH)
        return

    df = pd.DataFrame(all_records)
    try:
        df.to_excel(OUTPUT_PATH, index=False)
    except ImportError as exc:
        logging.error("Failed to write Excel file: %s", exc)
        logging.error(
            "Ensure the 'openpyxl' package is installed to enable Excel export"
        )
        return
    logging.info("Extraction complete. Results saved to %s", OUTPUT_PATH)

if __name__ == "__main__":
    main()
