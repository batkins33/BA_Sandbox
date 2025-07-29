import argparse
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import re
from doctr.models import ocr_predictor
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=250, colorspace=fitz.csRGB)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[..., :3]
        pages.append(img)

    texts = [None] * len(pages)
    with ThreadPoolExecutor(max_workers=page_workers) as executor:
        futures = [executor.submit(ocr_page, model, idx, img) for idx, img in enumerate(pages)]
        for fut in as_completed(futures):
            idx, text = fut.result()
            texts[idx] = text

    full_text = "\n".join(texts)
    records = []
    for match in REGEX_PATTERN.finditer(full_text):
        groups = match.groupdict()
        records.append({
            "File": path.name,
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


if __name__ == "__main__":
    main()
