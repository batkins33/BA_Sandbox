import fitz  # PyMuPDF
import numpy as np
from doctr.models import ocr_predictor
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

PDF_PATH = r"C:\Users\brian.atkins\OneDrive - Lindamood Demolition\24-105 PHMS NPC - Documents\PM\Invoices\invoices\A_P_Invoice_0060771-0399-9_10162024.pdf"
OCR_TEXT_PATH = "ocr_output.txt"
EXCEL_PATH = "wm_invoice_loads8.xlsx"

print("Loading DocTR OCR model...")
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

print("Opening PDF...")
doc = fitz.open(PDF_PATH)
pages = []
for i in range(len(doc)):
    try:
        page = doc[i]
        pix = page.get_pixmap(dpi=250, colorspace=fitz.csRGB)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        if pix.n == 4:
            img = img[..., :3]
        pages.append(img)
        print(f"Rendered page {i+1}")
    except Exception as e:
        print(f"Error rendering page {i+1}: {e}")

def doctr_export_to_text(export):
    """Flattens a doctr export dict to simple lines of text (best for regex extraction)."""
    if isinstance(export, dict) and "blocks" in export:
        lines = []
        for block in export["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    if "words" in line:
                        line_text = " ".join(w["value"] for w in line["words"] if "value" in w)
                        if line_text.strip():
                            lines.append(line_text)
        return "\n".join(lines)
    return ""

def ocr_page(idx_img):
    idx, img = idx_img
    try:
        result = model([img])
        if hasattr(result, "pages") and result.pages:
            export = result.pages[0].export()
            text = doctr_export_to_text(export)
            return idx, text
        else:
            return idx, ""
    except Exception as e:
        print(f"OCR failed for page {idx+1}: {e}")
        return idx, ""

print("Running OCR with threading...")
texts = [None] * len(pages)
with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers to your CPU/GPU
    futures = [executor.submit(ocr_page, (idx, img)) for idx, img in enumerate(pages)]
    for future in as_completed(futures):
        idx, text = future.result()
        texts[idx] = text
        print(f"OCR completed for page {idx+1}")

full_text = "\n".join(texts)
with open(OCR_TEXT_PATH, "w", encoding="utf-8") as f:
    f.write(full_text)
print(f"OCR output written to {OCR_TEXT_PATH}")

# --- REGEX DATA EXTRACTION ---
print("Parsing OCR output...")
# Using DOTALL makes '.' match newlines so the pattern runs faster on the
# multi-line OCR text without relying on heavy '[\s\S]*?' constructs.
pattern = re.compile(
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
    re.IGNORECASE | re.DOTALL
)

records = []
for match in pattern.finditer(full_text):
    groups = match.groupdict()
    records.append({
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

print(f"Parsed {len(records)} ticket records.")

df = pd.DataFrame(records)
df.to_excel(EXCEL_PATH, index=False)
print(f"Saved results to {EXCEL_PATH}")
