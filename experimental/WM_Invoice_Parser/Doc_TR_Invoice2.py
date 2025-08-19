import fitz  # PyMuPDF
import numpy as np
from doctr.models import ocr_predictor
import pandas as pd
import re

PDF_PATH = r"C:\Users\brian.atkins\OneDrive - Lindamood Demolition\24-105 PHMS NPC - Documents\PM\Invoices\invoices\A_P_Invoice_0060771-0399-9_10162024.pdf"

OCR_TEXT_PATH = "ocr_output.txt"
EXCEL_PATH = "wm_invoice_loads6.xlsx"

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

def extract_text_from_export(export):
    # Handles various DocTR export() versions
    if isinstance(export, dict):
        if "blocks" in export:
            return "\n".join([blk.get('value', '') for blk in export["blocks"]])
        elif "value" in export:
            return export["value"]
        else:
            return str(export)
    else:
        return str(export)

print("Running OCR...")
texts = []
for idx, img in enumerate(pages):
    try:
        result = model([img])
        if hasattr(result, "pages") and result.pages:
            if hasattr(result.pages[0], "export"):
                export = result.pages[0].export()
                text = extract_text_from_export(export)
                texts.append(text)
                print(f"OCR OK page {idx+1}")
            else:
                print(f"Page {idx+1} has no export()")
        else:
            print(f"OCR failed for page {idx+1}: no pages in result")
    except Exception as e:
        print(f"OCR failed for page {idx+1}: {e}")

full_text = "\n".join(texts)
with open(OCR_TEXT_PATH, "w", encoding="utf-8") as f:
    f.write(full_text)
print(f"OCR output written to {OCR_TEXT_PATH}")

# --- REGEX DATA EXTRACTION ---
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
        "Vehicle": groups["vehicle"],
        "Date": groups["date"],
        "Ticket#": groups.get("ticket", ""),
        "Qty": groups["qty"],
        "Rate": groups["rate"],
        "Amount": groups["amount"],
        "Profile#": groups["profile"],
        "Generator": groups["generator"].strip(),
        "Manifest#": groups["manifest"],
        "TicketTotal": groups["ticket_total"],
    })

print(f"Parsed {len(records)} ticket records.")

df = pd.DataFrame(records)
df.to_excel(EXCEL_PATH, index=False)
print(f"Saved results to {EXCEL_PATH}")
