import pdfplumber
import pandas as pd
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from PIL import Image

# --- CONFIG ---
PDF_PATH = r"C:\Users\brian.atkins\OneDrive - Lindamood Demolition\24-105 PHMS NPC - Documents\PM\Invoices\invoices\A_P_Invoice_0060771-0399-9_10162024.pdf"
OUTPUT_XLSX = "wm_invoice_loads4.xlsx"
THREADS = 16  # Tune this to your CPU

# --- DocTR Model (init once, it's big!) ---
print("Loading DocTR OCR model...")
model = ocr_predictor(pretrained=True)

def page_to_image(page):
    pil_img = page.to_image(resolution=300).original
    # Convert to RGB just in case (DocTR expects this)
    return pil_img.convert("RGB")

def doctr_ocr_image(image):
    # DocTR expects a list of images for batch prediction
    result = model([image])
    # Extract all text in reading order
    words = []
    for block in result.pages[0].blocks:
        for line in block.lines:
            words.append(" ".join([w.value for w in line.words]))
    return "\n".join(words)

def ocr_pdf_page(page):
    img = page_to_image(page)
    text = doctr_ocr_image(img)
    return text

def main():
    print("Opening PDF...")
    all_text = []

    with pdfplumber.open(PDF_PATH) as pdf:
        print(f"Total pages: {len(pdf.pages)}")
        with ThreadPoolExecutor(max_workers=THREADS) as executor:
            future_to_idx = {executor.submit(ocr_pdf_page, page): idx for idx, page in enumerate(pdf.pages)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    text = future.result()
                    all_text.append((idx, text))
                    print(f"OCR complete for page {idx + 1}")
                except Exception as e:
                    print(f"Error OCRing page {idx + 1}: {e}")

    # Combine and sort by page order
    all_text.sort()
    full_text = "\n".join(t for _, t in all_text)

    # --- Save raw OCR text for debugging ---
    with open("debug_ocr_doctr.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    # --- Your Regex Parsing (unchanged) ---
    print("Parsing OCR output...")
    pattern = re.compile(
        r"Vehicle#:\s*(?P<vehicle>.*?)\s+(?P<date>\d{2}/\d{2}/\d{2})\s*\|?\s*(?P<ticket>\d+)?[\s\S]+?"
        r"Soil\s*-\s*Class\s*2\s*Non-Industrial\s*(?P<qty>[0-9.]+)\s*TON\s*(?P<rate>[0-9.]+)\s*(?P<amount>[0-9.]+)[\s\S]+?"
        r"Profile\s*#\s*(?P<profile>[^\s]+)[\s\S]+?"
        r"Generator\s*(?P<generator>.*?)\s*[0-9.]+[\s\S]+?"
        r"Manifest#:\s*(?P<manifest>[0-9A-Za-z]+)[\s\S]+?"
        r"Ticket Total\s*(?P<ticket_total>[0-9.]+)",
        re.MULTILINE
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
    df.to_excel(OUTPUT_XLSX, index=False)
    print(f"Saved results to {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()
