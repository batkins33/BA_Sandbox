import re
import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- CONFIG ----
PDF_PATH = r"C:\Users\brian.atkins\OneDrive - Lindamood Demolition\24-105 PHMS NPC - Documents\PM\Invoices\invoices\A_P_Invoice_0060771-0399-9_10162024.pdf"
OUTPUT_XLSX = "wm_invoice_loads2.xlsx"
THREADS = 16  # Increase to number of CPU cores for more speed

# ---- OCR Function ----
def ocr_pdf_page(page):
    # Convert pdfplumber page to PIL image
    img = page.to_image(resolution=300).original
    text = pytesseract.image_to_string(img)
    return text

# ---- Main ----
def main():
    all_text = []

    print("Opening PDF...")
    with pdfplumber.open(PDF_PATH) as pdf:
        print(f"Total pages: {len(pdf.pages)}")

        with ThreadPoolExecutor(max_workers=THREADS) as executor:
            future_to_page = {executor.submit(ocr_pdf_page, page): i for i, page in enumerate(pdf.pages)}
            for future in as_completed(future_to_page):
                i = future_to_page[future]
                try:
                    ocr_text = future.result()
                    all_text.append((i, ocr_text))
                    print(f"OCR done for page {i+1}")
                except Exception as e:
                    print(f"Error OCRing page {i+1}: {e}")

    # Sort by page number
    all_text.sort()
    full_text = "\n".join(text for _, text in all_text)

    # ---- REGEX Parsing ----
    print("Parsing OCR output...")
    records = []
    pattern = re.compile(
        r"Vehicle#:\s*(?P<vehicle>.*?)\s+(?P<date>\d{2}/\d{2}/\d{2})\s*\|?\s*(?P<ticket>\d+)?[\s\S]+?"
        r"Soil\s*-\s*Class\s*2\s*Non-Industrial\s*(?P<qty>[0-9.]+)\s*TON\s*(?P<rate>[0-9.]+)\s*(?P<amount>[0-9.]+)[\s\S]+?"
        r"Profile\s*#\s*(?P<profile>[^\s]+)[\s\S]+?"
        r"Generator\s*(?P<generator>.*?)\s*[0-9.]+[\s\S]+?"
        r"Manifest#:\s*(?P<manifest>[0-9A-Za-z]+)[\s\S]+?"
        r"Ticket Total\s*(?P<ticket_total>[0-9.]+)",
        re.MULTILINE
    )

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

    # ---- EXPORT ----
    df = pd.DataFrame(records)
    df.to_excel(OUTPUT_XLSX, index=False)
    print(f"Saved results to {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()
