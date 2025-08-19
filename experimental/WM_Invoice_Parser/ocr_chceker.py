
import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd

pdf_path = r"C:\Users\brian.atkins\OneDrive - Lindamood Demolition\24-105 PHMS NPC - Documents\PM\Invoices\invoices\A_P_Invoice_0060771-0399-9_10162024.pdf"
records = []

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        # Try extract text normally first
        text = page.extract_text()
        if not text or len(text.strip()) == 0:
            # Use OCR if blank
            img = page.to_image(resolution=300)
            ocr_text = pytesseract.image_to_string(img.original)
            print(f"OCR PAGE {i+1}:\n", ocr_text[:2000])
        else:
            print(f"TEXT PAGE {i+1}:\n", text[:2000])
