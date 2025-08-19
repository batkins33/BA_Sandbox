# extract_invoice.py

# ----- User-defined paths -----
INPUT_PATH = "path/to/yourfile.pdf"    # or "path/to/pdf_directory"
OUTPUT_PATH = "output.xlsx"
PAGE_WORKERS = 4  # You can change number of threads here

# ----- Script starts here -----
import os
import fitz
import numpy as np
import pandas as pd
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from concurrent.futures import ThreadPoolExecutor

def extract_text_from_pdf(pdf_path, page_workers=4):
    doc = DocumentFile.from_pdf(pdf_path)
    model = ocr_predictor(pretrained=True)
    result = model(doc)
    all_text = []
    for page in result.pages:
        page_text = []
        for block in page.blocks:
            for line in block.lines:
                line_text = ' '.join([w.value for w in line.words])
                page_text.append(line_text)
        all_text.append('\n'.join(page_text))
    return all_text

def extract_from_single_file(pdf_path, output_path, page_workers=4):
    print(f"Processing: {pdf_path}")
    try:
        text_pages = extract_text_from_pdf(pdf_path, page_workers)
        df = pd.DataFrame({'page': list(range(1, len(text_pages)+1)), 'text': text_pages})
        df.to_excel(output_path, index=False)
        print(f"Exported to {output_path}")
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

def extract_from_directory(pdf_dir, output_path, page_workers=4):
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    records = []
    for pdf_path in pdf_files:
        try:
            text_pages = extract_text_from_pdf(pdf_path, page_workers)
            for i, text in enumerate(text_pages):
                records.append({'file': os.path.basename(pdf_path), 'page': i+1, 'text': text})
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    df = pd.DataFrame(records)
    df.to_excel(output_path, index=False)
    print(f"Exported to {output_path}")

def main():
    if os.path.isdir(INPUT_PATH):
        extract_from_directory(INPUT_PATH, OUTPUT_PATH, PAGE_WORKERS)
    elif os.path.isfile(INPUT_PATH):
        extract_from_single_file(INPUT_PATH, OUTPUT_PATH, PAGE_WORKERS)
    else:
        print("Input path does not exist.")

if __name__ == "__main__":
    main()
