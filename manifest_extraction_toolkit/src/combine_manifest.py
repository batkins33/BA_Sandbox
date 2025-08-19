import os
from pathlib import Path
from pypdf import PdfReader, PdfWriter
import pandas as pd
import re
from datetime import datetime

# === Configuration ===
SOURCE_DIR = Path(r"C:\Users\brian.atkins\OneDrive - Lindamood Demolition\Documents - Truck Tickets\General\24-105-PHMS New Pediatric Campus\MANIFEST\ToBeProcessed\Combined\2025-05")
DEST_DIR = Path(r"C:\Users\brian.atkins\OneDrive - Lindamood Demolition\Documents - Truck Tickets\General\24-105-PHMS New Pediatric Campus\MANIFEST\Combined")

def merge_pdfs_with_bookmarks(pdf_paths):
    writer = PdfWriter()
    log_entries = []
    page_index = 0

    for path in pdf_paths:
        reader = PdfReader(path)
        num_pages = len(reader.pages)
        file_name = path.name

        for page in reader.pages:
            writer.add_page(page)

        # Bookmark: pypdf>=3.9.0 uses add_outline_item(title, page_number)
        try:
            writer.add_outline_item(title=file_name, page_number=page_index)
        except TypeError:
            # For older versions fallback
            writer.add_outline_item(file_name, page_index)

        log_entries.append({
            'Original File': file_name,
            'Start Page': page_index + 1,
            'Page Count': num_pages
        })

        page_index += num_pages

    return writer, page_index, pd.DataFrame(log_entries)

def is_date_folder(folder_name):
    return re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", folder_name) is not None

def get_output_path(subdir_name, page_count):
    date_obj = datetime.strptime(subdir_name, "%m.%d.%Y")
    ym_folder = date_obj.strftime("%Y-%m")
    output_filename = f"24-105_{subdir_name}_manifest_combined_{page_count}.pdf"
    return DEST_DIR / ym_folder / output_filename

def get_log_path(output_pdf_path):
    return output_pdf_path.with_suffix(".xlsx")

for subdir in SOURCE_DIR.iterdir():
    if subdir.is_dir() and is_date_folder(subdir.name):
        pdf_files = sorted(
            [f for f in subdir.rglob("*.pdf") if f.is_file()],
            key=lambda f: f.stat().st_ctime
        )

        if not pdf_files:
            print(f"[SKIPPED] No PDFs found in {subdir}")
            continue

        print(f"[PROCESSING] Merging {len(pdf_files)} PDFs in {subdir.name} (oldest to newest)...")
        try:
            writer, page_count, log_df = merge_pdfs_with_bookmarks(pdf_files)
            output_path = get_output_path(subdir.name, page_count)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as out_file:
                writer.write(out_file)

            log_path = get_log_path(output_path)
            log_df.to_excel(log_path, index=False)

            print(f"[SAVED] {output_path} ({page_count} pages)")
            print(f"[LOG]   {log_path}")

        except Exception as e:
            print(f"[ERROR] Failed to process {subdir}: {e}")

