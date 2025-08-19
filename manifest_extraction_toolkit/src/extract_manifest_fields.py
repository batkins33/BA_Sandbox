"""Anchor-based OCR pipeline for manifest field extraction.

Replaces the "full-page OCR + regex" approach with precise ROI extraction
using anchor-based positioning for structured field extraction.
"""

from __future__ import annotations
import re
import io
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from pytesseract import Output
import pandas as pd

logger = logging.getLogger(__name__)

# ---------- helpers

def render_page(doc, pno: int, dpi: int = 300) -> Image.Image:
    page = doc[pno]
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes(output="png")))

def ocr_words(img: Image.Image) -> pd.DataFrame:
    df = pytesseract.image_to_data(img, output_type=Output.DATAFRAME)
    df = df.dropna(subset=["text"]).copy()
    df["right"] = df["left"] + df["width"]
    df["bottom"] = df["top"] + df["height"]
    return df

def find_line_anchor(words: pd.DataFrame, needed: List[str], y_tolerance: int = 10) -> Optional[Tuple[int,int,int,int]]:
    """
    Find a line that contains all 'needed' tokens (case-insensitive).
    Returns (x0,y0,x1,y1) of the union bbox of those tokens on that line.
    """
    # group by line markers (page_num, block_num, par_num, line_num)
    grp_cols = ["page_num","block_num","par_num","line_num"]
    for _, line in words.groupby(grp_cols):
        tokens = " ".join(line["text"].astype(str).tolist()).lower()
        if all(tok.lower() in tokens for tok in needed):
            # union bbox
            x0 = int(line["left"].min())
            y0 = int(line["top"].min())
            x1 = int(line["right"].max())
            y1 = int(line["bottom"].max())
            return (x0,y0,x1,y1)
    return None

def crop(img: Image.Image, box: Tuple[int,int,int,int]) -> Image.Image:
    # ensure box in bounds
    w,h = img.size
    x0,y0,x1,y1 = box
    x0 = max(0, x0); y0 = max(0,y0); x1 = min(w, x1); y1 = min(h, y1)
    return img.crop((x0,y0,x1,y1))

def binarize_for_handwriting(im: Image.Image) -> Image.Image:
    # Strong contrast/denoise for handwriting
    g = ImageOps.grayscale(im)
    g = g.filter(ImageFilter.MedianFilter(size=3))
    # Adaptive threshold-ish via point mapping
    return g.point(lambda p: 255 if p > 190 else 0)

def ocr_text(im: Image.Image, digits_only=False, single_line=False) -> str:
    cfg = []
    if digits_only:
        cfg.append("tessedit_char_whitelist=0123456789")
    # PSM: 7 single line, 6 block of text
    psm = "7" if single_line else "6"
    return pytesseract.image_to_string(
        im, config=f"--oem 3 --psm {psm} " + ("-c " + " ".join(cfg) if cfg else "")
    ).strip()

# ---------- field specs (anchors and ROI recipes)

@dataclass
class FieldSpec:
    name: str
    anchor_tokens: List[str]
    roi_from_anchor: callable  # (img, anchor_bbox)->(x0,y0,x1,y1)
    post: callable             # cleaning/validation

def right_of(anchor, dx=10, w=220, dy=-5, h=36):
    # small box to the right of a label row
    ax0,ay0,ax1,ay1 = anchor
    return (ax1+dx, ay0+dy, ax1+dx+w, ay0+dy+h)

def below_band(anchor, img, top_pad=8, bottom=70, left=0.08, right=0.62):
    # a left column band beneath the label line
    W,H = img.size
    ax0,ay0,ax1,ay1 = anchor
    y0 = ay1 + top_pad
    y1 = min(H, y0 + bottom)
    return (int(W*left), y0, int(W*right), y1)

_manifest_re = re.compile(r"^\d{8}$")

def clean_manifest(s: str) -> str:
    """Validate manifest number as exactly 8 digits."""
    s = re.sub(r"[^\d]", "", s)
    return s if _manifest_re.match(s) else ""

plate_re = re.compile(r"[A-Z0-9]{2,8}(?:-[A-Z0-9]{1,4})?$", re.I)
dot_num_re = re.compile(r"(?:US\s*DOT|USDOT|DOT)\D*(\d{4,9})", re.I)

def clean_plate(s: str) -> str:
    """Normalize license plate to uppercase, allow A-Z0-9 and optional hyphen."""
    s = s.replace(" ", "").upper()
    cand = plate_re.findall(s)
    return cand[0] if cand else s[:10]

def clean_dot(s: str) -> str:
    """Extract DOT ID number, prefer USDOT format."""
    m = dot_num_re.search(s)
    if m: 
        return m.group(1)
    # fallback: take a 5–9 digit run
    m2 = re.search(r"\b\d{5,9}\b", s)
    return m2.group(0) if m2 else s.strip()

def normalize_phone(s: str) -> str:
    """Normalize phone to (xxx) xxx-xxxx format when possible."""
    digits = re.sub(r"[^\d]", "", s)
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return s.strip()

def pass_through(s: str) -> str:
    """Basic text cleaning - normalize whitespace."""
    return " ".join(s.split())

# Manifest Number (A.)
FIELD_SPECS = [
    FieldSpec(
        name="manifest_number",
        anchor_tokens=["manifest","number"],
        roi_from_anchor=lambda img, anc: right_of(anc, dx=12, w=200, dy=-2, h=40),
        post=clean_manifest,
    ),
    # Transporter 1
    FieldSpec(
        name="t1_company",
        anchor_tokens=["transporter","1","company","name"],
        roi_from_anchor=lambda img, anc: below_band(anc, img, top_pad=8, bottom=80, left=0.08, right=0.62),
        post=pass_through,
    ),
    FieldSpec(
        name="t1_us_epa_id",
        anchor_tokens=["us","epa","id","number"],
        roi_from_anchor=lambda img, anc: right_of(anc, dx=8, w=220, dy=4, h=32),
        post=pass_through,
    ),
    FieldSpec(
        name="t1_state_id",
        anchor_tokens=["state","id"],
        roi_from_anchor=lambda img, anc: right_of(anc, dx=8, w=220, dy=4, h=32),
        post=pass_through,
    ),
    FieldSpec(
        name="t1_phone",
        anchor_tokens=["phone"],
        roi_from_anchor=lambda img, anc: right_of(anc, dx=8, w=220, dy=4, h=32),
        post=normalize_phone,
    ),
    # Transporter 2
    FieldSpec(
        name="t2_company",
        anchor_tokens=["transporter","2","company","name"],
        roi_from_anchor=lambda img, anc: below_band(anc, img, top_pad=8, bottom=80, left=0.08, right=0.62),
        post=pass_through,
    ),
    FieldSpec(
        name="t2_us_epa_id",
        anchor_tokens=["us","epa","id","number"],
        roi_from_anchor=lambda img, anc: right_of(anc, dx=8, w=220, dy=4, h=32),
        post=pass_through,
    ),
    FieldSpec(
        name="t2_state_id",
        anchor_tokens=["state","id"],
        roi_from_anchor=lambda img, anc: right_of(anc, dx=8, w=220, dy=4, h=32),
        post=pass_through,
    ),
    FieldSpec(
        name="t2_phone",
        anchor_tokens=["phone"],
        roi_from_anchor=lambda img, anc: right_of(anc, dx=8, w=220, dy=4, h=32),
        post=normalize_phone,
    ),
]

def extract_handwritten_bottom(img: Image.Image) -> Dict[str,str]:
    """
    Split bottom band into 3 columns and OCR each for handwritten:
    license_plate, dot_id, truck_number
    """
    W,H = img.size
    # Take bottom 18% of page height
    band = crop(img, (int(W*0.05), int(H*0.82), int(W*0.95), int(H*0.97)))
    # Column splits by thirds (tune if your form differs)
    cols = []
    for i in range(3):
        x0 = int(band.width * (i/3))
        x1 = int(band.width * ((i+1)/3))
        cols.append(band.crop((x0, 0, x1, band.height)))

    # preprocess for handwriting
    cols = [binarize_for_handwriting(c) for c in cols]

    plate_txt = ocr_text(cols[0], single_line=True)
    dot_txt   = ocr_text(cols[1], single_line=True)
    truck_txt = ocr_text(cols[2], single_line=True)

    return {
        "license_plate": clean_plate(plate_txt),
        "dot_id": clean_dot(dot_txt),
        "truck_number": pass_through(truck_txt),
    }

def extract_fields_from_page(img: Image.Image) -> Dict[str, str]:
    """Extract all fields from a single page using anchor-based ROI extraction."""
    words = ocr_words(img)
    out = {}

    for spec in FIELD_SPECS:
        anc = find_line_anchor(words, spec.anchor_tokens)
        if not anc:
            logger.warning(f"Anchor not found for {spec.name}: {spec.anchor_tokens}")
            out[spec.name] = ""
            continue
        
        roi = spec.roi_from_anchor(img, anc)
        crop_im = crop(img, roi)
        digits_only = spec.name == "manifest_number"
        single_line = spec.name.startswith(("t1_us_epa_id","t2_us_epa_id","t1_state_id","t2_state_id","t1_phone","t2_phone","manifest_number"))
        
        try:
            raw_txt = ocr_text(crop_im, digits_only=digits_only, single_line=single_line)
            cleaned = spec.post(raw_txt)
            out[spec.name] = cleaned
            
            # Log validation failures
            if not cleaned and raw_txt:
                logger.warning(f"Validation failed for {spec.name}: '{raw_txt}'")
                
        except Exception as e:
            logger.error(f"OCR failed for {spec.name}: {e}")
            out[spec.name] = ""

    out.update(extract_handwritten_bottom(img))
    return out

def process_pdf(pdf_path: Path) -> Dict[str, str]:
    """Process PDF and extract manifest fields from first valid page."""
    try:
        with fitz.open(pdf_path) as doc:
            # Try first 3 pages for anchors
            for pno in range(min(3, len(doc))):
                img = render_page(doc, pno, dpi=300)
                fields = extract_fields_from_page(img)
                
                # Validate manifest number position (should be in top 25% of page)
                if fields.get("manifest_number"):
                    fields["source_file"] = pdf_path.name
                    fields["page"] = pno + 1
                    return fields
            
            # Fallback to page 1 if no manifest anchor found
            img = render_page(doc, 0, dpi=300)
            fields = extract_fields_from_page(img)
            fields["source_file"] = pdf_path.name
            fields["page"] = 1
            logger.warning(f"No manifest anchor found in {pdf_path.name}, using page 1 fallback")
            return fields
            
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}")
        return {"source_file": pdf_path.name, "page": 1, "error": str(e)}

def main(src_dir: str, out_xlsx: str):
    """Main function to process directory of PDFs and extract manifest fields."""
    src_path = Path(src_dir)
    pdfs = sorted(src_path.rglob("*.pdf"))
    
    if not pdfs:
        logger.warning(f"No PDF files found in {src_dir}")
        return
    
    logger.info(f"Processing {len(pdfs)} PDF files...")
    
    rows = []
    for p in pdfs:
        logger.info(f"Processing {p.name}")
        result = process_pdf(p)
        rows.append(result)
    
    # Create DataFrame with specified column order
    columns = [
        "source_file", "page", "manifest_number",
        "t1_company", "t1_us_epa_id", "t1_state_id", "t1_phone",
        "t2_company", "t2_us_epa_id", "t2_state_id", "t2_phone",
        "license_plate", "dot_id", "truck_number"
    ]
    
    df = pd.DataFrame(rows)
    # Reorder columns, add missing ones as empty
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    df = df[columns + [c for c in df.columns if c not in columns]]
    
    # Save results
    output_path = Path(out_xlsx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    
    logger.info(f"Results saved to {output_path}")
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract manifest fields using anchor-based OCR"
    )
    parser.add_argument("src", help="Source directory containing PDF files (recurses)")
    parser.add_argument("--out", default="manifest_fields.xlsx", help="Output Excel file")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    main(args.src, args.out)
