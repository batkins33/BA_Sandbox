"""Utility functions for the receipt processing app."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Default categories mapped to vendor keywords
CATEGORY_MAP: dict[str, list[str]] = {
    "fuel": ["shell", "chevron", "exxon", "gas"],
    "meals": ["restaurant", "grill", "mcdonald", "subway", "burger"],
    "supplies": ["office depot", "staples", "lowes", "home depot"],
}


def load_vendor_categories(csv_path: str | Path) -> dict[str, str]:
    """Load a vendor to category mapping from a CSV file."""
    df = pd.read_csv(csv_path)
    mapping: dict[str, str] = {
        str(row["vendor"]).strip().lower(): str(row["category"]).strip()
        for _, row in df.iterrows()
        if not pd.isna(row.get("vendor")) and not pd.isna(row.get("category"))
    }
    return mapping


def vendor_csv_to_json(csv_path: str | Path, json_path: str | Path) -> None:
    """Convert vendor category CSV to JSON lookup file."""
    mapping = load_vendor_categories(csv_path)
    pd.Series(mapping).to_json(json_path)

@dataclass
class ReceiptFields:
    """Container for the key fields parsed from a receipt."""

    vendor: str
    date: str
    subtotal: Optional[float]
    tax: Optional[float]
    total: Optional[float]
    payment_method: Optional[str]
    card_last4: Optional[str]
    category: str
    lines: list[str]


def extract_fields(
    lines: Iterable[str],
    category_map: dict[str, list[str]] | None = None,
    vendor_lookup: dict[str, str] | None = None,
) -> ReceiptFields:
    """Extract key information from the OCR text lines.

    Parameters
    ----------
    lines:
        Iterable of OCR text lines.
    category_map:
        Optional mapping of categories to vendor keywords.
    vendor_lookup:
        Optional mapping of vendor names to categories loaded from CSV/JSON.
    """

    if category_map is None:
        category_map = CATEGORY_MAP

    lines = [line.strip() for line in lines]
    full_text = "\n".join(lines).lower()

    date_match = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", full_text)

    subtotal = None
    tax = None
    total = None
    payment_method = None
    card_last4 = None

    for line in lines:
        lower = line.lower()

        if subtotal is None and re.search(r"sub[-\s]*total|net amount", lower):
            m = re.search(r"([0-9]+[.,][0-9]{2})", line.replace(",", ""))
            if m:
                subtotal = float(m.group(1))

        if tax is None and "tax" in lower:
            m = re.search(r"([0-9]+[.,][0-9]{2})", line.replace(",", ""))
            if m:
                tax = float(m.group(1))

        if total is None and re.search(r"\b(?:total|amount due)\b", lower):
            m = re.search(r"([0-9]+[.,][0-9]{2})", line.replace(",", ""))
            if m:
                total = float(m.group(1))

        if payment_method is None:
            for method in ["visa", "mastercard", "amex", "discover", "debit", "credit", "cash"]:
                if method in lower:
                    payment_method = method.capitalize() if method != "amex" else "AMEX"
                    if method == "cash":
                        break
                    card_match = re.search(r"(?:\*{2,}|x+)\s*(\d{4})", lower)
                    if not card_match:
                        card_match = re.search(r"ending in\s*(\d{4})", lower)
                    if card_match:
                        card_last4 = card_match.group(1)
                    break

    date = date_match.group(1) if date_match else ""
    vendor = lines[0] if lines else "Unknown"

    category = "uncategorized"
    if vendor_lookup:
        category = vendor_lookup.get(vendor.lower(), category)
    if category == "uncategorized":
        for cat, keywords in category_map.items():
            if any(kw in full_text for kw in keywords):
                category = cat
                break

    return ReceiptFields(
        vendor=vendor,
        date=date,
        subtotal=subtotal,
        tax=tax,
        total=total,
        payment_method=payment_method,
        card_last4=card_last4,
        category=category,
        lines=list(lines),
    )
