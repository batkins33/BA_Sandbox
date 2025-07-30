"""Utility functions for the receipt processing app."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# Default categories mapped to vendor keywords
CATEGORY_MAP: dict[str, list[str]] = {
    "fuel": ["shell", "chevron", "exxon", "gas"],
    "meals": ["restaurant", "grill", "mcdonald", "subway", "burger"],
    "supplies": ["office depot", "staples", "lowes", "home depot"],
}

@dataclass
class ReceiptFields:
    vendor: str
    date: str
    total: str
    category: str
    lines: list[str]


def extract_fields(lines: Iterable[str], category_map: dict[str, list[str]] | None = None) -> ReceiptFields:
    """Extract key information from the OCR text lines."""
    if category_map is None:
        category_map = CATEGORY_MAP

    lines = [line.strip() for line in lines]
    full_text = "\n".join(lines).lower()

    date_match = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", full_text)
    total_match = re.search(r"(?:total|amount due)[^\d]*(\d+[.,]?\d*)", full_text)

    date = date_match.group(1) if date_match else ""
    total = total_match.group(1) if total_match else ""

    vendor = lines[0] if lines else "Unknown"

    category = "uncategorized"
    for cat, keywords in category_map.items():
        if any(kw in full_text for kw in keywords):
            category = cat
            break

    return ReceiptFields(vendor=vendor, date=date, total=total, category=category, lines=list(lines))
