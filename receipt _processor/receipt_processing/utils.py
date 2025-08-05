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
    line_items: list[dict[str, float | str]]
    image_path: Optional[Path] = None


def _last_amount(line: str) -> Optional[float]:
    """Return the last monetary amount on ``line`` ignoring percentages.

    Parameters
    ----------
    line:
        The text line to search for amounts.

    Returns
    -------
    Optional[float]
        The last monetary value found or ``None`` if none present.
    """

    cleaned = line.replace(",", "")
    matches = list(re.finditer(r"([0-9]+[.,][0-9]{2})(?!\s*%)", cleaned))
    if matches:
        return float(matches[-1].group(1))
    return None


def clean_ocr_lines(lines: Iterable[str]) -> list[str]:
    """Normalize raw OCR lines for easier downstream parsing.

    This utility performs a few light preprocessing steps:

    * Strips leading/trailing whitespace.
    * Joins obviously broken lines – a line that ends with an alphanumeric
      character (suggesting the word was cut off) or a very short line followed
      by a longer one will be concatenated with the following line.
    * Removes blank and duplicate lines while preserving order.

    Parameters
    ----------
    lines:
        Raw OCR output lines.

    Returns
    -------
    list[str]
        Cleaned and merged lines.
    """

    # Remove blank lines early
    raw = [l.strip() for l in lines if l and l.strip()]

    merged: list[str] = []
    i = 0
    while i < len(raw):
        line = raw[i]
        if i + 1 < len(raw):
            nxt = raw[i + 1]
            # Heuristics: join if the line is very short or looks unfinished
            if (
                len(line) <= 3
                or line.endswith("-")
                or (line[-1].islower() and nxt[0].islower())
                or (line[-1].isdigit() and nxt[0].isdigit())
            ):
                line = f"{line} {nxt}"
                i += 1
        if not merged or line != merged[-1]:
            merged.append(line)
        i += 1

    # Remove any remaining duplicates while preserving order
    seen: set[str] = set()
    cleaned: list[str] = []
    for line in merged:
        if line not in seen:
            seen.add(line)
            cleaned.append(line)

    return cleaned


def extract_fields(
    lines: Iterable[str],
    category_map: dict[str, list[str]] | None = None,
    vendor_lookup: dict[str, str] | None = None,
) -> ReceiptFields:
    """Extract key information from the OCR text lines."""

    if category_map is None:
        category_map = CATEGORY_MAP

    # Normalise the OCR output first
    lines = clean_ocr_lines(lines)
    full_text = "\n".join(lines).lower()

    date_match = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", full_text)

    subtotal = tax = total = None
    payment_method = card_last4 = None
    line_items: list[dict[str, float | str]] = []

    # Frequently occurring patterns
    subtotal_re = re.compile(r"\bsub[\s-]*total\b|net amount", re.I)
    tax_re = re.compile(r"\b(?:sales\s+)?tax\b|\b(?:gst|hst|vat)\b", re.I)
    total_re = re.compile(r"\b(?:total|grand total|amount due|balance due)\b", re.I)
    summary_re = re.compile(
        r"sub[\s-]*total|total|grand total|amount due|balance due|(?:sales\s+)?tax|gst|hst|vat",
        re.I,
    )
    costco_item_re = re.compile(r"^(?:[A-Z]\s+)?\d+\s+(.+?)\s+([0-9]+[.,][0-9]{2})$")
    generic_item_re = re.compile(r"^(.*?)[\s$]*([0-9]+[.,][0-9]{2})\s*$")

    i = 0
    while i < len(lines):
        line = lines[i]
        lower = line.lower()

        # Subtotal
        if subtotal is None and subtotal_re.search(lower):
            amount = _last_amount(line)
            if amount is None:
                for j in (1, 2):
                    if i + j < len(lines):
                        amount = _last_amount(lines[i + j])
                        if amount is not None:
                            break
            if amount is not None:
                subtotal = amount

        # Tax
        if tax is None and tax_re.search(lower):
            amount = _last_amount(line)
            if amount is None:
                for j in (1, 2):
                    if i + j < len(lines):
                        amount = _last_amount(lines[i + j])
                        if amount is not None:
                            break
            if amount is not None:
                tax = amount

        # Total
        if total is None and total_re.search(lower):
            amount = _last_amount(line)
            if amount is None:
                for j in (1, 2):
                    if i + j < len(lines):
                        amount = _last_amount(lines[i + j])
                        if amount is not None:
                            break
            if amount is not None:
                total = amount

        # Payment method and card digits
        if payment_method is None:
            for method in ["visa", "mastercard", "amex", "discover", "debit", "credit", "cash"]:
                if method in lower:
                    payment_method = method.capitalize() if method != "amex" else "AMEX"
                    if method != "cash":
                        search_lines = [line]
                        if i + 1 < len(lines):
                            search_lines.append(lines[i + 1])
                        if i > 0:
                            search_lines.append(lines[i - 1])
                        card_pat = re.compile(r"(?:\*{2,}|x+)\s*(\d{4})|ending in\s*(\d{4})", re.I)
                        for l in search_lines:
                            card_match = card_pat.search(l)
                            if card_match:
                                card_last4 = card_match.group(1) or card_match.group(2)
                                break
                    break

        # Line-item extraction
        if not summary_re.search(lower):
            item_line = line
            match = costco_item_re.match(item_line) or generic_item_re.match(item_line)
            consumed_next = False
            if (
                not match
                and i + 1 < len(lines)
                and not summary_re.search(lines[i + 1].lower())
                and re.search(r"\d", line)
            ):
                combined = f"{line} {lines[i + 1]}"
                match = costco_item_re.match(combined) or generic_item_re.match(combined)
                if match:
                    item_line = combined
                    consumed_next = True
            if match:
                desc = match.group(1).strip()
                amount = float(match.group(2).replace(",", ""))
                if desc:
                    line_items.append({"item": desc, "amount": amount})
                if consumed_next:
                    i += 1

        i += 1

    # Derive missing monetary fields when possible
    if total is None and subtotal is not None and tax is not None:
        total = round(subtotal + tax, 2)
    elif subtotal is None and total is not None and tax is not None:
        subtotal = round(total - tax, 2)

    date = date_match.group(1) if date_match else ""

    # Attempt to determine vendor by scanning all lines for known patterns
    vendor = "Unknown"
    vendor_patterns: dict[str, re.Pattern[str]] = {
        "costco": re.compile(r"costco|member\s*(?:id|#)", re.I),
        "walmart": re.compile(r"wal[-\s]?mart", re.I),
        "target": re.compile(r"target", re.I),
        "home depot": re.compile(r"home\s+depot", re.I),
    }
    for line in lines:
        for name, pattern in vendor_patterns.items():
            if pattern.search(line):
                vendor = name.title()
                break
        if vendor != "Unknown":
            break
    if vendor == "Unknown" and lines:
        vendor = lines[0]

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
        line_items=line_items,
    )
