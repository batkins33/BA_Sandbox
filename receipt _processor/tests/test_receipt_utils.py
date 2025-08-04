import pytest
from receipt_processing.utils import (
    extract_fields,
    ReceiptFields,
    CATEGORY_MAP,
    load_vendor_categories,
)


def test_extract_fields_basic():
    lines = [
        "Kroger",
        "Date: 10/12/2024",
        "Subtotal $52.30",
        "Sales Tax $3.79",
        "Total 56.09",
        "Visa **** 4921",
    ]
    vendor_map = {"kroger": "Groceries"}
    fields = extract_fields(lines, vendor_lookup=vendor_map)
    assert isinstance(fields, ReceiptFields)
    assert fields.vendor == "Kroger"
    assert fields.date == "10/12/2024"
    assert fields.subtotal == 52.30
    assert fields.tax == 3.79
    assert fields.total == 56.09
    assert fields.payment_method == "Visa"
    assert fields.card_last4 == "4921"
    assert fields.category == "Groceries"


def test_extract_fields_uncategorized():
    lines = [
        "Unknown Vendor",
        "Amount Due 12.00",
    ]
    fields = extract_fields(lines)
    assert fields.category == "uncategorized"
    assert fields.total == 12.00
    assert fields.subtotal is None
    assert fields.tax is None
    assert fields.payment_method is None


def test_tax_percentage_ignored():
    lines = [
        "Shop",
        "Sales Tax 8.25% 1.23",
        "Total 10.00",
    ]
    fields = extract_fields(lines)
    assert fields.tax == 1.23


@pytest.mark.parametrize("label", ["GST", "HST", "VAT"])
def test_tax_label_variants(label):
    lines = [
        "Store",
        f"{label} 2.00",
    ]
    fields = extract_fields(lines)
    assert fields.tax == 2.00


@pytest.mark.parametrize("line", ["Sub-Total $10.00", "Net Amount $10.00"])
def test_extract_fields_subtotal_variants(line):
    lines = [
        "Store",
        "Date: 01/01/2024",
        line,
        "Tax $0.80",
        "Total $10.80",
    ]
    fields = extract_fields(lines)
    assert fields.subtotal == 10.00


