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


