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


@pytest.mark.parametrize("line", ["Grand Total $10.00", "Balance Due $10.00"])
def test_extract_fields_total_variants(line):
    lines = [
        "Store",
        line,
    ]
    fields = extract_fields(lines)
    assert fields.total == 10.00


def test_compute_total_from_subtotal_and_tax():
    lines = [
        "Store",
        "Subtotal $10.00",
        "Tax $0.80",
    ]
    fields = extract_fields(lines)
    assert fields.total == 10.80
    assert fields.subtotal == 10.00
    assert fields.tax == 0.80


def test_compute_subtotal_from_total_and_tax():
    lines = [
        "Store",
        "Tax $0.80",
        "Total $10.80",
    ]
    fields = extract_fields(lines)
    assert fields.subtotal == 10.00
    assert fields.total == 10.80
    assert fields.tax == 0.80


def test_line_item_parsing():
    lines = [
        "Store",
        "Burger 4.99",
        "Fries 2.99",
        "Total 7.98",
    ]
    fields = extract_fields(lines)
    assert fields.line_items == [
        {"item_description": "Burger", "price": 4.99, "quantity": None, "tax": False},
        {"item_description": "Fries", "price": 2.99, "quantity": None, "tax": False},
    ]


def test_line_item_with_quantity_and_tax():
    lines = [
        "Store",
        "2 Burger 4.99 T",
        "Total 9.98",
    ]
    fields = extract_fields(lines)
    assert fields.line_items == [
        {"item_description": "Burger", "price": 4.99, "quantity": 2, "tax": True},
    ]


