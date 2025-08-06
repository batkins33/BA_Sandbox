import pytest
from receipt_processing import (
    extract_fields,
    ReceiptFields,
    assign_item_category,
    compute_confidence_score,
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
        {
            "item_description": "Burger",
            "price": 4.99,
            "quantity": None,
            "taxable": False,
            "category": assign_item_category("Burger"),
        },
        {
            "item_description": "Fries",
            "price": 2.99,
            "quantity": None,
            "taxable": False,
            "category": assign_item_category("Fries"),
        },
    ]


def test_line_item_with_quantity_and_tax():
    lines = [
        "Store",
        "2 Burger 4.99 T",
        "Total 9.98",
    ]
    fields = extract_fields(lines)
    assert fields.line_items == [
        {
            "item_description": "Burger",
            "price": 4.99,
            "quantity": 2,
            "taxable": True,
            "category": assign_item_category("Burger"),
        },
    ]


def test_infer_taxable_items():
    lines = [
        "Store",
        "Apple 1.00",
        "Banana 1.00",
        "Tax 0.08",
        "Total 2.08",
    ]
    fields = extract_fields(lines)
    taxable_total = sum(
        item["price"] * (item.get("quantity") or 1)
        for item in fields.line_items
        if item["taxable"]
    )
    assert taxable_total == pytest.approx(1.00)
    assert sum(1 for item in fields.line_items if item["taxable"]) == 1


def test_assign_item_category_match():
    keyword_map = {
        "food": ["burger", "fries"],
        "beverage": ["soda", "cola"],
    }
    assert assign_item_category("Cheeseburger", keyword_map) == "food"
    assert assign_item_category("Diet Cola", keyword_map) == "beverage"


def test_assign_item_category_other():
    keyword_map = {"food": ["burger"]}
    assert assign_item_category("Laptop Sleeve", keyword_map) == "Other"


def test_confidence_score_high():
    lines = [
        "Store",
        "Date: 01/01/2024",
        "Apple 1.00",
        "Banana 1.00",
        "Subtotal 2.00",
        "Tax 0.16",
        "Total 2.16",
        "Visa **** 1234",
        "2 items",
    ]
    fields = extract_fields(lines)
    score = compute_confidence_score(fields)
    assert score == 1.0


def test_confidence_score_low():
    lines = ["Store", "Total 10.00"]
    fields = extract_fields(lines)
    score = compute_confidence_score(fields)
    assert score < 0.5

