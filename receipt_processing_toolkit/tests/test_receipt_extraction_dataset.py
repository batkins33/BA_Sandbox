import pytest
from receipt_processing import extract_fields, assign_item_category

# Fixed set of test receipts with expected extraction results
TEST_RECEIPTS = [
    {
        "name": "grocery_basic",
        "lines": [
            "Kroger",
            "Date: 10/12/2024",
            "Subtotal $52.30",
            "Sales Tax $3.79",
            "Total 56.09",
            "Visa **** 4921",
        ],
        "vendor_lookup": {"kroger": "Groceries"},
        "expected": {
            "vendor": "Kroger",
            "date": "10/12/2024",
            "subtotal": 52.30,
            "tax": 3.79,
            "total": 56.09,
            "payment_method": "Visa",
            "card_last4": "4921",
            "category": "Groceries",
            "line_items": [],
        },
    },
    {
        "name": "restaurant_line_items",
        "lines": [
            "Store A",
            "1 Burger 4.00 T",
            "1 Fries 2.00",
            "Subtotal 6.00",
            "Tax 0.40",
            "Total 6.40",
            "Date: 05/05/2024",
            "Mastercard **** 5678",
        ],
        "vendor_lookup": {"store a": "Meals"},
        "expected": {
            "vendor": "Store A",
            "date": "05/05/2024",
            "subtotal": 6.00,
            "tax": 0.40,
            "total": 6.40,
            "payment_method": "Mastercard",
            "card_last4": "5678",
            "category": "Meals",
            "line_items": [
                {
                    "item_description": "Burger",
                    "price": 4.00,
                    "quantity": 1,
                    "taxable": True,
                    "category": assign_item_category("Burger"),
                },
                {
                    "item_description": "Fries",
                    "price": 2.00,
                    "quantity": 1,
                    "taxable": False,
                    "category": assign_item_category("Fries"),
                },
            ],
        },
    },
]


@pytest.mark.parametrize("case", TEST_RECEIPTS, ids=lambda c: c["name"])
def test_receipt_extraction_accuracy(case):
    """Process a test receipt and compare extracted fields with expected values."""
    fields = extract_fields(case["lines"], vendor_lookup=case.get("vendor_lookup"))
    expected = case["expected"]

    for key, expected_value in expected.items():
        actual_value = getattr(fields, key)
        if isinstance(expected_value, float):
            assert actual_value == pytest.approx(expected_value), f"Mismatch for {key} in {case['name']}"
        else:
            assert actual_value == expected_value, f"Mismatch for {key} in {case['name']}"
