from receipt_processing.utils import extract_fields, ReceiptFields, CATEGORY_MAP


def test_extract_fields_basic():
    lines = [
        "Shell Gas Station",
        "Date: 01/02/2024",
        "Total: $45.67",
    ]
    fields = extract_fields(lines)
    assert isinstance(fields, ReceiptFields)
    assert fields.vendor == "Shell Gas Station"
    assert fields.date == "01/02/2024"
    assert fields.total == "45.67"
    assert fields.category == "fuel"


def test_extract_fields_uncategorized():
    lines = [
        "Unknown Vendor",
        "Amount Due 12.00",
    ]
    fields = extract_fields(lines)
    assert fields.category == "uncategorized"
    assert fields.total == "12.00"

