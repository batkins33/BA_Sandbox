import importlib
import sys
import types
from pathlib import Path

import pandas as pd


def test_duplicate_receipts_skipped(tmp_path, monkeypatch):
    dummy_config = {
        "input_dir": Path("."),
        "output_dir": Path("."),
        "log_file": Path("receipt_log.xlsx"),
        "line_items_sheet": "Items",
        "auto_crop_enabled": False,
        "auto_orient_enabled": False,
        "use_vendor_csv": False,
        "vendor_csv_path": None,
        "low_confidence_threshold": 0.0,
        "low_confidence_log": Path("."),
        "category_map": {},
        "local_sales_tax_rate": 0.0825,
        "cropping_pad": 0,
    }
    config_module = types.SimpleNamespace(CONFIG=dummy_config)
    monkeypatch.setitem(sys.modules, "config", config_module)
    monkeypatch.setitem(sys.modules, "receipt_processing.config", config_module)

    dummy_integrations = types.SimpleNamespace(
        push_to_firefly=lambda *a, **k: None,
        push_to_google_sheets=lambda *a, **k: None,
        push_to_sharepoint=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "integrations", dummy_integrations)
    monkeypatch.setitem(sys.modules, "receipt_processing.integrations", dummy_integrations)

    dummy_ml = types.SimpleNamespace(
        load_model=lambda *a, **k: None,
        predict_category=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "ml_categorizer", dummy_ml)
    monkeypatch.setitem(sys.modules, "receipt_processing.ml_categorizer", dummy_ml)

    import receipt_processing.utils as utils
    monkeypatch.setitem(sys.modules, "utils", utils)

    main = importlib.import_module("receipt_processing.main")
    _append_to_log = main._append_to_log

    log_path = tmp_path / "log.xlsx"
    monkeypatch.setattr(main, "LOG_FILE", log_path)

    row = {
        "receipt_id": "1",
        "date": "2024-01-01",
        "vendor": "Store",
        "subtotal": 10.0,
        "tax": 1.0,
        "total": 11.0,
        "category": "Other",
        "payment_method": "Visa",
        "card_last4": "1234",
        "confidence_score": 0.9,
        "line_items": "",
        "filename": "1.jpg",
        "processed_time": "2024-01-01T00:00:00",
        "Receipt_Img": "",
        "Total_Img": "",
        "CardLast4_Img": "",
    }

    _append_to_log([row], [])
    assert log_path.exists()

    _append_to_log([row.copy()], [])

    with pd.ExcelFile(log_path) as xls:
        df = pd.read_excel(xls, sheet_name=0)
    assert len(df) == 1
