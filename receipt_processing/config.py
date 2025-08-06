from __future__ import annotations

from pathlib import Path
import yaml

# Default configuration values
DEFAULT_CONFIG = {
    "input_dir": r"C:\\Users\\brian.atkins\\Dropbox\\Personal\\Receipts\\input",
    "output_dir": r"G:\\My Drive\\receipts\\processed",
    "log_file": r"G:\\My Drive\\receipts\\receipt_log.xlsx",
    "line_items_sheet": "LineItems",
    "auto_crop_enabled": True,
    "auto_orient_enabled": True,
    "use_vendor_csv": True,
    "vendor_csv_path": "vendor_categories.csv",
    "low_confidence_threshold": 0.6,
    "low_confidence_log": "low_confidence_receipts.log",
    "local_sales_tax_rate": 0.08,
    "cropping_pad": 10,
    "category_map": {
        "fuel": ["shell", "chevron", "exxon", "gas"],
        "meals": ["restaurant", "grill", "mcdonald", "subway", "burger"],
        "supplies": ["office depot", "staples", "lowes", "home depot"],
    },
}

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load configuration from ``path`` merged with defaults."""
    data: dict | None = None
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    cfg = {**DEFAULT_CONFIG, **(data or {})}
    # Convert known path fields to ``Path`` objects
    for key in [
        "input_dir",
        "output_dir",
        "log_file",
        "vendor_csv_path",
        "low_confidence_log",
    ]:
        if cfg.get(key) is not None:
            cfg[key] = Path(cfg[key])
    return cfg


CONFIG = load_config()
