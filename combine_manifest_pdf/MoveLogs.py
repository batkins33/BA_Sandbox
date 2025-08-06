import os
from pathlib import Path
import shutil

COMBINED_ROOT = Path(r"C:\Users\brian.atkins\OneDrive - Lindamood Demolition\Documents - Truck Tickets\General\24-105-PHMS New Pediatric Campus\MANIFEST\Combined")

for month_dir in COMBINED_ROOT.iterdir():
    if month_dir.is_dir():
        log_dir = month_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        for file in month_dir.glob("*.xlsx"):
            dest = log_dir / file.name
            print(f"[MOVING] {file.name} → {dest}")
            shutil.move(str(file), str(dest))

print("\n[COMPLETE] All logs moved to 'logs' folders.")
