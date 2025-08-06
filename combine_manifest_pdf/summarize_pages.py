import pandas as pd
from pathlib import Path
from datetime import datetime
import re

COMBINED_ROOT = Path(r"C:\Users\brian.atkins\OneDrive - Lindamood Demolition\Documents - Truck Tickets\General\24-105-PHMS New Pediatric Campus\MANIFEST\Combined")
OUTPUT_FILE = COMBINED_ROOT / "combined_manifest_summary.xlsx"

all_logs = []

# Traverse all logs folders
for log_file in COMBINED_ROOT.rglob("logs/*.xlsx"):
    match = re.search(r"24-105_(\d{2}\.\d{2}\.\d{4})_manifest_combined", log_file.name)
    if not match:
        continue

    folder_date = datetime.strptime(match.group(1), "%m.%d.%Y")
    month = folder_date.strftime("%Y-%m")
    day = folder_date.strftime("%m.%d.%Y")
    week = folder_date.strftime("Week %U")  # Week starts on Sunday

    try:
        df = pd.read_excel(log_file)
        df["Date"] = day
        df["Month"] = month
        df["Week"] = week
        df["Day"] = day
        all_logs.append(df[["Original File", "Page Count", "Month", "Week", "Day"]])
    except Exception as e:
        print(f"[ERROR] Failed to read {log_file}: {e}")

if not all_logs:
    print("No Excel logs found.")
else:
    combined_df = pd.concat(all_logs)

    # Pivot table: nested group by Month > Week > Day
    pivot = combined_df.groupby(["Month", "Week", "Day"])["Page Count"].sum().reset_index()

    # Save result
    pivot.to_excel(OUTPUT_FILE, index=False)
    print(f"[SAVED] Pivot table written to {OUTPUT_FILE}")
