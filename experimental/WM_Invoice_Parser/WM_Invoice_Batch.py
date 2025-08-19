import re
import pdfplumber
import pandas as pd
import os

input_dir = r"C:\Path\To\Your\PDFs"  # <-- update to your folder path
output_file = "all_wm_invoices.xlsx"

records = []

# Compile the regex pattern just once
pattern = re.compile(
    r"DETAILS OF SERVICE\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})\s*"
    r"Vehicle:\s*([^\s]+)\s*Carrier:\s*([^\s]*)\s*Ticket#:\s*([^\s]+)\s*"
    r"Profile#:\s*([^\s]+)\s*Manifest#:\s*([^\s]+)\s*Generator:\s*([\s\S]+?)\s*Container:.*?"
    r"DESCRIPTION\s+(.+?)\s+(\d+\.\d{2})\s+([A-Z]+)\s+\$(\d+\.\d{2})\s+\$(\d+\.\d{2})\s*"
    r"TICKET TOTAL", re.MULTILINE
)

for fname in os.listdir(input_dir):
    if fname.lower().endswith('.pdf'):
        pdf_path = os.path.join(input_dir, fname)
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"

        for match in pattern.finditer(text):
            date, vehicle, carrier, ticket, profile, manifest, generator, desc, qty, uom, rate, amt = match.groups()
            records.append({
                "File Name": fname,
                "Date": date.strip(),
                "Vehicle": vehicle.strip(),
                "Carrier": carrier.strip(),
                "Ticket#": ticket.strip(),
                "Profile#": profile.strip(),
                "Manifest#": manifest.strip(),
                "Generator": generator.replace('\n', ' ').strip(),
                "Description": desc.strip(),
                "Quantity": float(qty),
                "UOM": uom.strip(),
                "Rate": float(rate),
                "Amount": float(amt)
            })

# Write to Excel or CSV
df = pd.DataFrame(records)
df.to_excel(output_file, index=False)
print(f"Done! Saved all loads to {output_file}")
