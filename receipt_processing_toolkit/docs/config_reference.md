# Configuration Reference

The application reads settings from `config.yaml` located in the project root. All fields are optional; missing values fall back to the defaults shown below.

```yaml
input_dir: "C:\\Users\\brian.atkins\\Dropbox\\Personal\\Receipts\\input"
output_dir: "G:\\My Drive\\receipts\\processed"
log_file: "G:\\My Drive\\receipts\\receipt_log.xlsx"
line_items_sheet: "LineItems"
auto_crop_enabled: true
auto_orient_enabled: true
use_vendor_csv: true
vendor_csv_path: "vendor_categories.csv"
low_confidence_threshold: 0.6
low_confidence_log: "low_confidence_receipts.log"
local_sales_tax_rate: 0.08
cropping_pad: 10
category_map:
  fuel: ["shell", "chevron", "exxon", "gas"]
  meals: ["restaurant", "grill", "mcdonald", "subway", "burger"]
  supplies: ["office depot", "staples", "lowes", "home depot"]
```

### Field descriptions

- **input_dir**: Directory watched for new receipts.
- **output_dir**: Destination for processed receipts grouped by category.
- **log_file**: Excel workbook storing the processing log.
- **line_items_sheet**: Worksheet name for line items within the log workbook.
- **auto_crop_enabled**: Automatically crop receipt images before OCR.
- **auto_orient_enabled**: Rotate images upright using EXIF data and heuristics.
- **use_vendor_csv**: Load additional vendor-to-category mappings from a CSV file.
- **vendor_csv_path**: Path to the CSV file used when `use_vendor_csv` is true.
- **low_confidence_threshold**: Receipts scoring below this value are logged for review.
- **low_confidence_log**: File path where low-confidence receipts are recorded.
- **local_sales_tax_rate**: Default tax rate used to infer taxable items.
- **cropping_pad**: Margin in pixels added around detected content during cropping.
- **category_map**: Default mapping of vendor keywords to categories.
