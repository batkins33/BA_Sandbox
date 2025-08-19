"""CLI entry point for combine_manifest_pdf module."""

import sys
import argparse
from pathlib import Path

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m combine_manifest_pdf",
        description="Manifest PDF processing utilities"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Manifest extraction command
    extract_parser = subparsers.add_parser(
        "extract_manifest_fields",
        help="Extract manifest fields using anchor-based OCR"
    )
    extract_parser.add_argument("src_dir", help="Source directory containing PDF files")
    extract_parser.add_argument("--out", default="manifest_fields.xlsx", help="Output Excel file")
    
    # Combine PDFs command
    combine_parser = subparsers.add_parser(
        "combine",
        help="Combine multiple manifest PDFs into one"
    )
    combine_parser.add_argument("output", help="Output PDF file")
    combine_parser.add_argument("inputs", nargs="+", help="Input PDF files")
    
    args = parser.parse_args()
    
    if args.command == "extract_manifest_fields":
        from .extract_manifest_fields import main as extract_main
        extract_main(args.src_dir, args.out)
    elif args.command == "combine":
        from .main import main as combine_main
        combine_main([args.output] + args.inputs)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()