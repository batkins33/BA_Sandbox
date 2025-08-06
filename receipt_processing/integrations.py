"""Placeholder integration hooks for external systems.

These stubs document where credentials/configuration should be supplied
in the future and allow the core pipeline to call out to integrations
without causing runtime errors.
"""

from __future__ import annotations

from typing import Any, Dict


def push_to_firefly(receipt_data: Dict[str, Any]) -> None:
    """Stub for Firefly integration.

    Add API calls here.  Authentication credentials should be provided
    via environment variables or a configuration file.
    """

    print("[integration] push_to_firefly called")


def push_to_google_sheets(receipt_data: Dict[str, Any]) -> None:
    """Stub for Google Sheets integration."""
    print("[integration] push_to_google_sheets called")


def push_to_sharepoint(receipt_data: Dict[str, Any]) -> None:
    """Stub for SharePoint integration."""
    print("[integration] push_to_sharepoint called")

