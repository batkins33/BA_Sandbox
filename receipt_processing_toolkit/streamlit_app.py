from __future__ import annotations

"""Simple Streamlit UI for reviewing and correcting receipt data."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from receipt_processing.config import CONFIG

LOG_FILE = Path(os.getenv("RECEIPT_LOG_PATH", "receipt_log.xlsx"))
LOW_CONF_THRESHOLD = CONFIG["low_confidence_threshold"]

st.title("Receipt Review")

if LOG_FILE.exists():
    df = pd.read_excel(LOG_FILE)
else:
    st.warning("receipt_log.xlsx not found. A new file will be created on save.")
    df = pd.DataFrame(
        columns=[
            "receipt_id",
            "date",
            "vendor",
            "subtotal",
            "tax",
            "total",
            "category",
            "payment_method",
            "card_last4",
            "line_items",
            "filename",
            "processed_time",
            "Receipt_Img",
            "Total_Img",
            "CardLast4_Img",
        ]
    )

if "confidence_score" not in df.columns:
    df["confidence_score"] = None
df["needs_review"] = df["confidence_score"].fillna(0) < LOW_CONF_THRESHOLD

# Filters
col1, col2, col3, col4 = st.columns(4)
with col1:
    vendor_filter = st.text_input("Vendor contains")
with col2:
    category_filter = st.text_input("Category contains")
with col3:
    date_filter = st.text_input("Date contains")
with col4:
    low_conf_only = st.checkbox("Low confidence only")

filtered_df = df.copy()
if vendor_filter:
    filtered_df = filtered_df[filtered_df["vendor"].str.contains(vendor_filter, case=False, na=False)]
if category_filter:
    filtered_df = filtered_df[filtered_df["category"].str.contains(category_filter, case=False, na=False)]
if date_filter:
    filtered_df = filtered_df[filtered_df["date"].astype(str).str.contains(date_filter)]
if low_conf_only:
    filtered_df = filtered_df[filtered_df["needs_review"]]

edited_df = st.data_editor(
    filtered_df,
    num_rows="dynamic",
    key="editor",
    column_config={
        "confidence_score": st.column_config.ProgressColumn(
            "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
        )
    },
)

with st.expander("Bulk assign category to vendor"):
    bulk_vendor = st.text_input("Vendor name")
    bulk_category = st.text_input("New category")
    if st.button("Apply to all matching vendor"):
        mask = df["vendor"].str.lower() == bulk_vendor.lower()
        df.loc[mask, "category"] = bulk_category
        st.success("Category updated. Don't forget to save.")

if st.button("Save changes"):
    df.update(edited_df)
    df["needs_review"] = df["confidence_score"].fillna(0) < LOW_CONF_THRESHOLD
    df.to_excel(LOG_FILE, index=False)
    st.success("Changes saved")

uploaded = st.file_uploader("Upload new receipt", type=["png", "jpg", "jpeg", "pdf"])
if uploaded is not None:
    input_dir = Path(os.getenv("RECEIPT_INPUT_DIR", "input"))
    input_dir.mkdir(parents=True, exist_ok=True)
    dest = input_dir / uploaded.name
    with open(dest, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Saved {dest}")
