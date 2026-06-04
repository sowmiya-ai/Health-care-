import re
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Direct Google Sheet URL - loads automatically without prompting
SHEET_URL = "https://docs.google.com/spreadsheets/d/1yy9nuI2vJmZffKOsXLjSP3aHJFoY4QcyzbbiiTr0U0s/export?format=csv&gid=276980714"
FORECAST_DAYS_DEFAULT = 14

APP_TITLE = "HHS UAC Predictive Forecasting Dashboard"
SOURCE_SHEET = "HHS_Unaccompanied_Alien_Children_Program"

HEADERS = {
    "date": "Date",
    "apprehended": "Children apprehended and placed in CBP custody*",
    "cbpCustody": "Children in CBP custody",
    "transferred": "Children transferred out of CBP custody",
    "hhsCare": "Children in HHS Care",
    "discharged": "Children discharged from HHS Care",
}


def normalize_header(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def clean_number(value: object) -> float:
    if pd.isna(value):
        return 0
    text = str(value).replace(",", "").replace("*", "").strip()
    if text == "":
        return 0
    return pd.to_numeric(text, errors="coerce") if pd.notna(pd.to_numeric(text, errors="coerce")) else 0


def extract_sheet_id(url: str) -> str | None:
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url or "")
    return match.group(1) if match else None


def extract_gid(url: str, default_gid: str = "0") -> str:
    match = re.search(r"[#&?]gid=([0-9]+)", url or "")
    return match.group(1) if match else default_gid


def google_sheet_csv_url(url: str, gid: str | None = None) -> str:
    if "output=csv" in url or "format=csv" in url:
        return url
    sheet_id = extract_sheet_id(url)
    if not sheet_id:
        return url
    gid_value = gid or extract_gid(url)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid_value}"


@st.cache_data(ttl=300)
def read_google_sheet(url: str, gid: str | None = None) -> pd.DataFrame:
    csv_url = google_sheet_csv_url(url, gid)
    return pd.read_csv(csv_url)


def read_uploaded_file(file) -> pd.DataFrame:
    name = (file.name or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("No data found.")

    normalized_columns = {normalize_header(col): col for col in df.columns}

    missing = []
    selected_cols = {}
    for key, expected_header in HEADERS.items():
        normalized = normalize_header(expected_header)
        if normalized not in normalized_columns:
            missing.append(expected_header)
        else:
            selected_cols[key] = normalized_columns[normalized]

    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[selected_cols["date"]], errors="coerce")
    out["apprehended"] = df[selected_cols["apprehended"]].apply(clean_number).astype(float)
    out["cbpCustody"] = df[selected_cols["cbpCustody"]].apply(clean_number).astype(float)
    out["transferred"] = df[selected_cols["transferred"]].apply(clean_number).astype(float)
    out["hhsCare"] = df[selected_cols["hhsCare"]].apply(clean_number).astype(float)
    out["discharged"] = df[selected_cols["discharged"]].apply(clean_number).astype(float)

    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    out["netPressure"] = out["transferred"] - out["discharged"]
    out["dateText"] = out["date"].dt.strftime("%b %d, %Y")
    out["dateKey"] = out["date"].dt.strftime("%Y-%m-%d")

    if len(out) < 7:
        raise ValueError(f"Minimum 7 valid daily records required. Current valid records: {len(out)}")

    return out


def average(series) -> float:arr = pd.Series(series).dropna()if arr.empty:return 0return float(arr.mean())

def standard_deviation(series) -> float:arr = pd.Series(series).dropna()if arr.empty:return 0
