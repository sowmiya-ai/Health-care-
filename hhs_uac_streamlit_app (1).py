
import re
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
file_name = r"C:\Users\hp\Downloads\HHS_Unaccompanied_Alien_Children_Program - HHS_Unaccompanied_Alien_Children_Program.csv"
raw_df = pd.read_csv(file_name)

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
def read_google_sheet(url: str, gid: str | None) -> pd.DataFrame:
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


def average(series) -> float:
    arr = pd.Series(series).dropna()
    if arr.empty:
        return 0
    return float(arr.mean())


def standard_deviation(series) -> float:
    arr = pd.Series(series).dropna()
    if arr.empty:
        return 0
    return float(arr.std(ddof=0))


def average_daily_trend(series, lookback: int = 14) -> float:
    arr = list(pd.Series(series).dropna().tail(lookback))
    if len(arr) < 2:
        return 0
    return (arr[-1] - arr[0]) / (len(arr) - 1)


def risk_level(net_pressure: float) -> str:
    if net_pressure >= 15:
        return "High"
    if net_pressure >= 7:
        return "Medium"
    return "Low"


def signed_number(value: float) -> str:
    value = int(round(value or 0))
    return f"+{value:,}" if value > 0 else f"{value:,}"


def build_forecast(data: pd.DataFrame, days: int) -> pd.DataFrame:
    last14 = data.tail(14)
    avg_transfer = average(last14["transferred"])
    avg_discharge = average(last14["discharged"])
    avg_net = avg_transfer - avg_discharge
    trend = average_daily_trend(data["hhsCare"], 14)
    volatility = standard_deviation(last14["netPressure"])

    last_date = data.iloc[-1]["date"]
    last_care = data.iloc[-1]["hhsCare"]

    rows = []
    for i in range(1, int(days) + 1):
        next_date = last_date + timedelta(days=i)
        transfer_forecast = max(0, round(avg_transfer))
        discharge_forecast = max(0, round(avg_discharge))
        net_pressure = transfer_forecast - discharge_forecast
        hhs_care_forecast = max(0, round(last_care + trend * i + avg_net * 0.25 * i))
        lower_bound = max(0, round(hhs_care_forecast - volatility * 1.5))
        upper_bound = max(0, round(hhs_care_forecast + volatility * 1.5))

        rows.append({
            "date": next_date,
            "dateText": next_date.strftime("%b %d, %Y"),
            "dateKey": next_date.strftime("%Y-%m-%d"),
            "hhsCareForecast": hhs_care_forecast,
            "lowerBound": lower_bound,
            "upperBound": upper_bound,
            "transferForecast": transfer_forecast,
            "dischargeForecast": discharge_forecast,
            "netPressure": net_pressure,
            "risk": risk_level(net_pressure),
        })

    return pd.DataFrame(rows)


def build_dashboard(data: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    latest = data.iloc[-1]
    previous = data.iloc[-2] if len(data) > 1 else latest
    last7 = data.tail(7)
    last14 = data.tail(14)

    avg_transfers7 = average(last7["transferred"])
    avg_discharges7 = average(last7["discharged"])
    avg_transfers14 = average(last14["transferred"])
    avg_discharges14 = average(last14["discharged"])

    forecast_end = forecast.iloc[-1]

    return {
        "latestDate": latest["dateText"],
        "kpis": {
            "hhsCare": latest["hhsCare"],
            "hhsCareChange": latest["hhsCare"] - previous["hhsCare"],
            "apprehended": latest["apprehended"],
            "cbpCustody": latest["cbpCustody"],
            "transferred": latest["transferred"],
            "discharged": latest["discharged"],
            "netPressure": latest["netPressure"],
            "avgTransfers7": round(avg_transfers7),
            "avgDischarges7": round(avg_discharges7),
            "avgNet7": round(avg_transfers7 - avg_discharges7),
            "avgTransfers14": round(avg_transfers14),
            "avgDischarges14": round(avg_discharges14),
            "avgNet14": round(avg_transfers14 - avg_discharges14),
            "forecastEndCare": forecast_end["hhsCareForecast"],
        },
    }


def build_recommendation(status: str) -> str:
    if status == "High Capacity Stress":
        return (
            "Scale shelter capacity, caseworker planning, and medical support immediately. "
            "Review discharge bottlenecks daily and prepare contingency placement capacity."
        )
    if status == "Moderate Watch":
        return (
            "Prepare standby placement capacity. Monitor transfer volume, discharge performance, "
            "and caseworker workload for the next 7 to 14 days."
        )
    return "Maintain current operating capacity. Continue daily monitoring for sudden transfer spikes or discharge slowdown."


def build_ai_analysis(data: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    latest = data.iloc[-1]
    last7 = data.tail(7)
    last14 = data.tail(14)

    avg_transfers7 = average(last7["transferred"])
    avg_discharges7 = average(last7["discharged"])
    avg_net7 = avg_transfers7 - avg_discharges7

    avg_transfers14 = average(last14["transferred"])
    avg_discharges14 = average(last14["discharged"])
    avg_net14 = avg_transfers14 - avg_discharges14

    forecast_end = forecast.iloc[-1]
    high_risk_days = int((forecast["risk"] == "High").sum())
    medium_risk_days = int((forecast["risk"] == "Medium").sum())

    status = "Stable"
    summary = "Current care load appears stable. Transfers and discharges are broadly balanced based on recent daily movement."
    risk_reason = "Low short-term capacity pressure based on recent transfer and discharge trend."

    if high_risk_days > 0 or avg_net7 >= 15:
        status = "High Capacity Stress"
        summary = "Forecast indicates high capacity stress. Incoming transfers may exceed discharge capacity in the short term."
        risk_reason = "High net pressure means transfers are running materially above discharges."
    elif medium_risk_days > 0 or avg_net7 > 0:
        status = "Moderate Watch"
        summary = "Forecast indicates moderate watch condition. HHS care load may rise if discharges do not offset incoming transfers."
        risk_reason = "Positive net pressure shows more children entering HHS care than leaving through discharge."

    findings = [
        f"Latest HHS care load is {int(round(latest['hhsCare'])):,}.",
        f"Latest net pressure is {signed_number(latest['netPressure'])} children.",
        f"7-day average net pressure is {signed_number(avg_net7)} children.",
        f"Forecast end care load is {int(round(forecast_end['hhsCareForecast'])):,}.",
    ]

    if high_risk_days > 0:
        findings.append(f"{high_risk_days} forecast day(s) are classified as High risk.")
    if medium_risk_days > 0:
        findings.append(f"{medium_risk_days} forecast day(s) are classified as Medium risk.")
    if high_risk_days == 0 and medium_risk_days == 0:
        findings.append("No Medium or High forecast risk days detected.")

    return {
        "status": status,
        "summary": summary,
        "riskReason": risk_reason,
        "latestCareLoad": latest["hhsCare"],
        "latestTransfers": latest["transferred"],
        "latestDischarges": latest["discharged"],
        "latestNetPressure": latest["netPressure"],
        "sevenDayAverageTransfers": round(avg_transfers7),
        "sevenDayAverageDischarges": round(avg_discharges7),
        "sevenDayNetPressure": round(avg_net7),
        "fourteenDayAverageTransfers": round(avg_transfers14),
        "fourteenDayAverageDischarges": round(avg_discharges14),
        "fourteenDayNetPressure": round(avg_net14),
        "forecastEndCareLoad": forecast_end["hhsCareForecast"],
        "highRiskDays": high_risk_days,
        "mediumRiskDays": medium_risk_days,
        "recommendation": build_recommendation(status),
        "keyFindings": findings,
    }


def build_report(data: pd.DataFrame, forecast: pd.DataFrame, ai: dict) -> dict:
    latest = data.iloc[-1]
    return {
        "title": "Predictive Forecasting of Care Load & Placement Demand",
        "sourceSheet": SOURCE_SHEET,
        "latestDate": latest["dateText"],
        "recordsUsed": len(data),
        "background": (
            "The dashboard estimates how many children may be under HHS care in the coming days or weeks "
            "and identifies whether discharge capacity is sufficient to offset incoming transfers."
        ),
        "problemStatement": (
            "The UAC Program requires short-term forecasts of children in HHS care, predictive estimates "
            "of discharge demand, and early-warning indicators of upcoming capacity stress."
        ),
        "objectives": [
            "Forecast the number of children in HHS care.",
            "Estimate future imbalance between intake and exits.",
            "Predict short-term discharge demand.",
            "Provide early warnings for healthcare planners.",
            "Quantify forecast uncertainty.",
            "Compare current trend, transfer, and discharge movement.",
        ],
        "currentMetrics": {
            "hhsCare": latest["hhsCare"],
            "apprehended": latest["apprehended"],
            "cbpCustody": latest["cbpCustody"],
            "transferred": latest["transferred"],
            "discharged": latest["discharged"],
            "netPressure": latest["netPressure"],
        },
        "aiStatus": ai["status"],
        "executiveSummary": ai["summary"],
        "forecastEndCareLoad": ai["forecastEndCareLoad"],
        "recommendation": ai["recommendation"],
        "methodology": [
            "Converted Date column into a daily time-series index.",
            "Sorted records chronologically.",
            "Cleaned numeric fields containing commas or blank values.",
            "Calculated net pressure as transfers minus discharges.",
            "Used recent 14-day transfer and discharge averages for short-term flow estimation.",
            "Used recent HHS care trend for care load projection.",
            "Used recent net-pressure volatility to create lower and upper forecast ranges.",
            "Classified risk as Low, Medium, or High based on short-term net pressure.",
        ],
    }


def status_class(status: str) -> str:
    if status == "High Capacity Stress":
        return "status-high"
    if status == "Moderate Watch":
        return "status-medium"
    return "status-low"


def risk_color(risk: str) -> str:
    if risk == "High":
        return "#b91c1c"
    if risk == "Medium":
        return "#c2410c"
    return "#15803d"


def apply_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {padding-top: 1.1rem; max-width: 1400px;}
        div[data-testid="stMetricValue"] {font-size: 1.65rem;}
        .hero {
            background: linear-gradient(135deg,#1e3a5f 0%,#1e40af 60%,#3730a3 100%);
            color: white; padding: 18px 24px; border-radius: 18px; margin-bottom: 12px;
            box-shadow: 0 6px 18px rgba(30,64,175,.25);
        }
        .hero h1 {margin: 0; font-size: 25px;}
        .hero p {margin: 6px 0 0; opacity: .88; font-size: 13px;}
        .kpi-card {
            border-radius: 16px; padding: 15px 16px; color:#fff; min-height: 118px;
            box-shadow: 0 4px 14px rgba(0,0,0,.12); margin-bottom: 10px;
        }
        .kpi-label {font-size: 11px; font-weight: 700; opacity:.9; text-transform: uppercase;}
        .kpi-value {font-size: 30px; font-weight: 900; line-height:1; margin-top: 8px;}
        .kpi-note {font-size: 12px; opacity: .9; margin-top: 6px;}
        .blue{background:linear-gradient(135deg,#1e40af,#3b82f6);}
        .teal{background:linear-gradient(135deg,#0f766e,#14b8a6);}
        .green{background:linear-gradient(135deg,#15803d,#22c55e);}
        .orange{background:linear-gradient(135deg,#c2410c,#f97316);}
        .purple{background:linear-gradient(135deg,#6d28d9,#a78bfa);}
        .indigo{background:linear-gradient(135deg,#3730a3,#6366f1);}
        .rose{background:linear-gradient(135deg,#be123c,#fb7185);}
        .cyan{background:linear-gradient(135deg,#0e7490,#22d3ee);}
        .panel {
            background:#fff; border:1px solid #e2e8f0; border-radius:16px; padding:16px;
            box-shadow: 0 2px 10px rgba(0,0,0,.06); margin-bottom: 12px;
        }
        .panel-title {font-weight:900; color:#1e3a5f; margin-bottom: 8px; font-size: 15px;}
        .status-pill {display:inline-block; padding:6px 13px; border-radius:999px; font-size:13px; font-weight:900;}
        .status-low {background:#dcfce7; color:#15803d;}
        .status-medium {background:#ffedd5; color:#c2410c;}
        .status-high {background:#fee2e2; color:#b91c1c;}
        .small-note {color:#64748b; font-size: 12px;}
        .report-header {
            background: linear-gradient(135deg,#1e3a5f,#1e40af); color: white;
            border-radius: 16px; padding: 18px 22px; margin-bottom: 12px;
        }
        .report-header h2 {margin:0; font-size: 22px;}
        .meta-pill {display:inline-block; margin:8px 6px 0 0; background:rgba(255,255,255,.15); padding:5px 10px; border-radius:999px; font-size:12px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, css_class: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card {css_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel(title: str, body_html: str) -> None:
    st.markdown(
        f"""<div class="panel"><div class="panel-title">{title}</div>{body_html}</div>""",
        unsafe_allow_html=True,
    )


def care_forecast_chart(data: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    actual = data.tail(30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual["date"], y=actual["hhsCare"], mode="lines+markers", name="Actual HHS Care"))
    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["hhsCareForecast"], mode="lines+markers", name="Forecast"))
    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["upperBound"], mode="lines", name="Upper Bound", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["lowerBound"], mode="lines", name="Lower Bound", line=dict(dash="dot")))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h"), yaxis_title="Children in HHS Care")
    return fig


def flow_chart(data: pd.DataFrame) -> go.Figure:
    actual = data.tail(30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual["date"], y=actual["transferred"], mode="lines+markers", name="Transfers"))
    fig.add_trace(go.Scatter(x=actual["date"], y=actual["discharged"], mode="lines+markers", name="Discharges"))
    fig.add_trace(go.Scatter(x=actual["date"], y=actual["netPressure"], mode="lines+markers", name="Net Pressure"))
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h"), yaxis_title="Children")
    return fig


def display_dashboard(data: pd.DataFrame, forecast: pd.DataFrame, dashboard: dict, ai: dict) -> None:
    k = dashboard["kpis"]

    row1 = st.columns(4)
    with row1[0]:
        kpi_card("Children in HHS Care", f"{int(k['hhsCare']):,}", "blue", f"Change: {signed_number(k['hhsCareChange'])}")
    with row1[1]:
        kpi_card("Transferred Out of CBP", f"{int(k['transferred']):,}", "teal")
    with row1[2]:
        kpi_card("Discharged from HHS", f"{int(k['discharged']):,}", "green")
    with row1[3]:
        kpi_card("Net Pressure", signed_number(k["netPressure"]), "orange")

    row2 = st.columns(4)
    with row2[0]:
        kpi_card("Apprehended → CBP", f"{int(k['apprehended']):,}", "purple")
    with row2[1]:
        kpi_card("In CBP Custody", f"{int(k['cbpCustody']):,}", "indigo")
    with row2[2]:
        kpi_card("7-Day Avg Net Pressure", signed_number(k["avgNet7"]), "rose")
    with row2[3]:
        kpi_card("Forecast End HHS Care", f"{int(k['forecastEndCare']):,}", "cyan")

    left, right = st.columns([2, 1])
    with left:
        panel("📈 HHS Care Load Forecast", "")
        st.plotly_chart(care_forecast_chart(data, forecast), use_container_width=True)
    with right:
        status = ai["status"]
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title">🤖 AI Early Warning</div>
                <span class="status-pill {status_class(status)}">{status}</span>
                <p><b>Summary</b><br>{ai['summary']}</p>
                <p><b>Risk Reason</b><br>{ai['riskReason']}</p>
                <p><b>Recommendation</b><br>{ai['recommendation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns([2, 1])
    with left:
        panel("🔀 Transfers vs Discharges", "")
        st.plotly_chart(flow_chart(data), use_container_width=True)
    with right:
        summary_rows = pd.DataFrame(
            [
                ["7-Day Avg Transfers", k["avgTransfers7"]],
                ["7-Day Avg Discharges", k["avgDischarges7"]],
                ["7-Day Avg Net Pressure", signed_number(k["avgNet7"])],
                ["14-Day Avg Transfers", k["avgTransfers14"]],
                ["14-Day Avg Discharges", k["avgDischarges14"]],
                ["14-Day Avg Net Pressure", signed_number(k["avgNet14"])],
            ],
            columns=["Metric", "Value"],
        )
        panel("📋 Rolling Flow Summary", "")
        st.dataframe(summary_rows, hide_index=True, use_container_width=True)

    st.markdown("### 📅 Forecast Table")
    display_forecast = forecast.copy()
    display_forecast = display_forecast.rename(
        columns={
            "dateText": "Date",
            "hhsCareForecast": "HHS Forecast",
            "lowerBound": "Lower",
            "upperBound": "Upper",
            "transferForecast": "Transfer",
            "dischargeForecast": "Discharge",
            "netPressure": "Net Pressure",
            "risk": "Risk",
        }
    )[["Date", "HHS Forecast", "Lower", "Upper", "Transfer", "Discharge", "Net Pressure", "Risk"]]
    st.dataframe(display_forecast, hide_index=True, use_container_width=True)


def display_ai(ai: dict) -> None:
    status = ai["status"]
    st.markdown(
        f"""
        <div class="panel">
            <span class="status-pill {status_class(status)}">{status}</span>
            <p><b>Summary:</b> {ai['summary']}</p>
            <p><b>Risk:</b> {ai['riskReason']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    metrics = [
        ("Latest HHS Care", ai["latestCareLoad"]),
        ("Latest Transfers", ai["latestTransfers"]),
        ("Latest Discharges", ai["latestDischarges"]),
        ("Latest Net Pressure", signed_number(ai["latestNetPressure"])),
        ("7-Day Avg Transfers", ai["sevenDayAverageTransfers"]),
        ("7-Day Avg Discharges", ai["sevenDayAverageDischarges"]),
        ("7-Day Net Pressure", signed_number(ai["sevenDayNetPressure"])),
        ("Forecast End Care", ai["forecastEndCareLoad"]),
    ]

    colors = ["blue", "teal", "green", "orange", "purple", "indigo", "rose", "cyan"]
    for idx, (label, value) in enumerate(metrics):
        with cols[idx % 4]:
            text_value = value if isinstance(value, str) else f"{int(value):,}"
            kpi_card(label, text_value, colors[idx])

    left, right = st.columns(2)
    with left:
        st.markdown("### 🔍 Key Findings")
        for item in ai["keyFindings"]:
            st.markdown(f"- {item}")
    with right:
        st.markdown("### ⚠️ Capacity Warning Summary")
        cap = pd.DataFrame(
            [
                ["High Risk Days", ai["highRiskDays"]],
                ["Medium Risk Days", ai["mediumRiskDays"]],
                ["14-Day Avg Transfers", ai["fourteenDayAverageTransfers"]],
                ["14-Day Avg Discharges", ai["fourteenDayAverageDischarges"]],
                ["14-Day Net Pressure", signed_number(ai["fourteenDayNetPressure"])],
            ],
            columns=["Metric", "Value"],
        )
        st.dataframe(cap, hide_index=True, use_container_width=True)

    st.info("💡 AI Recommendation: " + ai["recommendation"])


def display_report(report: dict) -> None:
    st.markdown(
        f"""
        <div class="report-header">
            <h2>{report['title']}</h2>
            <span class="meta-pill">📄 Source: <b>{report['sourceSheet']}</b></span>
            <span class="meta-pill">📅 Date: <b>{report['latestDate']}</b></span>
            <span class="meta-pill">📊 Records: <b>{report['recordsUsed']:,}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        panel("📋 Background & Context", f"<p>{report['background']}</p>")
    with c2:
        panel("❓ Problem Statement", f"<p>{report['problemStatement']}</p>")

    st.markdown("### 🎯 Project Objectives")
    for item in report["objectives"]:
        st.markdown(f"- {item}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📌 Executive Summary")
        st.markdown(f"<span class='status-pill {status_class(report['aiStatus'])}'>{report['aiStatus']}</span>", unsafe_allow_html=True)
        st.write(report["executiveSummary"])
    with c2:
        st.markdown("### 💡 Recommendation")
        st.write(report["recommendation"])

    st.markdown("### 📈 Current Metrics")
    metric_cols = st.columns(7)
    metrics = [
        ("Apprehended → CBP", report["currentMetrics"]["apprehended"]),
        ("In CBP Custody", report["currentMetrics"]["cbpCustody"]),
        ("Transferred Out CBP", report["currentMetrics"]["transferred"]),
        ("In HHS Care", report["currentMetrics"]["hhsCare"]),
        ("Discharged HHS", report["currentMetrics"]["discharged"]),
        ("Net Pressure", signed_number(report["currentMetrics"]["netPressure"])),
        ("Forecast End Care", report["forecastEndCareLoad"]),
    ]
    for col, (label, value) in zip(metric_cols, metrics):
        with col:
            display_value = value if isinstance(value, str) else f"{int(value):,}"
            st.metric(label, display_value)

    st.markdown("### 🔬 Analytical Methodology")
    for idx, item in enumerate(report["methodology"], start=1):
        st.markdown(f"{idx}. {item}")


def main() -> None:
    st.set_page_config(page_title="HHS UAC Forecast Dashboard", page_icon="🏛️", layout="wide")
    apply_css()

    st.markdown(
        """
        <div class="hero">
            <h1>🏛️ HHS UAC Predictive Forecasting Web App</h1>
            <p>Care Load Forecasting · Placement Demand · Discharge Prediction · Early Warning Analysis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Data Source")
        source_mode = st.radio("Choose source", ["Upload Excel/CSV", "Google Sheet URL"], index=0)
        forecast_days = st.selectbox("Forecast Days", [7, 14, 21, 30], index=1)

        raw_df = None

        if source_mode == "Upload Excel/CSV":
            uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])
            if uploaded is not None:
                raw_df = read_uploaded_file(uploaded)
            else:
                st.info("Upload your source file to load dashboard.")
        else:
            sheet_url = st.text_input("Google Sheet URL")
            gid = st.text_input("Sheet GID", value="0")
            st.caption("The Google Sheet must be shared as Anyone with the link → Viewer, or published as CSV.")
            if sheet_url:
                try:
                    raw_df = read_google_sheet(sheet_url, gid)
                except Exception as exc:
                    st.error(f"Unable to read Google Sheet: {exc}")

        st.markdown("---")
        st.caption("Required sheet columns must match the original Google Apps Script headers.")

    if raw_df is None:
        st.warning("Please upload an Excel/CSV file or enter a public Google Sheet URL.")
        st.stop()

    try:
        data = prepare_data(raw_df)
        forecast = build_forecast(data, int(forecast_days))
        dashboard = build_dashboard(data, forecast)
        ai = build_ai_analysis(data, forecast)
        report = build_report(data, forecast, ai)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.caption(f"Latest: {data.iloc[-1]['dateText']} | Records: {len(data):,} | Source sheet logic: {SOURCE_SHEET}")

    tab_dashboard, tab_ai, tab_report, tab_data = st.tabs(["📊 Dashboard", "🤖 AI Analysis", "📄 Report", "🧾 Data Preview"])

    with tab_dashboard:
        display_dashboard(data, forecast, dashboard, ai)

    with tab_ai:
        display_ai(ai)

    with tab_report:
        display_report(report)

    with tab_data:
        st.markdown("### Actual Data Used")
        st.dataframe(data.tail(90), hide_index=True, use_container_width=True)
        st.markdown("### Forecast Data")
        st.dataframe(forecast, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
