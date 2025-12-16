import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import time
import pytz
import base64
import json
import tempfile
import warnings
from datetime import datetime, timedelta, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Optional (Voice Output - Offline)
try:
    import pyttsx3
    _PYTTSX3_OK = True
except Exception:
    _PYTTSX3_OK = False

load_dotenv()
warnings.filterwarnings("ignore")

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="AI-Driven Chiller Health Monitoring System", layout="wide")

UTC = pytz.utc
IST = pytz.timezone("Asia/Kolkata")

# =========================
# ENDPOINTS
# =========================
PRED_POWER_ENDPOINT = "http://172.20.20.30/aizen/predictions/az-pr-42c6f160e6/proj_c9nakipxa.chiller_power_consumption_pred/chiller_power_consumption"
ANOMALY_ENDPOINT = "http://172.20.20.30/aizen/predictions/az-pr-42c6f160e6/proj_c9nakipxa.anomaly_prediction/proj_c9nakipxa"
ACTUAL_FEATUREVIEW_ENDPOINT = "http://172.20.20.30/aizen/featureview/proj_c9nakipxa.chiller_power_consumption_feat_only_fview_online/features"

DESIGN_POWER_ENDPOINT = "http://172.20.20.30/aizen/predictions/az-pr-fca3ad5fb6/proj_c9nakipxa.chiller_design_power_prediction/chiller_design_power"
DESIGN_POWER_ACTUAL_ENDPOINT = "http://172.20.20.30/aizen/featureview/proj_c9nakipxa.chiller_design_power_training_feat_only_fview_online/features"

CHILLER_REQ_ENDPOINT = "http://172.20.20.30/aizen/predictions/az-pr-049170ca39/proj_c9nakipxa.chiller_requirement_pred/design_power_per_tr"

# =========================
# REALTIME (BMS) API - ODataConnector
# =========================
# Example response format:
# [{"PointName":".../WATER_FLOW","Value":null,"Timestamp":"2025-12-16T05:18:23.4734356+00:00","Quality":2147483652}]
REALTIME_BASE_URL = os.getenv("REALTIME_BASE_URL", "http://223.31.216.236/ODataConnector/rest/RealtimeData")
REALTIME_USERNAME = os.getenv("REALTIME_USERNAME", "")  # set in Streamlit secrets / env
REALTIME_PASSWORD = os.getenv("REALTIME_PASSWORD", "")  # set in Streamlit secrets / env

# PointName templates found in your notepad file:
# ac:T5/TR/CHILLER/T5-TR-CHILLER-1/WATER_FLOW
# Some sites may use underscores; we support both templates.
REALTIME_POINTNAME_TEMPLATE_DASH = "ac:T5/TR/CHILLER/T5-TR-CHILLER-{n}/{tag}"
REALTIME_POINTNAME_TEMPLATE_UNDERSCORE = "ac:T5/TR/CHILLER/T5_TR_CHILLER_{n}/{tag}"

# Tags (suffixes) we will fetch per chiller (edit freely)
REALTIME_TAGS = [
    "AMBIENT_TEMP",
    "INLET_TEMP",
    "OUTLET_TEMP",
    "POWER_CONSUMPTION",
    "WATER_FLOW",
    "TOTAL_LOAD",
    "CHILLER_CAPACITY",
    "TEMP_SETPOINT",
    "MAIN_VLV_OPENING",
    "BYPASS_VLV_OPENING",
    "COMP1_ACTIVE_LOAD",
    "COMP2_ACTIVE_LOAD",
    "CHILLER_EFFICIENCY_DESIGN",
    "CHILLER_EFFICIENCY_VARIATION",
    "CHILLER_CAPACITY_PERR",
]


# =========================
# CSV FILES
# =========================
POWER_CSV = "power_predictions.csv"
ANOMALY_CSV = "anomaly_predictions.csv"
DESIGN_POWER_CSV = "design_power_predictions.csv"
CHILLER_REQ_CSV = "chillers_required_predictions.csv"

START_DATE_UTC = datetime(2025, 11, 26, 0, 0, 0, tzinfo=timezone.utc)

DEFAULT_TERRACE_AMB_TEMP = 31.5
DEFAULT_TOTAL_IT_LOAD = 300.0

# =========================
# LLM CONFIG
# =========================
LLM_API_URL = "https://api.openai.com/v1/chat/completions"
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Optional STT endpoint (if you upload audio)
OPENAI_STT_URL = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")

# =========================================================
# AI AGENT
# =========================================================
def call_llm_agent(page_name: str, operator_question: str, numeric_context: dict, kpi_context: str, alert_context: str) -> str:
    if not OPENAI_API_KEY:
        return (
            "⚠️ LLM not configured. Please set OPENAI_API_KEY in your environment.\n\n"
            "Example:\n"
            "Windows PowerShell:\n"
            "$env:OPENAI_API_KEY=\"YOUR_KEY\"\n\n"
            "Linux/Mac:\n"
            "export OPENAI_API_KEY=\"YOUR_KEY\""
        )

    system_prompt = """
You are an AI copilot for a Tier-3 Data Center Chiller Plant (BMS/DCIM environment).

You are:
- a very good and very very expert data scientist,
- a very good and very very expert data center BMS CHILLER engineer,
- a very good and very very expert data and graph analyst.

Rules:
- Use ONLY numeric/structured context provided. Do not hallucinate sensors.
- Explain trends in: predicted vs actual vs design power, anomaly factors, and maintenance scoring.
- Give actionable operations steps: what to check NOW, what to trend, and when to escalate.
- Keep response concise: bullets + short paragraphs.
- Refer to chillers by ID.

Page Guidance:
PAGE 1 (Plant + Efficiency/Power):
- Relate ON/OFF selection to what we are seeing (filtered view).
- Compare Actual vs Predicted vs Design.

PAGE 2 (Anomaly):
- Interpret factor mix: Efficiency / Compressor / Temperature balance.
- Explain Good vs Bad zone distribution.

PAGE 3 (Maintenance):
- Explain why certain chillers are high/medium/low priority.
- If CMMS is behind model date, clearly state risk and urgency.

PAGE 4 (Capacity Planning):
- Explain how terrace ambient and IT load affect required chillers.
"""

    user_payload = {
        "page": page_name,
        "question": operator_question,
        "numeric_context": numeric_context,
        "kpi_context": kpi_context,
        "alert_context": alert_context,
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Use this JSON dashboard context and answer:\n\n" + json.dumps(user_payload, default=str, indent=2)},
    ]

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {"model": LLM_MODEL_NAME, "messages": messages, "temperature": 0.2, "max_tokens": 900}

    try:
        resp = requests.post(LLM_API_URL, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ LLM call failed: {e}"


# =========================================================
# VOICE: TTS (Offline pyttsx3) + Optional STT (OpenAI)
# =========================================================
def tts_offline_to_wav_bytes(text: str) -> bytes | None:
    """
    Offline voice output using pyttsx3 -> wav file -> bytes.
    Works if pyttsx3 is installed and has a system TTS backend.
    """
    if not _PYTTSX3_OK:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wav_path = f.name
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        with open(wav_path, "rb") as rf:
            b = rf.read()
        try:
            os.remove(wav_path)
        except Exception:
            pass
        return b
    except Exception:
        return None


def stt_transcribe_audio_openai(uploaded_file) -> str:
    """
    Optional: voice input. Requires OPENAI_API_KEY.
    The user uploads a WAV/MP3/M4A; we send to OpenAI whisper.
    """
    if not OPENAI_API_KEY:
        return "⚠️ OPENAI_API_KEY not set, cannot transcribe. Type your question instead."

    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type),
        }
        data = {"model": OPENAI_STT_MODEL}
        resp = requests.post(OPENAI_STT_URL, headers=headers, files=files, data=data, timeout=90)
        resp.raise_for_status()
        j = resp.json()
        return str(j.get("text", "")).strip() or "⚠️ No text returned from transcription."
    except Exception as e:
        return f"⚠️ Transcription failed: {e}"


# =========================================================
# UI THEME
# =========================================================
def _img_to_data_uri(path: str) -> str:
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{data}"
    except Exception:
        return ""


def set_custom_background_and_header():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #1b2735 0, #090a0f 45%, #000000 100%);
            color: #f0f0f0;
        }
        .sify-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.5rem 1.5rem;
            background-color: rgba(0, 0, 0, 0.92);
            border-bottom: 1px solid #333333;
            box-shadow: 0 2px 8px rgba(0,0,0,0.6);
        }
        .sify-header-title {
            color: #ffffff;
            font-size: 1.2rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .sify-header-subtitle {
            color: #d0d0d0;
            font-size: 0.8rem;
        }
        .left-nav {
            background-color: rgba(0, 0, 0, 0.88);
            padding: 0.75rem 0.5rem 1.5rem 0.5rem;
            height: 100vh;
            border-right: 1px solid #333333;
            border-radius: 0 16px 16px 0;
            box-shadow: 4px 0 12px rgba(0,0,0,0.6);
        }
        .left-nav-title {
            font-size: 0.75rem;
            color: #cccccc;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.35rem;
        }

        /* --- Chiller cards --- */
        .ch-card {
            background: rgba(0,0,0,0.65);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 8px 18px rgba(0,0,0,0.45);
        }
        .ch-top {
            background: #004b7a;
            padding: 8px 10px;
            font-weight: 800;
            font-size: 14px;
            letter-spacing: 0.04em;
            color: #fff;
            text-align: center;
        }
        .ch-status-on { background: #0aa000; color: #fff; padding: 6px 10px; text-align:center; font-weight:800; }
        .ch-status-off{ background: #b40000; color: #fff; padding: 6px 10px; text-align:center; font-weight:800; }

        .ch-body {
            padding: 10px 12px 10px 12px;
            font-size: 13px;
            color: #e7e7e7;
            line-height: 1.55;
            min-height: 220px;
        }
        .ch-k { color:#bdbdbd; font-weight:700; }
        .ch-v { color:#ffffff; font-weight:700; }

        section.main > div { padding-top: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    logo_uri = _img_to_data_uri("sifylogo.png")
    dc_uri = _img_to_data_uri("sifydcimage.png")

    header_html = f"""
    <div class="sify-header">
        <div>
            {'<img src="' + logo_uri + '" alt="Sify" style="height:46px;">' if logo_uri else ''}
        </div>
        <div style="display:flex; flex-direction:column; gap:2px;">
            <div class="sify-header-title">
                AI-Driven Chiller Health Monitoring System
            </div>
            <div class="sify-header-subtitle">
                MUM-03-T5-L1 | Sify Infiniti Spaces Limited – Tech Park, Rabale, Navi Mumbai
            </div>
        </div>
        <div style="margin-left:auto;">
            {'<img src="' + dc_uri + '" alt="Sify DC" style="height:64px; border-radius:10px;">' if dc_uri else ''}
        </div>
    </div>

    """
    st.markdown(header_html, unsafe_allow_html=True)


# =========================================================
# UTILITIES
# =========================================================
def safe_post(url, payload, retries=3, timeout=30):
    headers = {"Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


# =========================================================
# REALTIME (BMS) API HELPERS
# =========================================================
def _get_realtime_auth():
    # Priority: Streamlit secrets -> env -> blank
    try:
        u = st.secrets.get("REALTIME_USERNAME", REALTIME_USERNAME)
        p = st.secrets.get("REALTIME_PASSWORD", REALTIME_PASSWORD)
    except Exception:
        u, p = REALTIME_USERNAME, REALTIME_PASSWORD
    u = (u or "").strip()
    p = (p or "").strip()
    return (u, p) if (u and p) else None


def fetch_realtime_point(point_name: str, timeout: int = 20) -> dict:
    """Fetch one point. Returns dict: {point_name, value, ts_utc, quality, ok, error}."""
    url = REALTIME_BASE_URL
    params = {"PointName": point_name}
    auth = _get_realtime_auth()

    out = {"point_name": point_name, "value": None, "ts_utc": None, "quality": None, "ok": False, "error": None}
    try:
        r = requests.get(url, params=params, auth=auth, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        # API can return list[dict] as per your format
        rec = None
        if isinstance(j, list) and len(j) > 0 and isinstance(j[0], dict):
            rec = j[0]
        elif isinstance(j, dict):
            rec = j
        if not rec:
            out["error"] = "Empty response"
            return out

        out["value"] = rec.get("Value", None)
        out["quality"] = rec.get("Quality", None)
        ts = rec.get("Timestamp", None)
        if ts:
            out["ts_utc"] = pd.to_datetime(ts, errors="coerce", utc=True)
        out["ok"] = True
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def build_pointname(chiller_no: int, tag: str, prefer: str = "dash") -> str:
    if prefer == "underscore":
        return REALTIME_POINTNAME_TEMPLATE_UNDERSCORE.format(n=chiller_no, tag=tag)
    return REALTIME_POINTNAME_TEMPLATE_DASH.format(n=chiller_no, tag=tag)


def fetch_realtime_snapshot(chiller_numbers: list[int], prefer_template: str = "dash") -> pd.DataFrame:
    """Fetch a snapshot for multiple chillers. Returns a tidy dataframe."""
    rows = []
    for n in chiller_numbers:
        for tag in REALTIME_TAGS:
            pn = build_pointname(n, tag, prefer=prefer_template)
            rec = fetch_realtime_point(pn)
            rows.append({
                "chiller_no": n,
                "tag": tag,
                "point_name": pn,
                "value": rec.get("value"),
                "timestamp_utc": rec.get("ts_utc"),
                "quality": rec.get("quality"),
                "ok": rec.get("ok"),
                "error": rec.get("error"),
            })
    df = pd.DataFrame(rows)
    return df


def realtime_pivot(df_tidy: pd.DataFrame) -> pd.DataFrame:
    """Convert tidy snapshot to wide (one row per chiller)."""
    if df_tidy is None or df_tidy.empty:
        return pd.DataFrame()
    wide = df_tidy.pivot_table(index="chiller_no", columns="tag", values="value", aggfunc="first")
    wide.columns = [str(c) for c in wide.columns]
    wide = wide.reset_index()
    # attach max timestamp per chiller
    ts = df_tidy.groupby("chiller_no")["timestamp_utc"].max().reset_index().rename(columns={"timestamp_utc": "last_ts_utc"})
    out = wide.merge(ts, on="chiller_no", how="left")
    return out



def _to_plot_time(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_convert("UTC").dt.tz_localize(None)


def highlight_time_gaps(fig, df, value_cols, gap_color="rgba(0, 255, 0, 0.20)", default_freq_hours=1.0):
    if df is None or df.empty or "timestamp" not in df.columns:
        return fig

    tmp = df.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
    tmp = tmp.dropna(subset=["timestamp"]).sort_values("timestamp")
    if tmp.empty:
        return fig

    diffs = tmp["timestamp"].diff().dropna()
    if diffs.empty:
        return fig

    median_delta = diffs.median()
    if pd.isna(median_delta) or median_delta <= pd.Timedelta(0):
        median_delta = pd.to_timedelta(default_freq_hours, unit="h")

    gap_threshold = median_delta * 1.5
    ts_values = tmp["timestamp"].tolist()

    for i in range(len(ts_values) - 1):
        t0 = ts_values[i]
        t1 = ts_values[i + 1]
        if (t1 - t0) > gap_threshold:
            fig.add_vrect(x0=t0, x1=t1, fillcolor=gap_color, layer="below", line_width=0)

    if value_cols:
        cols_present = [c for c in value_cols if c in tmp.columns]
        if cols_present:
            null_mask = tmp[cols_present].isna().all(axis=1)
            freq_half = median_delta / 2
            for t in tmp.loc[null_mask, "timestamp"]:
                fig.add_vrect(x0=t - freq_half, x1=t + freq_half, fillcolor=gap_color, layer="below", line_width=0)

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color="rgba(0,255,0,0.8)"),
            name="Data Gap / Null Region",
            showlegend=True,
        )
    )
    return fig


def _drop_error_rows(df):
    if df is None or df.empty:
        return df
    bad = pd.Series(False, index=df.index)
    for c in df.columns:
        try:
            bad = bad | df[c].astype(str).str.contains("error_detail", na=False)
        except Exception:
            pass
    return df.loc[~bad].copy()


def _cast_h6_int64(df):
    if df is None or df.empty:
        return df
    if "hierarchy6" in df.columns:
        df["hierarchy6"] = pd.to_numeric(df["hierarchy6"], errors="coerce").astype("Int64")
    return df


def _normalize_ts(t):
    if isinstance(t, str):
        t = pd.to_datetime(t)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


# =========================================================
# CSV SCHEMA NORMALIZATION
# =========================================================
def _map_legacy_power_columns(df):
    if df is None or df.empty:
        return df
    cols = df.columns

    if "chiller_power_consumption" not in cols:
        if "POWER_CONSUMPTION" in cols:
            df["chiller_power_consumption"] = pd.to_numeric(df["POWER_CONSUMPTION"], errors="coerce")
        elif "power_consumption" in cols:
            df["chiller_power_consumption"] = pd.to_numeric(df["power_consumption"], errors="coerce")
        else:
            df["chiller_power_consumption"] = np.nan

    if "actual_chiller_power_consumption" not in cols:
        if "ACTUAL_POWER_CONSUMPTION" in cols:
            df["actual_chiller_power_consumption"] = pd.to_numeric(df["ACTUAL_POWER_CONSUMPTION"], errors="coerce")
        elif "actual_power_consumption" in cols:
            df["actual_chiller_power_consumption"] = pd.to_numeric(df["actual_power_consumption"], errors="coerce")
        else:
            df["actual_chiller_power_consumption"] = np.nan

    if "chiller_status" not in cols:
        if "ON_OFF_STATUS" in cols:
            df["chiller_status"] = pd.to_numeric(df["ON_OFF_STATUS"], errors="coerce")
        else:
            df["chiller_status"] = np.nan

    return df


def _map_legacy_design_columns(df):
    if df is None or df.empty:
        return df
    cols = df.columns
    if "chiller_design_power" not in cols and "CHILLER_DESIGN_POWER" in cols:
        df["chiller_design_power"] = pd.to_numeric(df["CHILLER_DESIGN_POWER"], errors="coerce")

    if "actual_chiller_design_power" not in cols:
        if "ACTUAL_CHILLER_DESIGN_POWER" in cols:
            df["actual_chiller_design_power"] = pd.to_numeric(df["ACTUAL_CHILLER_DESIGN_POWER"], errors="coerce")
        else:
            df["actual_chiller_design_power"] = pd.to_numeric(df.get("actual_chiller_design_power", np.nan), errors="coerce")

    return df


def ensure_power_csv(df):
    if df is None:
        df = pd.DataFrame()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    if "Chiller" not in df.columns and "hierarchy6" in df.columns:
        df["Chiller"] = np.nan
    df = _map_legacy_power_columns(df)
    if "ABS_ERROR_POWER" not in df.columns:
        df["ABS_ERROR_POWER"] = np.nan
    return df


def ensure_design_power_csv(df):
    if df is None:
        df = pd.DataFrame()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    if "Chiller" not in df.columns and "hierarchy6" in df.columns:
        df["Chiller"] = np.nan

    for col in [
        "chiller_design_power",
        "actual_chiller_design_power",
        "predicted_design_power_scaled",
        "actual_design_power_scaled",
        "ABS_ERROR_DESIGN_SCALED",
        "capacity_kw",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df = _map_legacy_design_columns(df)
    return df


def ensure_chiller_req_csv(df):
    if df is None:
        df = pd.DataFrame()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    if "Chiller" not in df.columns and "hierarchy6" in df.columns:
        df["Chiller"] = np.nan
    if "chiller_required_count" not in df.columns:
        if "chiller_count_required" in df.columns:
            df = df.rename(columns={"chiller_count_required": "chiller_required_count"})
        else:
            df["chiller_required_count"] = np.nan
    for col in ["terrace_amb_temp", "Total_IT_Load"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def write_power_csv(df):
    df = _drop_error_rows(df)
    df = _cast_h6_int64(df)
    df.to_csv(POWER_CSV, index=False)


def write_design_power_csv(df):
    df = _drop_error_rows(df)
    df = _cast_h6_int64(df)
    df.to_csv(DESIGN_POWER_CSV, index=False)


def write_chiller_req_csv(df):
    df = _drop_error_rows(df)
    df = _cast_h6_int64(df)
    df.to_csv(CHILLER_REQ_CSV, index=False)


def _chiller_on_series(df):
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    col = "chiller_status" if "chiller_status" in df.columns else ("ON_OFF_STATUS" if "ON_OFF_STATUS" in df.columns else None)
    if col:
        def _to_bool(x):
            try:
                if pd.isna(x):
                    return True
                return float(x) > 0.5
            except Exception:
                return True
        return df[col].apply(_to_bool)
    else:
        return pd.Series(True, index=df.index)


def compute_power_errors(df):
    df = ensure_power_csv(df.copy() if df is not None else pd.DataFrame())
    if df.empty:
        return df
    df["ACTUAL_NUM"] = pd.to_numeric(df.get("actual_chiller_power_consumption", np.nan), errors="coerce")
    df["PRED_NUM"] = pd.to_numeric(df.get("chiller_power_consumption", np.nan), errors="coerce")
    on_series = _chiller_on_series(df)
    valid_mask = (df["ACTUAL_NUM"].notna()) & (df["ACTUAL_NUM"] > 0) & (df["PRED_NUM"].notna()) & (on_series.astype(bool))
    df.loc[valid_mask, "ABS_ERROR_POWER"] = (df.loc[valid_mask, "ACTUAL_NUM"] - df.loc[valid_mask, "PRED_NUM"]).abs()
    df.drop(columns=["ACTUAL_NUM", "PRED_NUM"], inplace=True, errors="ignore")
    return df


def compute_design_scaled_errors(df):
    df = ensure_design_power_csv(df.copy() if df is not None else pd.DataFrame())
    if df.empty:
        return df

    cap_col = "capacity_kw" if "capacity_kw" in df.columns else None
    if cap_col is None:
        return df

    df["CAPACITY_NUM"] = pd.to_numeric(df[cap_col], errors="coerce")
    df["PRED_D_NUM"] = pd.to_numeric(df.get("chiller_design_power", np.nan), errors="coerce")
    df["ACT_D_NUM"] = pd.to_numeric(df.get("actual_chiller_design_power", np.nan), errors="coerce")

    ok_pred = df["CAPACITY_NUM"].notna() & df["PRED_D_NUM"].notna()
    ok_act = df["CAPACITY_NUM"].notna() & df["ACT_D_NUM"].notna()

    df.loc[ok_pred, "predicted_design_power_scaled"] = df.loc[ok_pred, "PRED_D_NUM"] * df.loc[ok_pred, "CAPACITY_NUM"]
    df.loc[ok_act, "actual_design_power_scaled"] = df.loc[ok_act, "ACT_D_NUM"] * df.loc[ok_act, "CAPACITY_NUM"]

    ok_err = df["predicted_design_power_scaled"].notna() & df["actual_design_power_scaled"].notna()
    df.loc[ok_err, "ABS_ERROR_DESIGN_SCALED"] = (df.loc[ok_err, "actual_design_power_scaled"] - df.loc[ok_err, "predicted_design_power_scaled"]).abs()

    df.drop(columns=["CAPACITY_NUM", "PRED_D_NUM", "ACT_D_NUM"], inplace=True, errors="ignore")
    return df


# =========================================================
# MODEL CLIENT
# =========================================================
class ChillerMLModels:
    def __init__(self):
        self.power_url = PRED_POWER_ENDPOINT
        self.anomaly_url = ANOMALY_ENDPOINT
        self.actual_url = ACTUAL_FEATUREVIEW_ENDPOINT
        self.design_power_url = DESIGN_POWER_ENDPOINT
        self.design_power_actual_url = DESIGN_POWER_ACTUAL_ENDPOINT
        self.chiller_req_url = CHILLER_REQ_ENDPOINT

    def fetch_power(self, time_str, hierarchy6_list):
        payload = [{"rest_request_id": str(i), "sourcetime": time_str, "hierarchy6": int(h)} for i, h in enumerate(hierarchy6_list)]
        return safe_post(self.power_url, payload) or []

    def fetch_anomaly(self, time_str, hierarchy6_list):
        payload = [{"rest_request_id": str(i), "sourcetime": time_str, "hierarchy6": int(h)} for i, h in enumerate(hierarchy6_list)]
        return safe_post(self.anomaly_url, payload) or []

    def fetch_actual_power(self, time_str, hierarchy6):
        payload = [{
            "featureview_name": "proj_c9nakipxa.chiller_power_consumption_feat_only_fview_online",
            "schema_name": "proj_c9nakipxa",
            "inputs": {"sourcetime": time_str, "hierarchy6": int(hierarchy6)},
            "outputs": ["chiller_power_consumption"]
        }]
        resp = safe_post(self.actual_url, payload)
        if not resp or not isinstance(resp, dict):
            return np.nan
        out = resp.get("outputs", {})
        if not isinstance(out, dict) or not out:
            return np.nan
        preferred_key = "proj_c9nakipxa.chiller_power_consumption_feat_only_fview_online"
        candidates = [preferred_key] + list(out.keys())
        for k in candidates:
            v = out.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v[0].get("chiller_power_consumption", np.nan)
        return np.nan

    def fetch_design_power(self, time_str, hierarchy6_list):
        payload = [{"rest_request_id": str(i), "sourcetime": time_str, "hierarchy6": int(h)} for i, h in enumerate(hierarchy6_list)]
        return safe_post(self.design_power_url, payload) or []

    def fetch_actual_design_power(self, time_str, hierarchy6):
        payload = [{
            "featureview_name": "proj_c9nakipxa.chiller_design_power_training_feat_only_fview_online",
            "schema_name": "proj_c9nakipxa",
            "inputs": {"sourcetime": time_str, "hierarchy6": int(hierarchy6)},
            "outputs": ["chiller_design_power"]
        }]
        resp = safe_post(self.design_power_actual_url, payload)
        if not resp or not isinstance(resp, dict):
            return np.nan
        out = resp.get("outputs", {})
        if not isinstance(out, dict) or not out:
            return np.nan
        preferred_key = "proj_c9nakipxa.chiller_design_power_training_feat_only_fview_online"
        candidates = [preferred_key] + list(out.keys())
        for k in candidates:
            v = out.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v[0].get("chiller_design_power", np.nan)
        return np.nan

    def fetch_chillers_required(self, time_str, hierarchy6_list, terrace_amb_temp, total_it_load):
        payload = [{
            "rest_request_id": f"prediction_test-{i + 1}",
            "sourcetime": time_str,
            "hierarchy6": int(h),
            "terrace_amb_temp": float(terrace_amb_temp),
            "Total_IT_Load": float(total_it_load)
        } for i, h in enumerate(hierarchy6_list)]
        return safe_post(self.chiller_req_url, payload) or []


# =========================================================
# INCREMENTAL UPDATER
# =========================================================
def update_predictions_incremental(ml, chillers, hierarchy6_list, batch_size=5):
    power_df = pd.read_csv(POWER_CSV, parse_dates=["timestamp"]) if os.path.exists(POWER_CSV) else pd.DataFrame()
    anomaly_df = pd.read_csv(ANOMALY_CSV, parse_dates=["timestamp"]) if os.path.exists(ANOMALY_CSV) else pd.DataFrame()
    design_power_df = pd.read_csv(DESIGN_POWER_CSV, parse_dates=["timestamp"]) if os.path.exists(DESIGN_POWER_CSV) else pd.DataFrame()
    ch_req_df = pd.read_csv(CHILLER_REQ_CSV, parse_dates=["timestamp"]) if os.path.exists(CHILLER_REQ_CSV) else pd.DataFrame()

    power_df = ensure_power_csv(power_df)
    anomaly_df = _cast_h6_int64(_drop_error_rows(anomaly_df))
    design_power_df = ensure_design_power_csv(design_power_df)
    ch_req_df = ensure_chiller_req_csv(ch_req_df)

    last_time_power = START_DATE_UTC - timedelta(hours=1)
    last_time_anomaly = START_DATE_UTC - timedelta(hours=1)
    last_time_design = START_DATE_UTC - timedelta(hours=1)
    last_time_ch_req = START_DATE_UTC - timedelta(hours=1)

    if not power_df.empty:
        last_time_power = max(last_time_power, _normalize_ts(power_df["timestamp"].max()))
    if not anomaly_df.empty:
        last_time_anomaly = max(last_time_anomaly, _normalize_ts(anomaly_df["timestamp"].max()))
    if not design_power_df.empty:
        last_time_design = max(last_time_design, _normalize_ts(design_power_df["timestamp"].max()))
    if not ch_req_df.empty:
        last_time_ch_req = max(last_time_ch_req, _normalize_ts(ch_req_df["timestamp"].max()))

    start_power = last_time_power + timedelta(hours=1)
    start_anomaly = last_time_anomaly + timedelta(hours=1)
    start_design = last_time_design + timedelta(hours=1)
    start_ch_req = last_time_ch_req + timedelta(hours=1)

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    new_power, new_anomaly, new_design, new_ch_req = [], [], [], []

    # POWER
    current = start_power
    while current <= end:
        time_str = current.strftime("%Y-%m-%d %H:%M:%S")
        for i in range(0, len(hierarchy6_list), batch_size):
            chunk = [int(x) for x in hierarchy6_list[i:i + batch_size]]
            chillers_list = list(chillers.keys())[i:i + batch_size]
            raw = ml.fetch_power(time_str, chunk)
            dfb = pd.DataFrame(raw) if isinstance(raw, list) else pd.DataFrame()
            if not dfb.empty:
                dfb["timestamp"] = current
                dfb["Chiller"] = chillers_list[:len(dfb)]
                dfb["hierarchy6"] = pd.Series(chunk[:len(dfb)]).astype("Int64")
                actuals = []
                for row in dfb.itertuples(index=False):
                    h = getattr(row, "hierarchy6", None)
                    if pd.isna(h):
                        actuals.append(np.nan)
                    else:
                        actuals.append(ml.fetch_actual_power(time_str, int(h)))
                dfb["actual_chiller_power_consumption"] = actuals
                new_power.append(dfb)
        current += timedelta(hours=1)

    # ANOMALY
    current = start_anomaly
    while current <= end:
        time_str = current.strftime("%Y-%m-%d %H:%M:%S")
        for i in range(0, len(hierarchy6_list), batch_size):
            chunk = [int(x) for x in hierarchy6_list[i:i + batch_size]]
            chillers_list = list(chillers.keys())[i:i + batch_size]
            raw = ml.fetch_anomaly(time_str, chunk)
            dfb = pd.DataFrame(raw) if isinstance(raw, list) else pd.DataFrame()
            if not dfb.empty:
                dfb["timestamp"] = current
                dfb["Chiller"] = chillers_list[:len(dfb)]
                dfb["hierarchy6"] = pd.Series(chunk[:len(dfb)]).astype("Int64")
                new_anomaly.append(dfb)
        current += timedelta(hours=1)

    # DESIGN POWER
    current = start_design
    while current <= end:
        time_str = current.strftime("%Y-%m-%d %H:%M:%S")
        for i in range(0, len(hierarchy6_list), batch_size):
            chunk = [int(x) for x in hierarchy6_list[i:i + batch_size]]
            chillers_list = list(chillers.keys())[i:i + batch_size]
            raw = ml.fetch_design_power(time_str, chunk)
            dfb = pd.DataFrame(raw) if isinstance(raw, list) else pd.DataFrame()
            if not dfb.empty:
                dfb["timestamp"] = current
                dfb["Chiller"] = chillers_list[:len(dfb)]
                dfb["hierarchy6"] = pd.Series(chunk[:len(dfb)]).astype("Int64")

                actuals = []
                for row in dfb.itertuples(index=False):
                    h = getattr(row, "hierarchy6", None)
                    if pd.isna(h):
                        actuals.append(np.nan)
                    else:
                        actuals.append(ml.fetch_actual_design_power(time_str, int(h)))
                dfb["actual_chiller_design_power"] = actuals

                new_design.append(dfb)
        current += timedelta(hours=1)

    # CHILLER REQUIRED
    current = start_ch_req
    while current <= end:
        time_str = current.strftime("%Y-%m-%d %H:%M:%S")
        for i in range(0, len(hierarchy6_list), batch_size):
            chunk = [int(x) for x in hierarchy6_list[i:i + batch_size]]
            chillers_list = list(chillers.keys())[i:i + batch_size]
            raw = ml.fetch_chillers_required(time_str, chunk, DEFAULT_TERRACE_AMB_TEMP, DEFAULT_TOTAL_IT_LOAD)
            dfb = pd.DataFrame(raw) if isinstance(raw, list) else pd.DataFrame()
            if not dfb.empty:
                dfb["timestamp"] = current
                dfb["Chiller"] = chillers_list[:len(dfb)]
                dfb["hierarchy6"] = pd.Series(chunk[:len(dfb)]).astype("Int64")

                if "chiller_required_count" not in dfb.columns and "chiller_count_required" in dfb.columns:
                    dfb = dfb.rename(columns={"chiller_count_required": "chiller_required_count"})
                if "chiller_required_count" not in dfb.columns:
                    dfb["chiller_required_count"] = np.nan

                dfb["terrace_amb_temp"] = DEFAULT_TERRACE_AMB_TEMP
                dfb["Total_IT_Load"] = DEFAULT_TOTAL_IT_LOAD
                new_ch_req.append(dfb)
        current += timedelta(hours=1)

    # WRITE BACK
    if new_power:
        newdf = pd.concat(new_power, ignore_index=True)
        power_df = pd.concat([power_df, newdf], ignore_index=True) if not power_df.empty else newdf
        power_df = power_df.drop_duplicates(subset=["timestamp", "Chiller"], keep="last")
        power_df = ensure_power_csv(power_df)
        power_df = compute_power_errors(power_df)
        write_power_csv(power_df)

    if new_anomaly:
        newdf = pd.concat(new_anomaly, ignore_index=True)
        anomaly_df = pd.concat([anomaly_df, newdf], ignore_index=True) if not anomaly_df.empty else newdf
        anomaly_df = anomaly_df.drop_duplicates(subset=["timestamp", "Chiller"], keep="last")
        anomaly_df = _cast_h6_int64(_drop_error_rows(anomaly_df))
        anomaly_df.to_csv(ANOMALY_CSV, index=False)

    if new_design:
        newdf = pd.concat(new_design, ignore_index=True)
        design_power_df = pd.concat([design_power_df, newdf], ignore_index=True) if not design_power_df.empty else newdf
        design_power_df = design_power_df.drop_duplicates(subset=["timestamp", "Chiller"], keep="last")
        design_power_df = ensure_design_power_csv(design_power_df)
        design_power_df = compute_design_scaled_errors(design_power_df)
        write_design_power_csv(design_power_df)

    if new_ch_req:
        newdf = pd.concat(new_ch_req, ignore_index=True)
        ch_req_df = pd.concat([ch_req_df, newdf], ignore_index=True) if not ch_req_df.empty else newdf
        ch_req_df = ch_req_df.drop_duplicates(subset=["timestamp", "Chiller"], keep="last")
        ch_req_df = ensure_chiller_req_csv(ch_req_df)
        write_chiller_req_csv(ch_req_df)

    return power_df, anomaly_df, design_power_df, ch_req_df


# =========================================================
# GLOBAL CHILLER GRID (30 units)
# =========================================================
def build_30_chiller_map() -> dict:
    """
    Creates 30 chiller IDs:
      CH-01 .. CH-30  => hierarchy6 1..30
    Also keeps a "T5_CHILLER_C_x" alias if you want later.
    """
    m = {}
    for i in range(1, 31):
        m[f"CH-{i:02d}"] = float(i)
    return m


def init_chiller_state(chiller_keys: list[str]):
    if "chiller_state" not in st.session_state:
        st.session_state.chiller_state = {k: True for k in chiller_keys}
    if "chiller_focus" not in st.session_state:
        st.session_state.chiller_focus = chiller_keys[0] if chiller_keys else None
    # default setpoints
    for k in chiller_keys:
        spk = f"sp_{k}"
        if spk not in st.session_state:
            st.session_state[spk] = 21.00


def get_on_chillers(all_chillers: list[str]) -> list[str]:
    state = st.session_state.get("chiller_state", {})
    return [c for c in all_chillers if bool(state.get(c, True))]


def _pick_latest_row(df: pd.DataFrame, chiller: str) -> pd.Series | None:
    if df is None or df.empty or "Chiller" not in df.columns:
        return None
    sub = df[df["Chiller"] == chiller].copy()
    if sub.empty or "timestamp" not in sub.columns:
        return None
    sub = sub.sort_values("timestamp")
    return sub.iloc[-1]


def _safe_num(x, fmt="{:.2f}", suffix=""):
    try:
        if x is None or pd.isna(x):
            return "—"
        return fmt.format(float(x)) + suffix
    except Exception:
        return "—"


def render_chiller_grid_3x10(power_df: pd.DataFrame, anomaly_df: pd.DataFrame, design_power_df: pd.DataFrame, all_chillers: list[str], realtime_wide: pd.DataFrame | None = None):
    st.markdown("## Chiller Plant – 30 Units (3 × 10 Grid)")

    # 10 columns per row like screenshot
    per_row = 10
    rows = [all_chillers[i:i + per_row] for i in range(0, len(all_chillers), per_row)]

    # A compact explanation
    st.caption("Toggle a chiller ON/OFF below. ON chillers are included in graphs/tables across all pages.")

    for r in rows:
        cols = st.columns(per_row, gap="small")
        for idx, ch in enumerate(r):
            with cols[idx]:
                # SP box (top of each card like screenshot)
                st.number_input(f"SP {ch}", min_value=5.0, max_value=30.0, step=0.1, format="%.2f", key=f"sp_{ch}")

                is_on = bool(st.session_state.chiller_state.get(ch, True))
                status_html = "ch-status-on" if is_on else "ch-status-off"
                status_txt = "STATUS: ON" if is_on else "STATUS: OFF"

                # Latest values (use what's available)
                p_last = _pick_latest_row(power_df, ch)
                a_last = _pick_latest_row(anomaly_df, ch) if anomaly_df is not None else None
                d_last = _pick_latest_row(design_power_df, ch)

                # Realtime override (if provided): uses your ODataConnector points
                rt = None
                try:
                    if realtime_wide is not None and (not realtime_wide.empty):
                        ch_no = int(str(ch).split("-")[-1])
                        match = realtime_wide[realtime_wide["chiller_no"] == ch_no]
                        if not match.empty:
                            rt = match.iloc[0].to_dict()
                except Exception:
                    rt = None

                def _rt(tag, fallback=np.nan):
                    if rt is None:
                        return fallback
                    v = rt.get(tag, fallback)
                    return v

                # Map parameters (show if exists, else "—")
                # NOTE: your endpoints primarily return power; other params will show — unless present in data
                supply = p_last.get("supply_temp", np.nan) if isinstance(p_last, pd.Series) else np.nan
                inlet = p_last.get("chilled_water_inlet_c", p_last.get("chw_inlet_temperature", np.nan)) if isinstance(p_last, pd.Series) else np.nan
                outlet = p_last.get("chilled_water_outlet_c", p_last.get("chw_outlet_temperature", np.nan)) if isinstance(p_last, pd.Series) else np.nan
                ambient = p_last.get("ambient_temp_c", np.nan) if isinstance(p_last, pd.Series) else np.nan
                comp1 = p_last.get("compressor_1_active_load", np.nan) if isinstance(p_last, pd.Series) else np.nan
                comp2 = p_last.get("compressor_2_active_load", np.nan) if isinstance(p_last, pd.Series) else np.nan
                flow = p_last.get("flow_m3_hr", p_last.get("flow", np.nan)) if isinstance(p_last, pd.Series) else np.nan

                # If realtime snapshot exists, prefer it for card KPIs
                if rt is not None:
                    ambient = _rt("AMBIENT_TEMP", ambient)
                    inlet = _rt("INLET_TEMP", inlet)
                    outlet = _rt("OUTLET_TEMP", outlet)
                    flow = _rt("WATER_FLOW", flow)
                    comp1 = _rt("COMP1_ACTIVE_LOAD", comp1)
                    comp2 = _rt("COMP2_ACTIVE_LOAD", comp2)

                pred_kw = p_last.get("chiller_power_consumption", np.nan) if isinstance(p_last, pd.Series) else np.nan
                act_kw = p_last.get("actual_chiller_power_consumption", np.nan) if isinstance(p_last, pd.Series) else np.nan
                if rt is not None:
                    # Realtime POWER_CONSUMPTION is treated as 'Actual' for the card
                    act_kw = _rt("POWER_CONSUMPTION", act_kw)

                # If "actual" missing, show predicted in Power like screenshot
                power_show = act_kw if pd.notna(act_kw) else pred_kw

                # Build card body
                sp_val = st.session_state.get(f"sp_{ch}", 21.0)
                card_html = f"""
                <div class="ch-card">
                    <div class="ch-top">{ch}</div>
                    <div class="{status_html}">{status_txt}</div>
                    <div class="ch-body">
                        <div><span class="ch-k">Setpoint:</span> <span class="ch-v">{_safe_num(sp_val, "{:.1f}", "°C")}</span></div>
                        <div><span class="ch-k">Supply:</span> <span class="ch-v">{_safe_num(supply, "{:.2f}", "°C")}</span></div>
                        <div><span class="ch-k">Inlet:</span> <span class="ch-v">{_safe_num(inlet, "{:.2f}", "°C")}</span></div>
                        <div><span class="ch-k">Outlet:</span> <span class="ch-v">{_safe_num(outlet, "{:.2f}", "°C")}</span></div>
                        <div><span class="ch-k">Ambient:</span> <span class="ch-v">{_safe_num(ambient, "{:.2f}", "°C")}</span></div>
                        <div style="margin-top:6px;"></div>
                        <div><span class="ch-k">Comp-1:</span> <span class="ch-v">{_safe_num(comp1, "{:.0f}", "%")}</span></div>
                        <div><span class="ch-k">Comp-2:</span> <span class="ch-v">{_safe_num(comp2, "{:.0f}", "%")}</span></div>
                        <div><span class="ch-k">Power:</span> <span class="ch-v">{_safe_num(power_show, "{:.1f}", " kW")}</span></div>
                        <div><span class="ch-k">Flow:</span> <span class="ch-v">{_safe_num(flow, "{:.2f}", " m³/hr")}</span></div>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

                # Toggle button (like screenshot)
                if st.button(f"Toggle {ch}", key=f"toggle_{ch}"):
                    st.session_state.chiller_state[ch] = not bool(st.session_state.chiller_state.get(ch, True))
                    # keep focus aligned
                    st.session_state.chiller_focus = ch
                    st.rerun()


# =========================================================
# ANOMALY FEATURES + DIAGNOSIS (Page 2)
# =========================================================
def detect_anomaly_factors(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    score_col = None
    for c in out.columns:
        if ("anomaly" in c.lower()) and ("score" in c.lower()):
            score_col = c
            break
    out["anomaly_score"] = pd.to_numeric(out[score_col], errors="coerce") if score_col else np.nan

    anom_flag_col = None
    for c in out.columns:
        if c.lower() in ("anomaly", "is_anomaly", "anomaly_flag"):
            anom_flag_col = c
            break

    if anom_flag_col:
        def _is_anom(x):
            try:
                if pd.isna(x):
                    return False
                s = str(x).strip().lower()
                if s in ("true", "1", "-1", "yes"):
                    return True
                f = float(x)
                return f in (1.0, -1.0)
            except Exception:
                return False
        out["is_anomaly"] = out[anom_flag_col].apply(_is_anom)
    else:
        out["is_anomaly"] = out["anomaly_score"].apply(lambda x: pd.notna(x) and abs(float(x)) > 0.5)

    eff_cols = [c for c in out.columns if "eff" in c.lower() and ("anom" in c.lower() or "factor" in c.lower())]
    comp_cols = [c for c in out.columns if "compress" in c.lower() and ("anom" in c.lower() or "factor" in c.lower())]
    temp_cols = [c for c in out.columns if ("temp" in c.lower() or "dt" in c.lower()) and ("anom" in c.lower() or "factor" in c.lower())]

    if eff_cols:
        out["eff_factor"] = pd.to_numeric(out[eff_cols[0]], errors="coerce").fillna(0).clip(lower=0)
    else:
        eff_val_col = None
        for c in out.columns:
            if c.lower() in ("chiller_efficiency", "efficiency", "kw_tr", "kw/tr"):
                eff_val_col = c
                break
        if eff_val_col:
            out["eff_factor"] = (pd.to_numeric(out[eff_val_col], errors="coerce") > 1.10).astype(int)
        else:
            out["eff_factor"] = 0

    if comp_cols:
        out["comp_factor"] = pd.to_numeric(out[comp_cols[0]], errors="coerce").fillna(0).clip(lower=0)
    else:
        c1 = None
        c2 = None
        for c in out.columns:
            if c.lower() in ("compressor_1_active_load", "compressor1_active_load", "compressor_1_load", "comp1_load"):
                c1 = c
            if c.lower() in ("compressor_2_active_load", "compressor2_active_load", "compressor_2_load", "comp2_load"):
                c2 = c
        if c1 and c2:
            a = pd.to_numeric(out[c1], errors="coerce")
            b = pd.to_numeric(out[c2], errors="coerce")
            out["comp_factor"] = (abs(a - b) > 15).astype(int)
        else:
            out["comp_factor"] = 0

    if temp_cols:
        out["temp_factor"] = pd.to_numeric(out[temp_cols[0]], errors="coerce").fillna(0).clip(lower=0)
    else:
        chw_in = None
        chw_out = None
        for c in out.columns:
            if c.lower() in ("chw_inlet_temperature", "chilled_water_inlet_c", "chw_inlet_temp", "chw_in"):
                chw_in = c
            if c.lower() in ("chw_outlet_temperature", "chilled_water_outlet_c", "chw_outlet_temp", "chw_out"):
                chw_out = c
        if chw_in and chw_out:
            diff = pd.to_numeric(out[chw_in], errors="coerce") - pd.to_numeric(out[chw_out], errors="coerce")
            out["temp_factor"] = (diff < 3).astype(int)
        else:
            out["temp_factor"] = 0

    out["eff_factor"] = pd.to_numeric(out["eff_factor"], errors="coerce").fillna(0)
    out["comp_factor"] = pd.to_numeric(out["comp_factor"], errors="coerce").fillna(0)
    out["temp_factor"] = pd.to_numeric(out["temp_factor"], errors="coerce").fillna(0)

    return out


def diagnose_issues_from_anomaly(df_recent: pd.DataFrame) -> list:
    issues = []
    if df_recent is None or df_recent.empty or "Chiller" not in df_recent.columns:
        return issues

    g = df_recent.groupby("Chiller", dropna=False)
    for ch, d in g:
        if pd.isna(ch):
            continue
        eff_cnt = float(d["eff_factor"].sum()) if "eff_factor" in d.columns else 0
        comp_cnt = float(d["comp_factor"].sum()) if "comp_factor" in d.columns else 0
        temp_cnt = float(d["temp_factor"].sum()) if "temp_factor" in d.columns else 0
        total = eff_cnt + comp_cnt + temp_cnt

        if total <= 0:
            issues.append({"chiller": ch, "severity": "Low", "issues": ["No critical issues detected"]})
            continue

        msg = []
        if eff_cnt > 0:
            msg.append("High kW/TR — check heat exchangers / fouling / condenser approach")
        if comp_cnt > 0:
            msg.append("Compressor behaviour anomaly — check load sharing / vibration / oil / amps")
        if temp_cnt > 0:
            msg.append("Low ΔT / temperature balance issue — inspect flow balance / valves / sensors")

        severity = "Medium"
        if total >= 10:
            severity = "High"
        elif total >= 3:
            severity = "Medium"
        else:
            severity = "Low"

        issues.append({"chiller": ch, "severity": severity, "issues": msg})

    order = {"High": 3, "Medium": 2, "Low": 1}
    issues = sorted(issues, key=lambda x: order.get(x["severity"], 1), reverse=True)
    return issues


def display_issue_cards(issues):
    bg_color = "#3a3a3a"
    border_colors = {"High": "#ff4c4c", "Medium": "#ffb347", "Low": "#8bc34a"}
    text_color = "#e6e2e2"

    for issue in issues:
        chiller_id = issue.get("chiller", "Unknown")
        severity = issue.get("severity", "Medium")
        border_color = border_colors.get(severity, "#ffb347")
        issue_list_html = "".join([f"<li>{i}</li>" for i in issue.get("issues", [])])

        st.markdown(
            f"""
            <div style="background-color:{bg_color}; border-left: 6px solid {border_color};
                padding: 15px; margin: 10px 0; border-radius: 5px; color: {text_color};">
                <h3 style="margin:0;">{chiller_id} — Severity: {severity}</h3>
                <ul style="margin: 5px 0 0 15px;">
                    {issue_list_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# MAINTENANCE (Page 3)
# =========================================================
def _parse_cmms_date(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series.astype(str).str.strip(), errors="coerce", dayfirst=True, infer_datetime_format=True)
    mask = dt.isna()
    if mask.any():
        dt2 = pd.to_datetime(series.astype(str).str.strip(), errors="coerce", dayfirst=False, infer_datetime_format=True)
        dt.loc[mask] = dt2.loc[mask]
    return dt


def predict_maintenance_from_power(power_df: pd.DataFrame) -> pd.DataFrame:
    if power_df is None or power_df.empty or "Chiller" not in power_df.columns:
        return pd.DataFrame()

    df = power_df.copy()

    eff_col = None
    for c in df.columns:
        if c.lower() in ("chiller_efficiency", "efficiency", "kw_tr", "kw/tr", "chiller_kw_tr"):
            eff_col = c
            break

    comp1 = None
    comp2 = None
    for c in df.columns:
        if c.lower() in ("compressor_1_active_load", "comp1_load", "compressor1_active_load"):
            comp1 = c
        if c.lower() in ("compressor_2_active_load", "comp2_load", "compressor2_active_load"):
            comp2 = c

    rows = []
    for ch, d in df.groupby("Chiller"):
        if pd.isna(ch) or d.empty:
            continue

        if eff_col:
            avg_eff = pd.to_numeric(d[eff_col], errors="coerce").mean()
        else:
            avg_eff = pd.to_numeric(d["chiller_power_consumption"], errors="coerce").mean() / 300.0

        d_sorted = d.sort_values("timestamp") if "timestamp" in d.columns else d
        n = max(1, int(len(d_sorted) * 0.2))
        if len(d_sorted) >= 10:
            recent = pd.to_numeric(d_sorted.get(eff_col, d_sorted["chiller_power_consumption"]), errors="coerce").tail(n).mean()
            older = pd.to_numeric(d_sorted.get(eff_col, d_sorted["chiller_power_consumption"]), errors="coerce").head(n).mean()
            trend = (recent - older) if pd.notna(recent) and pd.notna(older) else 0
        else:
            trend = 0

        if comp1 and comp2:
            a = pd.to_numeric(d[comp1], errors="coerce")
            b = pd.to_numeric(d[comp2], errors="coerce")
            comp_var = pd.concat([a, b], axis=1).var().mean()
            comp_imb = abs(a.mean() - b.mean())
        else:
            comp_var = np.nan
            comp_imb = np.nan

        score_eff = 0.0
        score_trend = 0.0
        score_var = 0.0
        score_imb = 0.0
        score_noise = float(np.random.uniform(0, 5))

        if pd.notna(avg_eff):
            if avg_eff > 1.10:
                score_eff += 40
            elif avg_eff > 0.90:
                score_eff += 20
            elif avg_eff > 0.80:
                score_eff += 10

        if trend > 0.05:
            score_trend += 30
        elif trend > 0.02:
            score_trend += 15

        if pd.notna(comp_var) and comp_var > 300:
            score_var += 15
        if pd.notna(comp_imb) and comp_imb > 15:
            score_imb += 10

        maintenance_score = max(0.0, score_eff + score_trend + score_var + score_imb + score_noise)

        if maintenance_score >= 60:
            mtype = "Corrective"
            priority = "High"
            est_days = int(np.random.randint(7, 21))
        elif maintenance_score >= 35:
            mtype = "Preventive"
            priority = "Medium"
            est_days = int(np.random.randint(30, 90))
        else:
            mtype = "Routine"
            priority = "Low"
            est_days = int(np.random.randint(90, 180))

        next_date = (datetime.now() + timedelta(days=est_days))

        rows.append({
            "chiller_id": str(ch),
            "maintenance_score": float(maintenance_score),
            "maintenance_type": mtype,
            "priority": priority,
            "avg_efficiency": float(avg_eff) if pd.notna(avg_eff) else np.nan,
            "efficiency_trend": float(trend),
            "compressor_variance": float(comp_var) if pd.notna(comp_var) else np.nan,
            "compressor_imbalance": float(comp_imb) if pd.notna(comp_imb) else np.nan,
            "est_days": int(est_days),
            "next_maintenance": next_date,
            "records_count": int(len(d)),
            "score_efficiency": float(score_eff),
            "score_trend": float(score_trend),
            "score_variance": float(score_var),
            "score_imbalance": float(score_imbalance) if (score_imbalance := score_imb) or True else float(score_imb),
            "score_noise": float(score_noise),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["next_maintenance"] = pd.to_datetime(out["next_maintenance"], errors="coerce")
    return out


def display_maintenance_cards(maintenance_df):
    bg_color = "#3a3a3a"
    border_colors = {"High": "#f44336", "Medium": "#ff9800", "Low": "#4caf50"}
    text_color = "#e6e2e2"
    priority_order = {"High": 3, "Medium": 2, "Low": 1}

    df = maintenance_df.copy()
    if df.empty:
        return

    df["priority_score"] = df["priority"].map(priority_order).fillna(0)
    df = df.sort_values(["priority_score", "maintenance_score"], ascending=[False, False])

    for _, row in df.iterrows():
        border_color = border_colors.get(row["priority"], "#ff9800")
        nd = pd.to_datetime(row["next_maintenance"], errors="coerce")
        nd_str = nd.strftime("%Y-%m-%d") if pd.notna(nd) else "-"
        st.markdown(
            f"""
            <div style="background-color:{bg_color}; border-left: 6px solid {border_color};
                padding: 15px; margin: 10px 0; border-radius: 5px; color: {text_color};">
                <h3 style="margin:0;">Chiller {row['chiller_id']} - {row['priority']} Priority</h3>
                <p><strong>Maintenance Type:</strong> {row['maintenance_type']}</p>
                <p><strong>Predicted Date:</strong> {nd_str} ({int(row['est_days'])} day(s))</p>
                <p><strong>Health Score:</strong> {max(0, 100-row['maintenance_score']):.0f}/100</p>
                <p><strong>Avg Efficiency:</strong> {_safe_num(row['avg_efficiency'], "{:.3f}", "")}</p>
                <p><strong>Efficiency Trend:</strong> {_safe_num(row['efficiency_trend'], "{:+.3f}", "")}</p>
                <p><strong>Compressor Balance:</strong> {_safe_num(row['compressor_imbalance'], "{:.1f}", "")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_predictive_maintenance_dashboard(maintenance_df):
    st.markdown("## Predictive Maintenance Dashboard")

    if maintenance_df.empty:
        st.info("No maintenance predictions available.")
        return

    high_priority = int((maintenance_df["priority"] == "High").sum())
    medium_priority = int((maintenance_df["priority"] == "Medium").sum())
    low_priority = int((maintenance_df["priority"] == "Low").sum())
    fleet_health_score = max(0, 100 - float(maintenance_df["maintenance_score"].mean()))

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("High Priority", high_priority)
    with col2: st.metric("Medium Priority", medium_priority)
    with col3: st.metric("Low Priority", low_priority)
    with col4: st.metric("Fleet Health Score", f"{fleet_health_score:.0f}/100")

    bar_df = maintenance_df.sort_values("maintenance_score", ascending=False)
    fig_bar = px.bar(
        bar_df, x="chiller_id", y="maintenance_score", color="priority",
        barmode="group", title="Maintenance Priority by Chiller", template="plotly_dark",
        color_discrete_map={"High": "#f44336", "Medium": "#ff9800", "Low": "#4caf50"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Maintenance Score Breakdown by Factor")
    comp_df = maintenance_df.copy()
    comp_df["Health Score"] = comp_df["maintenance_score"].apply(lambda s: max(0.0, 100.0 - s))
    comp_df["Avg Efficiency"] = comp_df["score_efficiency"].clip(lower=0.0)
    comp_df["Efficiency Trend"] = comp_df["score_trend"].clip(lower=0.0)
    comp_df["Compressor variance"] = comp_df["score_variance"].clip(lower=0.0)
    comp_df["Compressor imbalance"] = comp_df["score_imbalance"].clip(lower=0.0)
    comp_df["Noise factor"] = comp_df["score_noise"].clip(lower=0.0)
    comp_df["Data volume"] = comp_df["records_count"].astype(float)

    melt_cols = ["Health Score", "Avg Efficiency", "Efficiency Trend", "Compressor variance", "Compressor imbalance", "Noise factor", "Data volume"]
    comp_melt = comp_df.melt(id_vars="chiller_id", value_vars=melt_cols, var_name="Component", value_name="Value")
    fig_stack = px.bar(comp_melt, x="chiller_id", y="Value", color="Component", barmode="stack",
                       title="Per-Chiller Maintenance Components (stacked)", template="plotly_dark")
    st.plotly_chart(fig_stack, use_container_width=True)

    st.subheader("Maintenance Timeline")
    timeline_df = maintenance_df.sort_values("est_days")
    fig_timeline = px.scatter(
        timeline_df, x="est_days", y="chiller_id", size="maintenance_score", color="priority",
        title="Maintenance Timeline (Days Until Next Maintenance)", template="plotly_dark", size_max=30,
        color_discrete_map={"High": "#f44336", "Medium": "#ff9800", "Low": "#4caf50"},
    )
    fig_timeline.update_layout(height=420)
    st.plotly_chart(fig_timeline, use_container_width=True)


def render_cmms_vs_model_table(maintenance_df: pd.DataFrame):
    st.markdown("### CMMS vs Model-Predicted Maintenance (Per Chiller)")

    cmms_possible_paths = [
        "/mnt/data/Chiller-CMMS Module timeline dataU.csv",
        "Chiller-CMMS Module timeline dataU.csv",
    ]
    cmms_path = next((p for p in cmms_possible_paths if os.path.exists(p)), None)

    if cmms_path is None:
        st.warning(
            "CMMS CSV file 'Chiller-CMMS Module timeline dataU.csv' not found.\n"
            "Place it in the app folder or /mnt/data to enable the comparison table."
        )
        return

    try:
        cmms_df_raw = pd.read_csv(cmms_path)
        if cmms_df_raw.shape[1] < 6:
            st.warning("CMMS CSV does not have enough columns (needs at least Column B and Column F).")
            return

        ch_col_excel = cmms_df_raw.columns[1]
        sched_col_excel = cmms_df_raw.columns[5]

        cmms_df = cmms_df_raw[[ch_col_excel, sched_col_excel]].copy()
        cmms_df = cmms_df.rename(columns={ch_col_excel: "chiller_id_cmms", sched_col_excel: "scheduled_maintenance"})
        cmms_df["chiller_id_cmms"] = cmms_df["chiller_id_cmms"].astype(str).str.strip()

        cmms_df["scheduled_maintenance"] = _parse_cmms_date(cmms_df["scheduled_maintenance"])
        cmms_df = cmms_df.dropna(subset=["scheduled_maintenance"])

        cmms_latest = (cmms_df.sort_values("scheduled_maintenance").groupby("chiller_id_cmms", as_index=False).tail(1).reset_index(drop=True))

        model_df = maintenance_df.copy()
        model_df["chiller_id_join"] = model_df["chiller_id"].astype(str).str.strip()
        model_df["predicted_maintenance"] = pd.to_datetime(model_df["next_maintenance"], errors="coerce")
        model_df = model_df.dropna(subset=["predicted_maintenance"])

        comp = pd.merge(model_df, cmms_latest, left_on="chiller_id_join", right_on="chiller_id_cmms", how="inner")

        if comp.empty:
            st.info("No overlapping chillers found between CMMS CSV and model predictions (check naming).")
            return

        comp["days_delta"] = (comp["scheduled_maintenance"] - comp["predicted_maintenance"]).dt.days
        comp["status"] = np.where(
            comp["scheduled_maintenance"] <= comp["predicted_maintenance"],
            "On time / Ahead (CMMS ≤ Model)",
            "Behind (CMMS > Model)",
        )

        comp_display = comp[["chiller_id", "predicted_maintenance", "scheduled_maintenance", "days_delta", "status", "maintenance_type", "priority", "maintenance_score"]].copy()

        comp_display["predicted_maintenance"] = pd.to_datetime(comp_display["predicted_maintenance"], errors="coerce").dt.strftime("%Y-%m-%d")
        comp_display["scheduled_maintenance"] = pd.to_datetime(comp_display["scheduled_maintenance"], errors="coerce").dt.strftime("%Y-%m-%d")

        comp_display = comp_display.sort_values(["days_delta"], ascending=False)

        def _row_color(row):
            if str(row["status"]).startswith("Behind"):
                return ["background-color: rgba(244, 67, 54, 0.35);"] * len(row)
            else:
                return ["background-color: rgba(76, 175, 80, 0.35);"] * len(row)

        styled = comp_display.style.apply(_row_color, axis=1).format({"days_delta": "{:+d}", "maintenance_score": "{:.1f}"})
        st.dataframe(styled, use_container_width=True)

        st.markdown(
            """
            **Legend**  
            ✅ **Green** = CMMS scheduled **on/before** model date (on time / ahead)  
            ❌ **Red** = CMMS scheduled **after** model date (behind)  
            **Days (CMMS − Model):** `+` = behind, `-` = ahead
            """
        )
    except Exception as e:
        st.error(f"Unable to read / process CMMS CSV file: {e}")


# =========================================================
# PAGE 1: PLANT GRID + EFFICIENCY/POWER
# =========================================================
def page_1_plant_and_efficiency(power_df, design_power_df, anomaly_df, all_chillers, on_chillers, realtime_wide=None):
    # --- Plant Grid at top ---
    render_chiller_grid_3x10(power_df, anomaly_df, design_power_df, all_chillers, realtime_wide=realtime_wide)

    st.markdown("---")
    st.markdown("## Predicted vs Actual vs Design Power — (Filtered by ON chillers)")

    if len(on_chillers) == 0:
        st.warning("All chillers are OFF. Turn ON at least one chiller to see charts.")
        return

    # Show charts only for ON chillers (exact requirement)
    selected = on_chillers

    # Focus dropdown (also filtered)
    ai_chiller = st.selectbox("AI Focus Chiller (ON only):", options=selected, index=0, key="ai_focus_page1")

    for ch in selected:
        sub_p = power_df[power_df["Chiller"] == ch].copy() if not power_df.empty else pd.DataFrame()
        sub_d = design_power_df[design_power_df["Chiller"] == ch].copy() if not design_power_df.empty else pd.DataFrame()

        if sub_p.empty and sub_d.empty:
            st.write(f"No data for {ch}")
            continue

        fig = go.Figure()

        if not sub_p.empty:
            sub_p = sub_p.sort_values("timestamp")
            fig.add_trace(go.Scatter(
                x=sub_p["timestamp"], y=pd.to_numeric(sub_p["chiller_power_consumption"], errors="coerce"),
                mode="lines+markers", name="Predicted Power"
            ))
            fig.add_trace(go.Scatter(
                x=sub_p["timestamp"], y=pd.to_numeric(sub_p["actual_chiller_power_consumption"], errors="coerce"),
                mode="lines+markers", name="Actual Power"
            ))

        if not sub_d.empty:
            sub_d = sub_d.sort_values("timestamp")
            if "predicted_design_power_scaled" in sub_d.columns and sub_d["predicted_design_power_scaled"].notna().sum() > 0:
                ds = pd.to_numeric(sub_d["predicted_design_power_scaled"], errors="coerce")
            else:
                ds = pd.to_numeric(sub_d.get("chiller_design_power", np.nan), errors="coerce")

            fig.add_trace(go.Scatter(
                x=sub_d["timestamp"], y=ds,
                mode="lines+markers", name="Design Power (Predicted)",
                line=dict(color="yellow")
            ))

        fig = highlight_time_gaps(fig, sub_p, value_cols=["chiller_power_consumption", "actual_chiller_power_consumption"])

        fig.update_layout(
            title=f"{ch} — Power vs Design (Predicted / Actual)",
            xaxis_title="Time",
            yaxis_title="Power / Design Power (kW)",
            template="plotly_dark",
            height=320,
            legend=dict(orientation="v", x=1.02, y=1.0),
            margin=dict(l=40, r=30, t=45, b=40),
        )
        fig.update_xaxes(type="date")
        st.plotly_chart(fig, use_container_width=True)

    # --- AI Q&A + Voice Agent ---
    st.markdown("## Voice to Voice — BMS AI Agent (Page 1)")
    st.caption("Type a question OR upload an audio file (optional STT requires OPENAI_API_KEY). The AI answer can be played as voice (offline TTS).")

    colA, colB = st.columns([2, 1.2], gap="large")
    with colA:
        preset_q = st.selectbox(
            "Quick question:",
            options=[
                "Explain why chiller power has increased and whether it is beyond design limits.",
                "Highlight which time window needs escalation today for this chiller.",
                "What immediate checks should operator do for sustained deviation vs design?",
            ],
            index=0,
            key="p1_preset",
        )
        typed_q = st.text_area("Or type your own BMS operator question:", value="", key="p1_typed")
        operator_q = typed_q.strip() or preset_q

        audio_up = st.file_uploader("Optional: Upload voice question (wav/mp3/m4a) for transcription", type=["wav", "mp3", "m4a"], key="p1_audio")

        if st.button("Ask BMS AI Agent", type="primary", key="p1_ask"):
            if audio_up is not None and (typed_q.strip() == ""):
                operator_q = stt_transcribe_audio_openai(audio_up)

            # numeric context for focus chiller (last 24h)
            sub_p = power_df[power_df["Chiller"] == ai_chiller].copy()
            sub_d = design_power_df[design_power_df["Chiller"] == ai_chiller].copy()

            last_ts = sub_p["timestamp"].max() if not sub_p.empty else sub_d["timestamp"].max()
            start_ts = last_ts - timedelta(hours=24) if pd.notna(last_ts) else None
            sub_p_24 = sub_p[sub_p["timestamp"] >= start_ts].copy() if start_ts is not None else sub_p.copy()
            sub_d_24 = sub_d[sub_d["timestamp"] >= start_ts].copy() if start_ts is not None else sub_d.copy()

            def _stats(x):
                x = pd.to_numeric(x, errors="coerce")
                if x.notna().sum() == 0:
                    return {"min": None, "max": None, "mean": None, "last": None}
                return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "last": float(x.iloc[-1])}

            design_series = sub_d_24.get("predicted_design_power_scaled", sub_d_24.get("chiller_design_power", np.nan))

            num_ctx = {
                "selected_on_chillers": on_chillers,
                "focus_chiller": ai_chiller,
                "window_hours": 24,
                "points_pred_power": int(len(sub_p_24)),
                "predicted_power_kw": _stats(sub_p_24.get("chiller_power_consumption", np.nan)),
                "actual_power_kw": _stats(sub_p_24.get("actual_chiller_power_consumption", np.nan)),
                "design_power_kw": _stats(design_series),
            }

            kpi_ctx = (
                "- Key KPIs: Actual vs Predicted, Actual vs Design.\n"
                "- If |Actual - Design| > 15% sustained, treat as risk.\n"
                "- If repeated spikes > 20% for >2 hours -> escalate."
            )
            alert_ctx = (
                "- Immediate escalation: |Actual − Design| > 20% for 2+ hours.\n"
                "- Model alert: |Actual − Predicted| > 10% for 3+ hours."
            )

            ans = call_llm_agent("PAGE 1 — Plant + Efficiency/Power", operator_q, num_ctx, kpi_ctx, alert_ctx)
            st.session_state["p1_last_answer"] = ans
            st.session_state["p1_last_question"] = operator_q
            st.rerun()

        if "p1_last_answer" in st.session_state:
            st.markdown("### AI Answer")
            st.markdown(st.session_state["p1_last_answer"])

    with colB:
        st.markdown("### 🔊 Voice Output")
        st.caption("Offline TTS uses pyttsx3 (if available).")
        if not _PYTTSX3_OK:
            st.warning("pyttsx3 not available. Install: `pip install pyttsx3` (voice output will then work).")
        else:
            if "p1_last_answer" in st.session_state:
                if st.button("Generate Voice for Last Answer", key="p1_tts_btn"):
                    wav = tts_offline_to_wav_bytes(st.session_state["p1_last_answer"])
                    if wav:
                        st.audio(wav, format="audio/wav")
                    else:
                        st.error("TTS failed on this machine (TTS backend missing).")


# =========================================================
# PAGE 2: ANOMALY (filtered by ON)
# =========================================================
def page_anomaly(anomaly_df, on_chillers, weeks=1):
    if anomaly_df is None or anomaly_df.empty:
        st.warning("No anomaly data available.")
        return

    if len(on_chillers) == 0:
        st.warning("All chillers are OFF. Turn ON chillers on Page 1 to view anomaly analysis.")
        return

    # filter to ON chillers
    df = anomaly_df.copy()
    df = df[df["Chiller"].isin(on_chillers)].copy()

    if df.empty:
        st.warning("No anomaly rows for selected ON chillers.")
        return

    st.markdown(
        """
        ### Anomaly Threshold Legend
        - **Low ΔT**: `CHW_IN − CHW_OUT < 3°C`
        - **High kW/TR**: `chiller_efficiency > 1.10`
        - **Compressor behaviour**: imbalance / unstable loads
        """
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    last_ts = df["timestamp"].max()
    start_ts = last_ts - pd.Timedelta(days=7 * weeks) if pd.notna(last_ts) else None
    df_recent = df[df["timestamp"] >= start_ts].copy() if start_ts is not None else df.copy()

    df_recent = detect_anomaly_factors(df_recent)

    st.markdown(f"## Anomaly Detection Summary (Last {weeks} week(s)) — ON chillers only")
    total_records = df_recent[["timestamp", "Chiller"]].drop_duplicates().shape[0] if "Chiller" in df_recent.columns else len(df_recent)
    total_anom = int(df_recent["is_anomaly"].sum()) if "is_anomaly" in df_recent.columns else 0
    st.write(f"Total Records Analyzed: {total_records}")
    st.write(f"Total Anomalies Detected: {total_anom}")

    st.markdown("### Anomaly Factors per Chiller")
    g1 = df_recent.groupby("Chiller", dropna=False)[["eff_factor", "comp_factor", "temp_factor"]].sum().reset_index()
    g1 = g1.dropna(subset=["Chiller"]).sort_values(["temp_factor", "comp_factor", "eff_factor"], ascending=False)

    g1_melt = g1.melt(id_vars="Chiller", value_vars=["eff_factor", "comp_factor", "temp_factor"], var_name="Factor", value_name="Count")
    factor_map = {"eff_factor": "Efficiency", "comp_factor": "Compressor behaviour", "temp_factor": "Temperature balance"}
    g1_melt["Factor"] = g1_melt["Factor"].map(factor_map).fillna(g1_melt["Factor"])

    fig1 = px.bar(g1_melt, x="Chiller", y="Count", color="Factor", barmode="stack", template="plotly_dark",
                  title="Anomaly Factors per Chiller (ON chillers)")
    fig1.update_layout(height=320, legend=dict(orientation="v", x=1.02, y=1.0))
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### Good vs Bad Zone Anomaly Score Distribution")
    score = pd.to_numeric(df_recent.get("anomaly_score", np.nan), errors="coerce")
    good_mask = score.between(-0.5, 0.5, inclusive="both")
    bad_mask = ~good_mask

    g2_good = df_recent.loc[good_mask, ["eff_factor", "comp_factor", "temp_factor"]].sum()
    g2_bad = df_recent.loc[bad_mask, ["eff_factor", "comp_factor", "temp_factor"]].sum()

    g2 = pd.DataFrame([
        {"Zone": "Good Zone (-0.5 to +0.5)", "Efficiency": g2_good.get("eff_factor", 0), "Compressor behaviour": g2_good.get("comp_factor", 0), "Temperature balance": g2_good.get("temp_factor", 0)},
        {"Zone": "Bad Zone (outside)", "Efficiency": g2_bad.get("eff_factor", 0), "Compressor behaviour": g2_bad.get("comp_factor", 0), "Temperature balance": g2_bad.get("temp_factor", 0)},
    ])
    g2_m = g2.melt(id_vars="Zone", value_vars=["Efficiency", "Compressor behaviour", "Temperature balance"], var_name="Factor", value_name="Count")

    fig2 = px.bar(g2_m, x="Zone", y="Count", color="Factor", barmode="stack", template="plotly_dark",
                  title="Good vs Bad Zone distribution (ON chillers)")
    fig2.update_layout(height=300, legend=dict(orientation="v", x=1.02, y=1.0))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Issue Diagnosis Summary")
    issues = diagnose_issues_from_anomaly(df_recent)
    display_issue_cards(issues)

    st.markdown("## AI Q&A — Anomaly (ON chillers)")
    focus_ch = st.selectbox("Select ON chiller:", options=sorted(df_recent["Chiller"].dropna().unique().tolist()), key="ai_anom_ch")

    preset = st.selectbox(
        "Quick question:",
        options=[
            "What is triggering anomalies and what actions should operations take?",
            "Explain factor mix and recommended actions.",
            "What alarm thresholds should we set for anomaly levels?",
        ],
        key="ai_anom_preset"
    )
    freeq = st.text_area("Or type your own anomaly question:", value="", key="ai_anom_free")
    final = freeq.strip() or preset

    if st.button("Ask AI about anomalies", type="primary", key="ai_anom_btn"):
        dch = df_recent[df_recent["Chiller"] == focus_ch].copy()
        num_ctx = {
            "selected_on_chillers": on_chillers,
            "chiller": focus_ch,
            "weeks_window": weeks,
            "records": int(len(dch)),
            "anomaly_score_stats": {
                "min": float(pd.to_numeric(dch.get("anomaly_score", np.nan), errors="coerce").min()) if len(dch) else None,
                "max": float(pd.to_numeric(dch.get("anomaly_score", np.nan), errors="coerce").max()) if len(dch) else None,
                "mean": float(pd.to_numeric(dch.get("anomaly_score", np.nan), errors="coerce").mean()) if len(dch) else None,
            },
            "factor_counts": {
                "efficiency": float(dch["eff_factor"].sum()) if "eff_factor" in dch.columns else 0,
                "compressor": float(dch["comp_factor"].sum()) if "comp_factor" in dch.columns else 0,
                "temp_balance": float(dch["temp_factor"].sum()) if "temp_factor" in dch.columns else 0,
            }
        }

        kpi_ctx = "- Anomaly score near 0 is normal. Large magnitude indicates abnormal behaviour."
        alert_ctx = "- Medium alert: repeated anomalies for >3 hours. High alert: severe anomalies sustained."

        ans = call_llm_agent("PAGE 2 — Anomaly", final, num_ctx, kpi_ctx, alert_ctx)
        st.markdown("### AI Explanation")
        st.markdown(ans)

        # Voice output for page 2 answer
        if _PYTTSX3_OK:
            wav = tts_offline_to_wav_bytes(ans)
            if wav:
                st.audio(wav, format="audio/wav")


# =========================================================
# PAGE 3: MAINTENANCE (filtered by ON)
# =========================================================
def page_maintenance(power_df, on_chillers, weeks=1):
    if power_df is None or power_df.empty:
        st.warning("No power data available to compute maintenance.")
        return

    if len(on_chillers) == 0:
        st.warning("All chillers are OFF. Turn ON chillers on Page 1 to view maintenance.")
        return

    df = power_df.copy()
    df = df[df["Chiller"].isin(on_chillers)].copy()

    if df.empty:
        st.warning("No power rows for selected ON chillers.")
        return

    st.markdown(
        """
        ## Maintenance Scoring Legend
        - **Average Efficiency (kW/TR)**: > 1.10 → +40, 0.90–1.10 → +20, 0.80–0.90 → +10  
        - **Efficiency Trend**: > +0.05 → +30, +0.02–0.05 → +15  
        - **Compressor Variance**: > 300 → +15  
        - **Compressor Imbalance**: > 15 → +10  
        """
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    last_ts = df["timestamp"].max()
    start_ts = last_ts - pd.Timedelta(days=7 * weeks) if pd.notna(last_ts) else None
    df_recent = df[df["timestamp"] >= start_ts].copy() if start_ts is not None else df.copy()

    maintenance_df = predict_maintenance_from_power(df_recent)

    render_predictive_maintenance_dashboard(maintenance_df)
    render_cmms_vs_model_table(maintenance_df)
    display_maintenance_cards(maintenance_df)

    st.markdown("## AI Q&A — Maintenance (ON chillers)")
    if maintenance_df.empty:
        st.info("No maintenance scores available yet for AI.")
        return

    preset = st.selectbox(
        "Quick question:",
        options=[
            "Which ON chillers require immediate maintenance and why?",
            "How to prioritise Corrective vs Preventive vs Routine actions?",
            "What are the main drivers of high maintenance scores in ON fleet?",
        ],
        key="ai_maint_preset",
    )
    freeq = st.text_area("Or type your own maintenance question:", value="", key="ai_maint_free")
    final = freeq.strip() or preset

    if st.button("Ask AI about maintenance", type="primary", key="ai_maint_btn"):
        fleet_health = max(0.0, 100.0 - float(maintenance_df["maintenance_score"].mean()))
        top5 = maintenance_df.sort_values("maintenance_score", ascending=False).head(5).to_dict(orient="records")

        num_ctx = {
            "selected_on_chillers": on_chillers,
            "weeks_window": weeks,
            "num_chillers_scored": int(len(maintenance_df)),
            "fleet_health_score_out_of_100": float(fleet_health),
            "top_chillers_by_score": top5,
        }

        kpi_ctx = "- High (>=60): Corrective / immediate. Medium (35–60): Preventive. Low (<35): Routine."
        alert_ctx = "- If CMMS date is behind model predicted date, risk is increasing."

        ans = call_llm_agent("PAGE 3 — Maintenance", final, num_ctx, kpi_ctx, alert_ctx)
        st.markdown("### AI Explanation")
        st.markdown(ans)

        if _PYTTSX3_OK:
            wav = tts_offline_to_wav_bytes(ans)
            if wav:
                st.audio(wav, format="audio/wav")


# =========================================================
# PAGE 4: CAPACITY PLANNING (uses ON chillers list)
# =========================================================
def page_capacity_planning(ml, chiller_map, on_chillers):
    st.subheader("Chiller Capacity Planning — Required Chillers (Based on ON chillers selection)")

    if len(on_chillers) == 0:
        st.warning("All chillers are OFF. Turn ON chillers on Page 1 to run capacity planning.")
        return

    colA, colB = st.columns(2)
    with colA:
        terrace_amb_temp = st.number_input("Terrace Ambient Temp (°C)", value=DEFAULT_TERRACE_AMB_TEMP, step=0.1, format="%.2f")
    with colB:
        total_it_load = st.number_input("Total IT Load (model units)", value=DEFAULT_TOTAL_IT_LOAD, step=1.0, format="%.1f")

    run = st.button("Compute required chillers (UTC now)", type="primary")
    if not run:
        return

    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    time_str = now_utc.strftime("%Y-%m-%d %H:%M:%S")

    # Only ON chillers sent to model
    h6_list = [chiller_map[c] for c in on_chillers if c in chiller_map]
    raw = ml.fetch_chillers_required(time_str, h6_list, terrace_amb_temp, total_it_load)
    out = pd.DataFrame(raw) if isinstance(raw, list) else pd.DataFrame()

    if out.empty:
        st.error("No response from the chiller requirement model.")
        return

    count_col = None
    for cand in ["chiller_required_count", "chiller_count_required", "required_chiller_count", "required_chillers"]:
        if cand in out.columns:
            count_col = cand
            break

    if count_col is None:
        st.error("Model response missing 'chiller_required_count'. Please verify endpoint schema.")
        st.dataframe(out, use_container_width=True)
        return

    counts = pd.to_numeric(out[count_col], errors="coerce").dropna()
    if counts.empty:
        st.error("Model returned no numeric required counts.")
        st.dataframe(out, use_container_width=True)
        return

    final_required = int(np.nanmax(counts.values))
    st.markdown(f"### Required Chillers (for ON group): **{final_required}**")

    # AI explanation + voice
    st.markdown("## AI Q&A — Capacity Planning")
    q = st.text_area("Ask why required chillers changed:", value="Explain how terrace ambient and IT load affected required chillers today.")
    if st.button("Ask AI about capacity planning", type="primary"):
        num_ctx = {
            "selected_on_chillers": on_chillers,
            "terrace_amb_temp": float(terrace_amb_temp),
            "total_it_load": float(total_it_load),
            "model_required_chillers": int(final_required),
        }
        kpi_ctx = "- Higher ambient increases cooling load; higher IT load increases required capacity."
        alert_ctx = "- If required chillers rises rapidly, verify IT load spike / ambient spike / sensor validity."

        ans = call_llm_agent("PAGE 4 — Capacity Planning", q, num_ctx, kpi_ctx, alert_ctx)
        st.markdown("### AI Explanation")
        st.markdown(ans)

        if _PYTTSX3_OK:
            wav = tts_offline_to_wav_bytes(ans)
            if wav:
                st.audio(wav, format="audio/wav")

    with st.expander("Show raw model response"):
        st.dataframe(out, use_container_width=True)


# =========================================================
# MAIN
# =========================================================
def main():
    set_custom_background_and_header()
    st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)

    # --- 30 chillers exactly ---
    chillers = build_30_chiller_map()
    chiller_names = list(chillers.keys())

    init_chiller_state(chiller_names)
    on_chillers = get_on_chillers(chiller_names)

    # --- Realtime snapshot (only ON chillers to reduce calls) ---
    realtime_wide = pd.DataFrame()
    if st.session_state.get("use_rt", False):
        # auto refresh (seconds)
        rt_sec = int(st.session_state.get("rt_refresh_sec", 20))
        # manual cache using session_state to avoid hammering API
        now = time.time()
        last_t = st.session_state.get("_rt_last_fetch_t", 0.0)
        if (now - last_t) >= max(5, rt_sec):
            try:
                on_nums = [int(str(c).split("-")[-1]) for c in on_chillers]
            except Exception:
                on_nums = []
            if on_nums:
                tidy = fetch_realtime_snapshot(on_nums, prefer_template=st.session_state.get("rt_template", "dash"))
                realtime_wide = realtime_pivot(tidy)
                st.session_state["_rt_last_fetch_t"] = now
                st.session_state["_rt_last_df"] = realtime_wide
                st.session_state["_rt_last_tidy"] = tidy
            else:
                realtime_wide = pd.DataFrame()
                st.session_state["_rt_last_df"] = realtime_wide
                st.session_state["_rt_last_tidy"] = pd.DataFrame()
        else:
            realtime_wide = st.session_state.get("_rt_last_df", pd.DataFrame())

    ml = ChillerMLModels()

    # auto refresh hourly
    st_autorefresh(interval=3600 * 1000, key="auto_refresh")

    # update incremental every run (for only those chillers)
    try:
        power_df, anomaly_df, design_power_df, ch_req_df = update_predictions_incremental(ml, chillers, list(chillers.values()))
    except Exception as e:
        st.warning(f"Update failed (endpoints unreachable). Using existing CSVs if present. Error: {e}")
        power_df = pd.read_csv(POWER_CSV, parse_dates=["timestamp"]) if os.path.exists(POWER_CSV) else pd.DataFrame()
        anomaly_df = pd.read_csv(ANOMALY_CSV, parse_dates=["timestamp"]) if os.path.exists(ANOMALY_CSV) else pd.DataFrame()
        design_power_df = pd.read_csv(DESIGN_POWER_CSV, parse_dates=["timestamp"]) if os.path.exists(DESIGN_POWER_CSV) else pd.DataFrame()
        ch_req_df = pd.read_csv(CHILLER_REQ_CSV, parse_dates=["timestamp"]) if os.path.exists(CHILLER_REQ_CSV) else pd.DataFrame()

    # normalize + plotting timestamps
    power_df = ensure_power_csv(power_df)
    design_power_df = compute_design_scaled_errors(ensure_design_power_csv(design_power_df))
    anomaly_df = _cast_h6_int64(_drop_error_rows(anomaly_df))
    ch_req_df = ensure_chiller_req_csv(ch_req_df)

    # Make sure "Chiller" column uses CH-xx naming (important!)
    # If your endpoint returns other naming, we align using hierarchy6
    if not power_df.empty and "hierarchy6" in power_df.columns:
        power_df["Chiller"] = power_df["hierarchy6"].apply(lambda x: f"CH-{int(x):02d}" if pd.notna(x) else np.nan)
    if not anomaly_df.empty and "hierarchy6" in anomaly_df.columns:
        anomaly_df["Chiller"] = anomaly_df["hierarchy6"].apply(lambda x: f"CH-{int(x):02d}" if pd.notna(x) else np.nan)
    if not design_power_df.empty and "hierarchy6" in design_power_df.columns:
        design_power_df["Chiller"] = design_power_df["hierarchy6"].apply(lambda x: f"CH-{int(x):02d}" if pd.notna(x) else np.nan)

    if not power_df.empty:
        power_df["timestamp"] = _to_plot_time(power_df["timestamp"])
        power_df["chiller_power_consumption"] = pd.to_numeric(power_df["chiller_power_consumption"], errors="coerce")
        power_df["actual_chiller_power_consumption"] = pd.to_numeric(power_df["actual_chiller_power_consumption"], errors="coerce")
        power_df = power_df.sort_values(["Chiller", "timestamp"])

    if not design_power_df.empty:
        design_power_df["timestamp"] = _to_plot_time(design_power_df["timestamp"])
        design_power_df = design_power_df.sort_values(["Chiller", "timestamp"])

    if not anomaly_df.empty:
        anomaly_df["timestamp"] = pd.to_datetime(anomaly_df["timestamp"], errors="coerce")

    nav_col, content_col = st.columns([0.8, 5.2], gap="small")
    with nav_col:
        st.markdown('<div class="left-nav">', unsafe_allow_html=True)
        st.markdown('<div class="left-nav-title">Dashboard</div>', unsafe_allow_html=True)
        nav_mode = st.radio(
            "Navigation",
            ["Page 1 — Plant + Efficiency", "Page 2 — Anomaly", "Page 3 — Maintenance", "Page 4 — Capacity Planning"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown('<div class="left-nav-title" style="margin-top:0.75rem;">Realtime</div>', unsafe_allow_html=True)
        use_rt = st.checkbox("Use Realtime API", value=False, key="use_rt")
        rt_template = st.selectbox("PointName template", options=["dash", "underscore"], index=0, key="rt_template")
        rt_refresh_sec = st.number_input("Refresh (seconds)", min_value=5, max_value=300, value=20, step=5, key="rt_refresh_sec")
        if use_rt:
            auth_ok = _get_realtime_auth() is not None
            st.caption("Source: ODataConnector RealtimeData (Basic Auth).")
            if not auth_ok:
                st.warning("Realtime credentials missing. Add REALTIME_USERNAME/REALTIME_PASSWORD in Streamlit secrets.")
        st.markdown("</div>", unsafe_allow_html=True)

    with content_col:
        if st.session_state.get("use_rt", False):
            with st.expander("Realtime Snapshot (ON chillers)", expanded=False):
                tidy = st.session_state.get("_rt_last_tidy", pd.DataFrame())
                if tidy is None or tidy.empty:
                    st.info("No realtime data fetched yet (turn ON at least one chiller on Page 1).")
                else:
                    st.dataframe(tidy.sort_values(["chiller_no","tag"]), use_container_width=True)

        # Show current ON set
        st.markdown(f"**ON Chillers:** {', '.join(on_chillers) if on_chillers else 'None'}")

        if nav_mode == "Page 1 — Plant + Efficiency":
            page_1_plant_and_efficiency(power_df, design_power_df, anomaly_df, chiller_names, on_chillers, realtime_wide=realtime_wide)

        elif nav_mode == "Page 2 — Anomaly":
            weeks = st.selectbox("Duration (in weeks)", options=[1, 2, 3, 4], index=0, key="anom_weeks")
            page_anomaly(anomaly_df, on_chillers, weeks=weeks)

        elif nav_mode == "Page 3 — Maintenance":
            weeks = st.selectbox("Duration (in weeks)", options=[1, 2, 3, 4], index=0, key="maint_weeks")
            page_maintenance(power_df, on_chillers, weeks=weeks)

        else:
            page_capacity_planning(ml, chillers, on_chillers)


if __name__ == "__main__":
    main()
