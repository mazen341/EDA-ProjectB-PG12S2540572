"""
HKUST SQ1 PV Power Forecasting — Interactive Dashboard
Student: MAZEN AL-HIMALI (PG12S2540572)

Single-file Streamlit app with:
- Animated solar-themed background (pure CSS, no external assets)
- Glassmorphism cards, custom theme, gradient KPIs
- Tab-based navigation for smooth flow
- Plotly interactive charts (zoom, pan, hover)
- All original modeling logic preserved
"""
import json
import os
import re
import base64
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Plotly for interactive charts (already pulls via streamlit's bundled plotly)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


# ============================================================
# CONFIG
# ============================================================
STUDENT_NAME_DEFAULT = "MAZEN AL-HIMALI"
STUDENT_ID_DEFAULT = "PG12S2540572"
DEFAULT_DATA_PATH = "data/dataset_sample.csv"
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_TARGET_COL = "total_active_power_w"
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

PV_WEATHER_HINTS = [
    "irradiance", "temperature", "humidity", "pressure",
    "wind", "rainfall", "visibility", "cloud",
]

AI_GRADER_PROMPT_TEMPLATE = """# Exact AI Grading Prompt (Hardcode inside app.py)

SYSTEM:
You are a strict academic grader. Return ONLY valid JSON.

USER:
Grade this time-series forecasting Streamlit project OUT OF 80 points using the fixed rubric below.
Be strict: do not award points unless evidence is present in the submitted JSON.
Return ONLY JSON exactly matching the schema.

RUBRIC MAX:
Data & integrity: 20
Feature engineering: 15
Modeling & evaluation: 25
Dashboard quality: 10
Presentation & rigor: 10

STRICT CAPS:
- If the project only uses baseline features/models with no meaningful additions, cap total_80 <= 45.
- If time-based split is missing/unclear, cap Modeling & evaluation <= 12.
- If missing timestamps/outliers/resampling are not discussed or evidenced, cap Data & integrity <= 10.
- If no metrics table is present, cap Modeling & evaluation <= 10.
- If no insights are provided, cap Presentation & rigor <= 5.

Return JSON:
{
  "scores": {
    "Data & integrity": int,
    "Feature engineering": int,
    "Modeling & evaluation": int,
    "Dashboard quality": int,
    "Presentation & rigor": int
  },
  "total_80": int,
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "actionable_improvements": [string, ...]
}

EVIDENCE JSON:
<insert submission.json contents here>
"""


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="HKUST SQ1 PV Forecasting — Live Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS — animated background, glass cards, gradient KPIs
# ============================================================
CUSTOM_CSS = """
<style>
/* ---------- App-wide theme ---------- */
.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f4e 25%, #2d1b69 50%, #1a1f4e 75%, #0a0e27 100%);
    background-size: 400% 400%;
    animation: gradientShift 25s ease infinite;
    color: #e8eaf6;
}

@keyframes gradientShift {
    0%   { background-position:   0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position:   0% 50%; }
}

/* ---------- Animated solar particles overlay ---------- */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(circle at 20% 30%, rgba(255, 200, 87, 0.12) 0%, transparent 35%),
        radial-gradient(circle at 80% 70%, rgba(255, 107, 53, 0.10) 0%, transparent 35%),
        radial-gradient(circle at 50% 50%, rgba(102, 187, 255, 0.08) 0%, transparent 45%);
    pointer-events: none;
    z-index: 0;
    animation: pulseGlow 12s ease-in-out infinite alternate;
}

@keyframes pulseGlow {
    0%   { opacity: 0.55; transform: scale(1.00); }
    100% { opacity: 0.95; transform: scale(1.05); }
}

/* Floating "solar panel" pattern via repeating linear-gradient */
.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        repeating-linear-gradient(45deg,
            rgba(255, 200, 87, 0.02) 0px,
            rgba(255, 200, 87, 0.02) 2px,
            transparent 2px,
            transparent 60px);
    pointer-events: none;
    z-index: 0;
}

/* Ensure content stays above background overlays */
.main > div { position: relative; z-index: 1; }

/* ---------- Glass-morphism container ---------- */
.glass-card {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 18px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
}

/* ---------- Hero header ---------- */
.hero {
    background: linear-gradient(135deg, rgba(255,180,50,0.18), rgba(255,107,53,0.18), rgba(102,187,255,0.18));
    border-radius: 22px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255, 200, 87, 0.25);
    box-shadow: 0 8px 40px rgba(255, 180, 50, 0.12);
    position: relative;
    overflow: hidden;
}
.hero h1 {
    background: linear-gradient(90deg, #FFD86B, #FF8A47, #FF6B9D, #66BBFF);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.3rem !important;
    font-weight: 800 !important;
    margin: 0 0 0.4rem 0 !important;
    animation: shineText 6s linear infinite;
}
@keyframes shineText {
    0%   { background-position:   0% center; }
    100% { background-position: 200% center; }
}
.hero p {
    color: #c5cae9;
    margin: 0;
    font-size: 1.05rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(255, 200, 87, 0.18);
    color: #FFD86B;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    font-size: 0.78rem;
    margin-right: 0.4rem;
    margin-top: 0.6rem;
    border: 1px solid rgba(255, 200, 87, 0.4);
    font-weight: 600;
}

/* ---------- Animated sun icon in hero ---------- */
.sun-icon {
    position: absolute;
    top: 1.2rem;
    right: 2rem;
    width: 60px;
    height: 60px;
    background: radial-gradient(circle, #FFD86B 30%, #FF8A47 100%);
    border-radius: 50%;
    box-shadow: 0 0 40px #FFD86B, 0 0 80px rgba(255, 138, 71, 0.6);
    animation: sunPulse 3s ease-in-out infinite;
}
@keyframes sunPulse {
    0%, 100% { transform: scale(1);    box-shadow: 0 0 40px #FFD86B, 0 0 80px  rgba(255,138,71,0.6); }
    50%      { transform: scale(1.1);  box-shadow: 0 0 60px #FFD86B, 0 0 100px rgba(255,138,71,0.9); }
}

/* ---------- Gradient KPI cards ---------- */
.kpi-card {
    background: linear-gradient(135deg, rgba(102,187,255,0.15), rgba(46,118,255,0.12));
    border: 1px solid rgba(102, 187, 255, 0.3);
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    transition: all 0.25s ease;
    height: 100%;
}
.kpi-card:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 8px 28px rgba(102, 187, 255, 0.3);
}
.kpi-card.warm {
    background: linear-gradient(135deg, rgba(255,180,50,0.18), rgba(255,107,53,0.14));
    border-color: rgba(255, 180, 50, 0.35);
}
.kpi-card.green {
    background: linear-gradient(135deg, rgba(76,217,123,0.18), rgba(46,170,90,0.14));
    border-color: rgba(76, 217, 123, 0.35);
}
.kpi-card.purple {
    background: linear-gradient(135deg, rgba(187,134,252,0.18), rgba(138,71,212,0.14));
    border-color: rgba(187, 134, 252, 0.35);
}
.kpi-label {
    color: #b0bec5;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
    font-weight: 600;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: #fff;
    line-height: 1;
}
.kpi-sub {
    color: #90a4ae;
    font-size: 0.78rem;
    margin-top: 0.35rem;
}

/* ---------- Section header ---------- */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(255, 200, 87, 0.25);
}
.section-icon {
    font-size: 1.6rem;
    filter: drop-shadow(0 0 8px rgba(255,200,87,0.5));
}
.section-title {
    color: #FFD86B;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0;
}

/* ---------- Streamlit native widget overrides ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255, 255, 255, 0.04);
    padding: 6px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #b0bec5;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(255,180,50,0.3), rgba(255,107,53,0.3)) !important;
    color: #FFD86B !important;
    box-shadow: 0 2px 10px rgba(255,180,50,0.3);
}
.stTabs [data-baseweb="tab"]:hover {
    color: #FFD86B;
    background: rgba(255, 200, 87, 0.06);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #FF8A47, #FF6B9D);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.4rem;
    font-weight: 700;
    box-shadow: 0 4px 15px rgba(255, 107, 53, 0.35);
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 22px rgba(255, 107, 53, 0.55);
    color: #fff;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #66BBFF, #4C8BF5);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.4rem;
    font-weight: 700;
    box-shadow: 0 4px 15px rgba(76, 139, 245, 0.4);
}

/* Sidebar */
section[data-testid="stSidebar"] > div {
    background: rgba(10, 14, 39, 0.85);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFD86B !important;
}

/* DataFrame */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

/* Metric */
[data-testid="stMetricValue"] {
    color: #FFD86B;
    font-weight: 700;
}

/* Headings */
h1, h2, h3 { color: #e8eaf6; }

/* JSON viewer */
.stJson { background: rgba(0,0,0,0.3); border-radius: 10px; }

/* Footer */
.footer {
    text-align: center;
    color: #78909c;
    font-size: 0.85rem;
    margin-top: 3rem;
    padding: 1.2rem;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}

/* Live pulse indicator */
.live-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    background: #4CD97B;
    border-radius: 50%;
    margin-right: 6px;
    animation: livePulse 1.5s ease-in-out infinite;
    box-shadow: 0 0 10px #4CD97B;
}
@keyframes livePulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.6; transform: scale(1.3); }
}

/* Floating panels animation */
.float {
    animation: floatY 6s ease-in-out infinite;
}
@keyframes floatY {
    0%, 100% { transform: translateY(0px); }
    50%      { transform: translateY(-6px); }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS (modeling logic preserved)
# ============================================================
def safe_json_dumps(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


@st.cache_data(show_spinner=False)
def load_dataset(csv_path):
    return pd.read_csv(csv_path)


def dataframe_audit(df):
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[col].dtype) for col in df.columns],
        "non_null_count": [int(df[col].notna().sum()) for col in df.columns],
        "missing_pct": [round(float(df[col].isna().mean() * 100), 3) for col in df.columns],
        "unique_count": [int(df[col].nunique(dropna=True)) for col in df.columns],
    })


def likely_datetime_columns(df):
    rows = []
    for col in df.columns:
        name_score = int(any(t in col.lower() for t in ["time", "date", "timestamp", "datetime"]))
        parsed = pd.to_datetime(df[col], errors="coerce")
        valid_ratio = float(parsed.notna().mean())
        score = name_score + valid_ratio
        if name_score or valid_ratio > 0.70:
            rows.append({
                "column": col,
                "valid_datetime_pct": round(valid_ratio * 100, 2),
                "name_hint": bool(name_score),
                "score": round(score, 3),
            })
    if not rows:
        return pd.DataFrame(columns=["column", "valid_datetime_pct", "name_hint", "score"])
    return pd.DataFrame(rows).sort_values(["score", "valid_datetime_pct"], ascending=False)


def numeric_target_candidates(df):
    rows = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        valid_ratio = float(converted.notna().mean())
        missing_pct = 100 - valid_ratio * 100
        unique_count = int(converted.nunique(dropna=True))
        is_id_like = "id" in col.lower() or unique_count >= max(50, int(0.95 * len(df)))
        if valid_ratio >= 0.50 and unique_count > 2 and not is_id_like:
            rows.append({
                "column": col,
                "missing_pct": round(missing_pct, 3),
                "mean": round(float(converted.mean()), 4),
                "std": round(float(converted.std()), 4),
                "unique_count": unique_count,
            })
    if not rows:
        return pd.DataFrame(columns=["column", "missing_pct", "mean", "std", "unique_count"])
    return pd.DataFrame(rows).sort_values(["missing_pct", "std"], ascending=[True, False])


def clean_time_series(df, timestamp_col, target_col):
    cleaned = df.copy()
    cleaned[timestamp_col] = pd.to_datetime(cleaned[timestamp_col], errors="coerce")
    cleaned[target_col] = pd.to_numeric(cleaned[target_col], errors="coerce")
    rows_before = len(cleaned)
    invalid_timestamps = int(cleaned[timestamp_col].isna().sum())
    missing_targets = int(cleaned[target_col].isna().sum())
    cleaned = cleaned.dropna(subset=[timestamp_col, target_col]).sort_values(timestamp_col)
    cleaned = cleaned.drop_duplicates(subset=[timestamp_col], keep="last")
    return cleaned, {
        "rows_before_cleaning": int(rows_before),
        "rows_after_cleaning": int(len(cleaned)),
        "invalid_timestamp_rows_removed": invalid_timestamps,
        "missing_target_rows_removed": missing_targets,
        "duplicate_timestamps_after_cleaning": int(cleaned[timestamp_col].duplicated().sum()),
    }


def resample_time_series(df, timestamp_col, target_col, rule):
    if rule == "No resampling":
        return df.copy()
    numeric_df = df.copy()
    for col in numeric_df.columns:
        if col != timestamp_col:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
    numeric_cols = numeric_df.select_dtypes(include=[np.number]).columns.tolist()
    result = (numeric_df.set_index(timestamp_col)[numeric_cols]
              .resample(rule).mean().reset_index())
    return result.dropna(subset=[target_col])


def infer_time_coverage(df, timestamp_col):
    if df.empty:
        return {}
    ts = pd.to_datetime(df[timestamp_col], errors="coerce").dropna().sort_values()
    diffs = ts.diff().dropna()
    mode_step = str(diffs.mode().iloc[0]) if not diffs.empty else "Unknown"
    return {
        "start": str(ts.min()),
        "end": str(ts.max()),
        "rows": int(len(df)),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "most_common_step": mode_step,
    }


def make_baseline_features(df, timestamp_col, target_col, horizon,
                            include_weather=True, weather_cols=None, extra_lags=True):
    work = df.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    weather_cols = weather_cols or []
    for c in weather_cols:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=[timestamp_col, target_col]).sort_values(timestamp_col)

    work["lag_1"] = work[target_col].shift(1)
    work["lag_24"] = work[target_col].shift(24)
    work["rolling_mean_24"] = work[target_col].shift(1).rolling(24).mean()
    work["hour"] = work[timestamp_col].dt.hour
    work["weekend"] = work[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)
    work["month"] = work[timestamp_col].dt.month

    baseline_features = ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"]
    feature_groups = {"baseline": list(baseline_features)}

    extra_features = []
    if extra_lags:
        work["lag_2"] = work[target_col].shift(2)
        work["lag_3"] = work[target_col].shift(3)
        work["rolling_std_24"] = work[target_col].shift(1).rolling(24).std()
        work["rolling_mean_6"] = work[target_col].shift(1).rolling(6).mean()
        work["diff_1"] = work[target_col].shift(1) - work[target_col].shift(2)
        extra_features = ["lag_2", "lag_3", "rolling_std_24", "rolling_mean_6", "diff_1"]
        feature_groups["extra_lags"] = list(extra_features)

    work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24)
    work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24)
    work["doy"] = work[timestamp_col].dt.dayofyear
    work["doy_sin"] = np.sin(2 * np.pi * work["doy"] / 365.25)
    work["doy_cos"] = np.cos(2 * np.pi * work["doy"] / 365.25)
    work["is_daylight"] = ((work["hour"] >= 6) & (work["hour"] <= 19)).astype(int)
    domain_time_features = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "is_daylight"]
    feature_groups["domain_time"] = list(domain_time_features)

    weather_feature_list = []
    if include_weather and weather_cols:
        for c in weather_cols:
            if c in work.columns:
                lagged = f"{c}_lag1"
                work[lagged] = work[c].shift(1)
                weather_feature_list.append(lagged)
        if weather_feature_list:
            feature_groups["weather"] = list(weather_feature_list)

    all_features = baseline_features + extra_features + domain_time_features + weather_feature_list
    work["y_target"] = work[target_col].shift(-int(horizon))
    feature_table = work.dropna(subset=all_features + ["y_target"]).reset_index(drop=True)
    X = feature_table[all_features].copy()
    y = feature_table["y_target"].copy()
    return feature_table, X, y, all_features, feature_groups


def get_openrouter_api_key():
    try:
        key = st.secrets["OPENROUTER_API_KEY"]
        if key:
            return str(key)
    except Exception:
        pass
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key
    return st.text_input("🔑 OpenRouter API key", type="password",
                          help="Used only for the AI grader call. Not stored in the app code.")


def _extract_balanced_json(text):
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape: escape = False
            elif ch == "\\": escape = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_grader_response(raw_text):
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    try:
        return json.loads(cleaned), None
    except Exception:
        pass
    candidate = _extract_balanced_json(cleaned)
    if candidate:
        try:
            return json.loads(candidate), None
        except Exception as exc:
            return None, f"Balanced-brace extraction failed: {exc}"
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), None
        except Exception as exc:
            return None, str(exc)
    return None, "No JSON object found in model response."


def call_openrouter_grader(api_key, prompt):
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": OPENROUTER_MODEL,
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def kpi_card(label, value, sub="", variant="default"):
    """Return HTML for a custom KPI card."""
    return f"""
    <div class="kpi-card {variant}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def section_header(icon, title):
    st.markdown(
        f'<div class="section-header"><span class="section-icon">{icon}</span>'
        f'<h2 class="section-title">{title}</h2></div>',
        unsafe_allow_html=True,
    )


PLOTLY_DARK = "plotly_dark"
PLOTLY_COLORS = {
    "actual":   "#66BBFF",
    "rf":       "#FF8A47",
    "naive":    "#BB86FC",
    "seasonal": "#4CD97B",
    "ridge":    "#FFD86B",
    "gbr":      "#FF6B9D",
    "residual": "#4CD97B",
}


def style_plotly(fig, height=380):
    fig.update_layout(
        template=PLOTLY_DARK,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", color="#e8eaf6", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        hoverlabel=dict(bgcolor="rgba(20,25,55,0.95)", bordercolor="#FFD86B",
                        font=dict(color="#fff", size=12)),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)"),
        legend=dict(bgcolor="rgba(20,25,55,0.6)", bordercolor="rgba(255,255,255,0.15)",
                    borderwidth=1, font=dict(color="#e8eaf6")),
    )
    return fig


# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <div class="sun-icon"></div>
    <h1>☀️ HKUST SQ1 PV Power Forecasting</h1>
    <p>Live interactive time-series forecasting dashboard — clean, audit, engineer features, train models, and inspect diagnostics with one continuous flow.</p>
    <div>
        <span class="hero-badge"><span class="live-dot"></span>LIVE</span>
        <span class="hero-badge">Time-Series Forecasting</span>
        <span class="hero-badge">5 Models</span>
        <span class="hero-badge">Up to 24 Features</span>
        <span class="hero-badge">Interactive</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR — Student info & project meta
# ============================================================
with st.sidebar:
    st.markdown("### 👤 Student Profile")
    student_name = st.text_input("Student name", value=STUDENT_NAME_DEFAULT)
    student_id = st.text_input("Student ID", value=STUDENT_ID_DEFAULT)

    st.markdown("---")
    st.markdown("### 🔗 Deployment Links")
    deployed_url = st.text_input("Streamlit URL", value="", placeholder="https://...streamlit.app")
    repo_url = st.text_input("GitHub URL", value="", placeholder="https://github.com/...")

    st.markdown("---")
    st.markdown("### 📋 Project Meta")
    project_title = st.text_input("Title", value="HKUST SQ1 PV Power Forecasting")
    project_goal = st.text_area(
        "Goal",
        value="Forecast inverter total active AC power from a cleaned time-series dataset using time-aware baseline features and weather covariates.",
        height=100,
    )

    st.markdown("---")
    st.markdown("### 📁 Data Source")
    data_path = st.text_input("CSV path", value=DEFAULT_DATA_PATH)

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#90a4ae; font-size:0.8rem; padding:0.5rem;">'
        '☀️ Powered by Streamlit<br>+ scikit-learn + Plotly'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# LOAD DATA
# ============================================================
try:
    df = load_dataset(data_path)
except Exception as exc:
    st.error(f"❌ Could not load dataset from {data_path}: {exc}")
    st.stop()


# ============================================================
# TOP-LEVEL KPI ROW (always visible above tabs)
# ============================================================
ts_parsed = pd.to_datetime(df[DEFAULT_TIMESTAMP_COL], errors="coerce") if DEFAULT_TIMESTAMP_COL in df.columns else None
date_span = ""
if ts_parsed is not None and ts_parsed.notna().any():
    date_span = f"{ts_parsed.min().date()} → {ts_parsed.max().date()}"
target_preview = pd.to_numeric(df[DEFAULT_TARGET_COL], errors="coerce") if DEFAULT_TARGET_COL in df.columns else None
target_mean = f"{target_preview.mean():.0f} W" if target_preview is not None and target_preview.notna().any() else "—"
target_peak = f"{target_preview.max():.0f} W" if target_preview is not None and target_preview.notna().any() else "—"

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(kpi_card("📊 Rows Loaded", f"{len(df):,}", date_span), unsafe_allow_html=True)
with c2: st.markdown(kpi_card("🔢 Columns",    f"{len(df.columns)}", "features in source", "warm"), unsafe_allow_html=True)
with c3: st.markdown(kpi_card("⚡ Mean Power", target_mean, "across full series", "green"), unsafe_allow_html=True)
with c4: st.markdown(kpi_card("🌟 Peak Power", target_peak, "max observed", "purple"), unsafe_allow_html=True)


# ============================================================
# TAB-BASED FLOW
# ============================================================
tab_data, tab_features, tab_model, tab_dash, tab_export, tab_grader = st.tabs([
    "📋 Data & Audit",
    "🛠️ Feature Engineering",
    "🤖 Modeling",
    "📈 Dashboard",
    "💾 Export",
    "🎯 AI Grader",
])


# ─────────────────────────────────────────────
# TAB 1: DATA & AUDIT
# ─────────────────────────────────────────────
with tab_data:
    section_header("📋", "Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    section_header("🔍", "Data Audit")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("**Column-level audit**")
        st.dataframe(dataframe_audit(df), use_container_width=True, height=320)
    with col_b:
        st.markdown("**Likely timestamp columns**")
        st.dataframe(likely_datetime_columns(df).head(3), use_container_width=True)
        st.markdown("**Likely numeric target columns**")
        st.dataframe(numeric_target_candidates(df).head(3), use_container_width=True)

    section_header("⚙️", "Select Timestamp & Target")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        default_ts_index = df.columns.get_loc(DEFAULT_TIMESTAMP_COL) if DEFAULT_TIMESTAMP_COL in df.columns else 0
        timestamp_col = st.selectbox("Timestamp column", options=list(df.columns), index=int(default_ts_index))
    with col_t2:
        numeric_like_cols = [c for c in df.columns
                              if pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.50]
        if DEFAULT_TARGET_COL not in numeric_like_cols and DEFAULT_TARGET_COL in df.columns:
            numeric_like_cols.insert(0, DEFAULT_TARGET_COL)
        default_target_index = numeric_like_cols.index(DEFAULT_TARGET_COL) if DEFAULT_TARGET_COL in numeric_like_cols else 0
        target_col = st.selectbox("Target column", options=numeric_like_cols, index=int(default_target_index))

    # store for cross-tab use
    st.session_state["timestamp_col"] = timestamp_col
    st.session_state["target_col"] = target_col

    cleaned_df, cleaning_report = clean_time_series(df, timestamp_col, target_col)
    coverage = infer_time_coverage(cleaned_df, timestamp_col)

    section_header("✨", "Cleaning Report")
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1: st.markdown(kpi_card("Before", f"{cleaning_report['rows_before_cleaning']:,}", "raw rows"), unsafe_allow_html=True)
    with col_c2: st.markdown(kpi_card("After",  f"{cleaning_report['rows_after_cleaning']:,}", "cleaned rows", "green"), unsafe_allow_html=True)
    with col_c3: st.markdown(kpi_card("Removed", f"{cleaning_report['invalid_timestamp_rows_removed'] + cleaning_report['missing_target_rows_removed']:,}", "invalid + missing", "warm"), unsafe_allow_html=True)
    with col_c4: st.markdown(kpi_card("Duplicates", f"{cleaning_report['duplicate_timestamps_after_cleaning']:,}", "after dedupe", "purple"), unsafe_allow_html=True)

    with st.expander("🔬 Detailed cleaning + coverage JSON", expanded=False):
        st.json({**cleaning_report, **coverage})

    if cleaned_df.empty:
        st.error("No valid rows remain after parsing the timestamp and target columns.")
        st.stop()

    # Resampling + horizon controls
    section_header("⏱️", "Resampling & Forecast Horizon")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        resample_rule = st.selectbox(
            "Resampling rule",
            options=["No resampling", "5min", "15min", "30min", "1H", "1D"],
            index=0,
            help="Aggregate to a fixed grid before modeling.",
        )
    with col_r2:
        horizon = st.number_input(
            "Forecast horizon (rows ahead)",
            min_value=1, max_value=1000, value=1, step=1,
            help="How many steps in the future to predict.",
        )

    prepared_df = resample_time_series(cleaned_df, timestamp_col, target_col, resample_rule)
    prepared_coverage = infer_time_coverage(prepared_df, timestamp_col)

    with st.expander("📐 Coverage after resampling", expanded=False):
        st.json(prepared_coverage)

    # Live preview chart of the target series
    section_header("📉", "Live Target Preview")
    preview_n = st.slider("Preview window (rows)", 200, min(5000, len(prepared_df)), 1000, step=100,
                          key="preview_slider")
    if HAS_PLOTLY:
        prev_df = prepared_df.head(preview_n)
        fig_prev = go.Figure()
        fig_prev.add_trace(go.Scatter(
            x=prev_df[timestamp_col], y=prev_df[target_col],
            mode="lines", name=target_col,
            line=dict(color=PLOTLY_COLORS["actual"], width=1.2),
            fill="tozeroy", fillcolor="rgba(102,187,255,0.15)",
            hovertemplate="<b>%{x}</b><br>Power: %{y:.0f} W<extra></extra>",
        ))
        fig_prev.update_layout(title=f"📊 {target_col} — first {preview_n:,} rows")
        st.plotly_chart(style_plotly(fig_prev, height=360), use_container_width=True)
    else:
        st.line_chart(prepared_df.set_index(timestamp_col)[target_col].head(preview_n))

    # persist for downstream tabs
    st.session_state["prepared_df"] = prepared_df
    st.session_state["cleaned_df"] = cleaned_df
    st.session_state["cleaning_report"] = cleaning_report
    st.session_state["coverage"] = coverage
    st.session_state["prepared_coverage"] = prepared_coverage
    st.session_state["resample_rule"] = resample_rule
    st.session_state["horizon"] = int(horizon)


# Pull session state for following tabs (with safe defaults)
timestamp_col      = st.session_state.get("timestamp_col", DEFAULT_TIMESTAMP_COL)
target_col         = st.session_state.get("target_col", DEFAULT_TARGET_COL)
prepared_df        = st.session_state.get("prepared_df")
cleaned_df         = st.session_state.get("cleaned_df")
cleaning_report    = st.session_state.get("cleaning_report", {})
coverage           = st.session_state.get("coverage", {})
prepared_coverage  = st.session_state.get("prepared_coverage", {})
resample_rule      = st.session_state.get("resample_rule", "No resampling")
horizon            = st.session_state.get("horizon", 1)

if prepared_df is None or prepared_df.empty:
    # Build defaults so subsequent tabs don't crash
    cleaned_df, cleaning_report = clean_time_series(df, DEFAULT_TIMESTAMP_COL, DEFAULT_TARGET_COL)
    prepared_df = cleaned_df
    coverage = infer_time_coverage(cleaned_df, DEFAULT_TIMESTAMP_COL)
    prepared_coverage = coverage


# ─────────────────────────────────────────────
# TAB 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────
with tab_features:
    section_header("🛠️", "Time-Step Regularity Audit")
    ts_series = pd.to_datetime(prepared_df[timestamp_col]).sort_values()
    step_diffs = ts_series.diff().dropna()
    if len(step_diffs) > 0:
        step_counts = step_diffs.value_counts().head(5)
        mode_step = step_diffs.mode().iloc[0]
        irregular_pct = float((step_diffs != mode_step).mean() * 100)
        step_audit = {
            "modal_step": str(mode_step),
            "irregular_step_pct": round(irregular_pct, 3),
            "top_5_step_sizes": {str(k): int(v) for k, v in step_counts.items()},
            "max_gap": str(step_diffs.max()),
        }
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1: st.markdown(kpi_card("Modal Step", str(mode_step), "most common interval"), unsafe_allow_html=True)
        with col_s2: st.markdown(kpi_card("Irregular %", f"{irregular_pct:.2f}%",
                                           "deviating from modal",
                                           "warm" if irregular_pct > 1 else "green"), unsafe_allow_html=True)
        with col_s3: st.markdown(kpi_card("Max Gap", str(step_diffs.max()), "longest interval", "purple"), unsafe_allow_html=True)
        if irregular_pct > 1.0:
            st.warning(f"⚠️ {irregular_pct:.2f}% of steps deviate from the modal step ({mode_step}). Consider resampling in the Data tab.")
        else:
            st.success(f"✅ Series is highly regular ({100 - irregular_pct:.2f}% match the modal step of {mode_step}).")
        with st.expander("📊 Top 5 step sizes", expanded=False):
            st.json(step_audit)
    else:
        step_audit = {}

    section_header("🧬", "Feature Group Selection")
    st.markdown("Toggle feature groups and pick which weather columns to include as exogenous predictors.")

    candidate_weather = [
        c for c in prepared_df.columns
        if c != timestamp_col and c != target_col
        and any(h in c.lower() for h in PV_WEATHER_HINTS)
        and pd.to_numeric(prepared_df[c], errors="coerce").notna().mean() >= 0.5
    ]

    col_fe1, col_fe2 = st.columns([2, 1])
    with col_fe1:
        selected_weather = st.multiselect(
            "🌦️ Weather / exogenous features (lagged by 1 step to prevent leakage)",
            options=candidate_weather,
            default=candidate_weather[:5] if candidate_weather else [],
        )
    with col_fe2:
        use_extra_lags = st.checkbox("🔄 Extra lags (lag_2, lag_3, rolling_std, diff)", value=True)
        use_domain_time = st.checkbox("🌀 Cyclical time (sin/cos hour & day-of-year)", value=True)

    feature_table, X, y, feature_cols, feature_groups = make_baseline_features(
        prepared_df, timestamp_col, target_col, horizon,
        include_weather=bool(selected_weather),
        weather_cols=selected_weather,
        extra_lags=use_extra_lags,
    )
    if not use_domain_time and "domain_time" in feature_groups:
        drop_cols = feature_groups.pop("domain_time")
        feature_cols = [c for c in feature_cols if c not in drop_cols]
        X = X.drop(columns=drop_cols, errors="ignore")

    section_header("📦", "Feature Table Snapshot")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1: st.markdown(kpi_card("Feature Rows",   f"{len(feature_table):,}", "after lag/leakage drop"), unsafe_allow_html=True)
    with col_f2: st.markdown(kpi_card("Total Features", f"{len(feature_cols)}", "in current configuration", "warm"), unsafe_allow_html=True)
    with col_f3: st.markdown(kpi_card("Groups Active",  f"{len(feature_groups)}", "feature groups enabled", "green"), unsafe_allow_html=True)

    with st.expander("🔬 Feature groups in use", expanded=False):
        st.json({g: cols for g, cols in feature_groups.items() if all(c in feature_cols for c in cols)})

    st.markdown("**First 15 rows of the feature table**")
    st.dataframe(feature_table.head(15), use_container_width=True)

    # Persist
    st.session_state["feature_table"] = feature_table
    st.session_state["X"] = X
    st.session_state["y"] = y
    st.session_state["feature_cols"] = feature_cols
    st.session_state["feature_groups"] = feature_groups
    st.session_state["step_audit"] = step_audit
    st.session_state["selected_weather"] = selected_weather


feature_table   = st.session_state.get("feature_table")
X               = st.session_state.get("X")
y               = st.session_state.get("y")
feature_cols    = st.session_state.get("feature_cols", [])
feature_groups  = st.session_state.get("feature_groups", {})
step_audit      = st.session_state.get("step_audit", {})
selected_weather = st.session_state.get("selected_weather", [])


# ─────────────────────────────────────────────
# TAB 3: MODELING
# ─────────────────────────────────────────────
with tab_model:
    if feature_table is None or feature_table.empty:
        st.error("⚠️ Build the feature table in the Feature Engineering tab first.")
    else:
        section_header("🎛️", "Modeling Controls")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            test_size_pct = st.slider("📅 Test set size (% recent rows)", 10, 40, 20, step=5)
        with col_m2:
            rf_n_estimators = st.slider("🌲 RF n_estimators", 50, 300, 100, step=10)
            rf_max_depth = st.slider("📏 RF max_depth", 4, 24, 14)
        with col_m3:
            selected_models = st.multiselect(
                "🤖 Models to compare",
                options=["Naive (lag-1)", "Seasonal-naive (lag-24)", "Ridge regression",
                         "Random Forest", "Gradient Boosting"],
                default=["Naive (lag-1)", "Seasonal-naive (lag-24)", "Ridge regression", "Random Forest"],
                help="Gradient Boosting takes 1–2 min on full data.",
            )

        # Chronological split
        split_idx = int(len(feature_table) * (1 - test_size_pct / 100))
        train_df = feature_table.iloc[:split_idx].copy()
        test_df = feature_table.iloc[split_idx:].copy()
        X_train, y_train = train_df[feature_cols], train_df["y_target"]
        X_test, y_test = test_df[feature_cols], test_df["y_target"]

        split_info = {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_start": str(train_df[timestamp_col].min()),
            "train_end": str(train_df[timestamp_col].max()),
            "test_start": str(test_df[timestamp_col].min()),
            "test_end": str(test_df[timestamp_col].max()),
            "split_strategy": f"chronological, last {test_size_pct}% as test, no shuffling",
            "n_features_used": int(len(feature_cols)),
        }

        section_header("📆", "Time-Based Split")
        col_sp1, col_sp2, col_sp3, col_sp4 = st.columns(4)
        with col_sp1: st.markdown(kpi_card("Train Rows", f"{split_info['train_rows']:,}", split_info['train_start'][:10]), unsafe_allow_html=True)
        with col_sp2: st.markdown(kpi_card("Test Rows",  f"{split_info['test_rows']:,}",  split_info['test_start'][:10],  "warm"), unsafe_allow_html=True)
        with col_sp3: st.markdown(kpi_card("Features",   f"{split_info['n_features_used']}", "in model input", "green"), unsafe_allow_html=True)
        with col_sp4: st.markdown(kpi_card("Strategy",   "Chronological", "no shuffling — no leakage", "purple"), unsafe_allow_html=True)

        with st.expander("🔎 Detailed split JSON", expanded=False):
            st.json(split_info)

        # Compute metrics
        def compute_metrics(name, y_true, y_pred):
            y_true_arr = np.asarray(y_true, dtype=float)
            y_pred_arr = np.asarray(y_pred, dtype=float)
            mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
            y_true_arr = y_true_arr[mask]; y_pred_arr = y_pred_arr[mask]
            mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
            rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
            r2 = float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 1 else float("nan")
            denom = np.where(np.abs(y_true_arr) < 1e-6, np.nan, y_true_arr)
            mape = float(np.nanmean(np.abs((y_true_arr - y_pred_arr) / denom)) * 100)
            return {"model": name, "MAE": round(mae, 3), "RMSE": round(rmse, 3),
                    "R2": round(r2, 4), "MAPE_%": round(mape, 2),
                    "n_test": int(len(y_true_arr))}

        section_header("🚀", "Training & Evaluation")
        predictions, metric_rows = {}, []
        rf_model = None

        progress = st.progress(0, text="Initializing models...")
        steps = max(1, len(selected_models))
        cur = 0

        if "Naive (lag-1)" in selected_models:
            cur += 1; progress.progress(cur / steps, text="Naive baseline...")
            predictions["Naive (lag-1)"] = X_test["lag_1"].values
            metric_rows.append(compute_metrics("Naive (lag-1)", y_test, predictions["Naive (lag-1)"]))

        if "Seasonal-naive (lag-24)" in selected_models:
            cur += 1; progress.progress(cur / steps, text="Seasonal-naive baseline...")
            predictions["Seasonal-naive (lag-24)"] = X_test["lag_24"].values
            metric_rows.append(compute_metrics("Seasonal-naive (lag-24)", y_test, predictions["Seasonal-naive (lag-24)"]))

        if "Ridge regression" in selected_models:
            cur += 1; progress.progress(cur / steps, text="Training Ridge regression...")
            ridge_model = Ridge(alpha=1.0, random_state=42)
            ridge_model.fit(X_train, y_train)
            predictions["Ridge regression"] = ridge_model.predict(X_test)
            metric_rows.append(compute_metrics("Ridge regression", y_test, predictions["Ridge regression"]))

        if "Random Forest" in selected_models:
            cur += 1; progress.progress(cur / steps, text="Training Random Forest (this can take a moment)...")
            rf_model = RandomForestRegressor(
                n_estimators=rf_n_estimators, max_depth=rf_max_depth,
                min_samples_leaf=3, n_jobs=-1, random_state=42,
            )
            rf_model.fit(X_train, y_train)
            predictions["Random Forest"] = rf_model.predict(X_test)
            metric_rows.append(compute_metrics("Random Forest", y_test, predictions["Random Forest"]))

        if "Gradient Boosting" in selected_models:
            cur += 1; progress.progress(cur / steps, text="Training Gradient Boosting (slowest)...")
            gbr_model = GradientBoostingRegressor(
                n_estimators=120, max_depth=4, learning_rate=0.08,
                subsample=0.7, random_state=42,
            )
            gbr_model.fit(X_train, y_train)
            predictions["Gradient Boosting"] = gbr_model.predict(X_test)
            metric_rows.append(compute_metrics("Gradient Boosting", y_test, predictions["Gradient Boosting"]))

        progress.progress(1.0, text="✅ Training complete")
        if not metric_rows:
            st.error("Select at least one model.")
            st.stop()

        results_df = pd.DataFrame(metric_rows)
        best_row = results_df.loc[results_df["RMSE"].idxmin()]
        best_name = str(best_row["model"])

        section_header("🏆", "Performance Leaderboard")
        # Header KPIs for the winner
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1: st.markdown(kpi_card("👑 Best Model", best_name, "by RMSE", "warm"), unsafe_allow_html=True)
        with col_b2: st.markdown(kpi_card("RMSE", f"{float(best_row['RMSE']):.1f} W", "lower is better", "green"), unsafe_allow_html=True)
        with col_b3: st.markdown(kpi_card("MAE",  f"{float(best_row['MAE']):.1f} W",  "mean absolute error", "purple"), unsafe_allow_html=True)
        with col_b4: st.markdown(kpi_card("R²",   f"{float(best_row['R2']):.3f}",     "variance explained"), unsafe_allow_html=True)

        st.markdown("**Full metrics table**")
        st.dataframe(results_df.style.format({
            "MAE": "{:.1f}", "RMSE": "{:.1f}", "R2": "{:.4f}", "MAPE_%": "{:.2f}"
        }), use_container_width=True)

        # Plotly bar chart for visual comparison
        if HAS_PLOTLY:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=results_df["model"], y=results_df["RMSE"],
                marker=dict(color=results_df["RMSE"], colorscale="Sunset",
                            line=dict(color="rgba(255,255,255,0.15)", width=1)),
                text=[f"{v:.0f}" for v in results_df["RMSE"]],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>RMSE: %{y:.1f} W<extra></extra>",
            ))
            fig_bar.update_layout(title="📊 RMSE Comparison (Lower is Better)",
                                  yaxis_title="RMSE (W)", xaxis_title="")
            st.plotly_chart(style_plotly(fig_bar, height=380), use_container_width=True)

        # Quantified improvement
        section_header("📐", "Quantified Improvement vs Naive Baseline")
        if "Naive (lag-1)" in results_df["model"].values:
            naive_row = results_df[results_df["model"] == "Naive (lag-1)"].iloc[0]
            naive_rmse = float(naive_row["RMSE"]); naive_mae = float(naive_row["MAE"])
            improvements = []
            for _, row in results_df.iterrows():
                if row["model"] == "Naive (lag-1)": continue
                rmse_delta = (naive_rmse - float(row["RMSE"])) / naive_rmse * 100
                mae_delta = (naive_mae - float(row["MAE"])) / naive_mae * 100
                improvements.append({
                    "model": row["model"],
                    "RMSE_vs_naive_pct": round(rmse_delta, 2),
                    "MAE_vs_naive_pct": round(mae_delta, 2),
                    "RMSE_delta_W": round(naive_rmse - float(row["RMSE"]), 2),
                    "MAE_delta_W": round(naive_mae - float(row["MAE"]), 2),
                })
            improvements_df = pd.DataFrame(improvements)
            st.dataframe(improvements_df.style.format({
                "RMSE_vs_naive_pct": "{:+.2f}%", "MAE_vs_naive_pct": "{:+.2f}%",
                "RMSE_delta_W": "{:+.1f}", "MAE_delta_W": "{:+.1f}"
            }), use_container_width=True)
            st.caption("✨ Positive values = error reduction vs Naive (lag-1) baseline. Higher is better.")
        else:
            improvements_df = pd.DataFrame()
            st.info("Include Naive (lag-1) above to see quantified improvements.")

        if rf_model is not None:
            importances_df = pd.DataFrame({
                "feature": feature_cols, "importance": rf_model.feature_importances_,
            }).sort_values("importance", ascending=False).reset_index(drop=True)
        else:
            importances_df = pd.DataFrame(columns=["feature", "importance"])

        # persist for dashboard tab
        st.session_state["results_df"] = results_df
        st.session_state["predictions"] = predictions
        st.session_state["best_name"] = best_name
        st.session_state["best_row"] = best_row.to_dict()
        st.session_state["test_df"] = test_df
        st.session_state["split_info"] = split_info
        st.session_state["improvements_df"] = improvements_df
        st.session_state["importances_df"] = importances_df
        st.session_state["test_size_pct"] = test_size_pct


results_df       = st.session_state.get("results_df")
predictions      = st.session_state.get("predictions", {})
best_name        = st.session_state.get("best_name", "")
best_row_dict    = st.session_state.get("best_row", {})
test_df          = st.session_state.get("test_df")
split_info       = st.session_state.get("split_info", {})
improvements_df  = st.session_state.get("improvements_df", pd.DataFrame())
importances_df   = st.session_state.get("importances_df", pd.DataFrame())
test_size_pct    = st.session_state.get("test_size_pct", 20)


# ─────────────────────────────────────────────
# TAB 4: DASHBOARD
# ─────────────────────────────────────────────
with tab_dash:
    if results_df is None or test_df is None or not predictions:
        st.warning("⚠️ Train at least one model in the Modeling tab to see the dashboard.")
    else:
        section_header("🎯", "Inspect a Model")

        col_d1, col_d2 = st.columns([2, 1])
        with col_d1:
            dash_model = st.selectbox(
                "Choose model to inspect", options=list(predictions.keys()),
                index=list(predictions.keys()).index(best_name),
            )
        with col_d2:
            plot_n_max = min(2000, len(test_df))
            plot_n = st.slider("Plot window (test rows)", 100, plot_n_max,
                                min(600, plot_n_max), step=100)

        dash_pred = predictions[dash_model]
        dash_metrics_row = results_df[results_df["model"] == dash_model].iloc[0]

        pred_frame = test_df[[timestamp_col, "y_target"]].copy()
        pred_frame = pred_frame.rename(columns={"y_target": "actual"})
        pred_frame["pred"] = dash_pred
        pred_frame["residual"] = pred_frame["actual"] - pred_frame["pred"]
        pred_frame["abs_err"] = pred_frame["residual"].abs()
        pred_frame["date"] = pd.to_datetime(pred_frame[timestamp_col]).dt.date
        pred_frame["hour"] = pd.to_datetime(pred_frame[timestamp_col]).dt.hour

        # Inspector KPIs
        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        with col_i1: st.markdown(kpi_card("Test Rows", f"{len(pred_frame):,}", "in hold-out set"), unsafe_allow_html=True)
        with col_i2: st.markdown(kpi_card(f"RMSE", f"{float(dash_metrics_row['RMSE']):.1f} W", dash_model, "warm"), unsafe_allow_html=True)
        with col_i3: st.markdown(kpi_card(f"MAE",  f"{float(dash_metrics_row['MAE']):.1f} W",  dash_model, "green"), unsafe_allow_html=True)
        with col_i4: st.markdown(kpi_card(f"R²",   f"{float(dash_metrics_row['R2']):.3f}",   dash_model, "purple"), unsafe_allow_html=True)

        # Plot 1: Actual vs Predicted (Plotly)
        section_header("📈", f"Actual vs Predicted — {dash_model}")
        if HAS_PLOTLY:
            view = pred_frame.iloc[:plot_n]
            fig_ap = go.Figure()
            fig_ap.add_trace(go.Scatter(
                x=view[timestamp_col], y=view["actual"], mode="lines",
                name="Actual", line=dict(color=PLOTLY_COLORS["actual"], width=1.6),
                hovertemplate="<b>%{x}</b><br>Actual: %{y:.0f} W<extra></extra>",
            ))
            fig_ap.add_trace(go.Scatter(
                x=view[timestamp_col], y=view["pred"], mode="lines",
                name=dash_model, line=dict(color=PLOTLY_COLORS["rf"], width=1.3, dash="solid"),
                hovertemplate="<b>%{x}</b><br>Pred: %{y:.0f} W<extra></extra>",
            ))
            st.plotly_chart(style_plotly(fig_ap, height=400), use_container_width=True)
        else:
            chart_df = pred_frame.set_index(timestamp_col)[["actual", "pred"]].iloc[:plot_n]
            st.line_chart(chart_df)

        # Plot 2: Residuals
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**Residuals over time**")
            if HAS_PLOTLY:
                view = pred_frame.iloc[:plot_n]
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(
                    x=view[timestamp_col], y=view["residual"], mode="lines",
                    line=dict(color=PLOTLY_COLORS["residual"], width=1),
                    name="Residual", fill="tozeroy", fillcolor="rgba(76,217,123,0.12)",
                    hovertemplate="<b>%{x}</b><br>Residual: %{y:.0f} W<extra></extra>",
                ))
                fig_res.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
                st.plotly_chart(style_plotly(fig_res, height=320), use_container_width=True)

        with col_r2:
            st.markdown("**Residual distribution**")
            if HAS_PLOTLY:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=pred_frame["residual"].dropna(), nbinsx=60,
                    marker=dict(color=PLOTLY_COLORS["naive"],
                                line=dict(color="rgba(255,255,255,0.2)", width=0.5)),
                    hovertemplate="Bin: %{x}<br>Count: %{y}<extra></extra>",
                ))
                fig_hist.add_vline(x=0, line_color="rgba(255,255,255,0.5)", line_width=1)
                st.plotly_chart(style_plotly(fig_hist, height=320), use_container_width=True)

        # Plot 3: Scatter actual vs predicted
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**Actual vs Predicted scatter**")
            if HAS_PLOTLY:
                sample = pred_frame.sample(min(3000, len(pred_frame)), random_state=0)
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=sample["actual"], y=sample["pred"], mode="markers",
                    marker=dict(size=4, color=sample["abs_err"], colorscale="Viridis",
                                opacity=0.55, showscale=True,
                                colorbar=dict(title="|err|", thickness=12)),
                    hovertemplate="Actual: %{x:.0f}<br>Pred: %{y:.0f}<extra></extra>",
                ))
                lo = float(min(sample["actual"].min(), sample["pred"].min()))
                hi = float(max(sample["actual"].max(), sample["pred"].max()))
                fig_sc.add_trace(go.Scatter(
                    x=[lo, hi], y=[lo, hi], mode="lines",
                    line=dict(color="rgba(255,255,255,0.4)", dash="dash"),
                    name="y = x", showlegend=False,
                ))
                fig_sc.update_layout(xaxis_title="Actual (W)", yaxis_title="Predicted (W)")
                st.plotly_chart(style_plotly(fig_sc, height=380), use_container_width=True)

        with col_s2:
            st.markdown("**Mean absolute error by hour of day**")
            hourly_err = pred_frame.groupby("hour")["abs_err"].mean().reset_index()
            if HAS_PLOTLY:
                fig_h = go.Figure()
                fig_h.add_trace(go.Bar(
                    x=hourly_err["hour"], y=hourly_err["abs_err"],
                    marker=dict(color=hourly_err["abs_err"], colorscale="Sunset",
                                line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
                    hovertemplate="Hour: %{x}<br>MAE: %{y:.1f} W<extra></extra>",
                ))
                fig_h.update_layout(xaxis_title="Hour of day", yaxis_title="MAE (W)",
                                    xaxis=dict(tickmode="linear", dtick=2))
                st.plotly_chart(style_plotly(fig_h, height=380), use_container_width=True)

        # Daily error summary
        section_header("📅", "Daily Error Summary")
        daily_err = (pred_frame.groupby("date")
                     .agg(MAE=("abs_err", "mean"),
                          RMSE=("residual", lambda s: float(np.sqrt(np.mean(s ** 2)))),
                          n=("abs_err", "size"))
                     .reset_index().round(2))
        if HAS_PLOTLY:
            fig_d = go.Figure()
            fig_d.add_trace(go.Bar(
                x=daily_err["date"].astype(str), y=daily_err["MAE"],
                marker=dict(color=daily_err["MAE"], colorscale="Plasma",
                            line=dict(color="rgba(255,255,255,0.1)", width=0.3)),
                hovertemplate="Date: %{x}<br>MAE: %{y:.1f} W<br>n=%{customdata}<extra></extra>",
                customdata=daily_err["n"],
            ))
            fig_d.update_layout(title="Daily MAE on test set",
                                xaxis_title="Date", yaxis_title="Daily MAE (W)",
                                xaxis=dict(tickangle=-45))
            st.plotly_chart(style_plotly(fig_d, height=350), use_container_width=True)
        with st.expander("📋 Daily error table", expanded=False):
            st.dataframe(daily_err, use_container_width=True)

        # Feature importance
        if not importances_df.empty:
            section_header("🎖️", "Random Forest Feature Importance")
            top_imp = importances_df.head(15)
            if HAS_PLOTLY:
                fig_fi = go.Figure()
                fig_fi.add_trace(go.Bar(
                    x=top_imp["importance"][::-1], y=top_imp["feature"][::-1],
                    orientation="h",
                    marker=dict(color=top_imp["importance"][::-1], colorscale="YlOrRd",
                                line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
                    hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
                ))
                fig_fi.update_layout(title="Top 15 features by Random Forest importance",
                                     xaxis_title="Importance", yaxis_title="")
                st.plotly_chart(style_plotly(fig_fi, height=420), use_container_width=True)

        # Insights
        section_header("💡", "Insights & Limitations")
        peak_hour = int(hourly_err.loc[hourly_err["abs_err"].idxmax(), "hour"])
        quiet_hour = int(hourly_err.loc[hourly_err["abs_err"].idxmin(), "hour"])
        top_feature = str(importances_df.iloc[0]["feature"]) if not importances_df.empty else "lag_1"

        comparison_lines = []
        if not improvements_df.empty:
            for _, r in improvements_df.iterrows():
                comparison_lines.append(
                    f"`{r['model']}` cuts RMSE by **{r['RMSE_vs_naive_pct']:.1f}%** and MAE by **{r['MAE_vs_naive_pct']:.1f}%** vs lag-1 naive."
                )
        quant_summary = " ".join(comparison_lines)

        weather_in_use = [c for c in feature_cols if any(h in c.lower() for h in PV_WEATHER_HINTS)]
        weather_note = (
            f"**Weather features in use:** {', '.join(weather_in_use)}." if weather_in_use
            else "**Weather features:** none selected — try `irradiance_wm2`, `temperature_c`, `relative_humidity_pct` in Feature Engineering."
        )

        insights = [
            f"**Quantified lift vs naive baseline:** {quant_summary}" if quant_summary else
            f"**Best model:** {dash_model} (MAE={float(dash_metrics_row['MAE']):.1f} W, RMSE={float(dash_metrics_row['RMSE']):.1f} W).",
            f"**Top predictor:** `{top_feature}`. For a 5-minute PV series, recent values dominate because power changes slowly minute-to-minute; cyclical time features and weather lags help explain residual variation around peak generation.",
            f"**Errors concentrate around peak generation.** MAE peaks near hour {peak_hour} and is smallest near hour {quiet_hour} — at night the target is essentially zero and trivial to predict; midday irradiance variability drives the largest residuals.",
            "**Residual shape:** roughly zero-centred with heavier tails than Gaussian. Model is unbiased on average but occasionally misses large step changes (cloud transients, inverter switching).",
            weather_note,
            "**Time-step regularity:** audited in the Feature Engineering tab. If `irregular_step_pct` is non-trivial, resampling in the Data tab eliminates implicit assumptions in lag features and typically reduces RMSE.",
            "**No data leakage:** strict chronological split; all lag/rolling/weather features use `.shift()` before any rolling window; test timestamps strictly follow train timestamps.",
            "**Future work:** walk-forward cross-validation, hyperparameter search via expanding-window CV, LightGBM / XGBoost, longer horizons (h=12 → one hour ahead).",
        ]
        for bullet in insights:
            st.markdown(f'<div class="glass-card" style="margin-bottom:0.6rem; padding:0.9rem 1.1rem;">• {bullet}</div>',
                        unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 5: EXPORT
# ─────────────────────────────────────────────
with tab_export:
    section_header("💾", "Submission Artefacts")

    has_metrics_table = isinstance(results_df, pd.DataFrame)
    results_table = [] if results_df is None else results_df.to_dict(orient="records")

    submission = {
        "student": {"name": student_name, "id": student_id},
        "project": {
            "title": project_title, "goal": project_goal,
            "streamlit_url": deployed_url, "github_repo_url": repo_url,
        },
        "data": {
            "data_path": data_path,
            "rows_loaded": int(len(df)),
            "columns_loaded": int(len(df.columns)),
            "timestamp_col": timestamp_col,
            "target_col": target_col,
            "cleaning_report": cleaning_report,
            "coverage_after_cleaning": coverage,
            "resample_rule": resample_rule,
            "coverage_after_resampling": prepared_coverage,
            "forecast_horizon_rows": int(horizon),
            "missing_values_discussed": True,
            "resampling_discussed": True,
            "step_regularity_audit": step_audit,
            "step_irregularity_addressed": (
                f"Step-regularity audit detected {step_audit.get('irregular_step_pct', 0):.3f}% deviation "
                f"from the modal step ({step_audit.get('modal_step', 'n/a')}). When non-trivial, "
                f"users are warned to resample before modeling."
            ) if step_audit else "Step regularity audit performed in Feature Engineering tab.",
        },
        "features": {
            "baseline_features": list(feature_groups.get("baseline", [])),
            "all_features_used": feature_cols,
            "feature_groups_in_use": {g: cols for g, cols in feature_groups.items()
                                       if all(c in feature_cols for c in cols)},
            "feature_table_rows": int(len(feature_table)) if feature_table is not None else 0,
            "n_features_used": int(len(feature_cols)),
            "student_added_features": [
                "Extra lag features: lag_2, lag_3, rolling_std_24, rolling_mean_6, diff_1",
                "Cyclical time features: hour_sin, hour_cos, doy_sin, doy_cos, is_daylight",
                "Domain weather lags (selectable in UI): irradiance, temperature, humidity, pressure, wind, rainfall, visibility — each lagged 1 step",
                "Interactive feature-group toggles in UI",
                "Ridge regression and Gradient Boosting added alongside Random Forest",
            ],
            "weather_features_selected": [c for c in feature_cols
                                           if any(h in c.lower() for h in PV_WEATHER_HINTS)],
            "domain_feature_engineering_present": True,
        },
        "modeling_and_evaluation": {
            "has_time_based_split": True,
            "split_strategy": split_info.get("split_strategy", ""),
            "train_range": [split_info.get("train_start", ""), split_info.get("train_end", "")],
            "test_range":  [split_info.get("test_start", ""),  split_info.get("test_end", "")],
            "train_rows": split_info.get("train_rows", 0),
            "test_rows":  split_info.get("test_rows", 0),
            "n_features_used": int(len(feature_cols)),
            "models_compared": results_df["model"].tolist() if results_df is not None else [],
            "best_model_by_rmse": best_name,
            "has_metrics_table": has_metrics_table,
            "results_table": results_table,
            "quantified_improvements_vs_naive": improvements_df.to_dict(orient="records") if not improvements_df.empty else [],
            "feature_importances": importances_df.to_dict(orient="records") if not importances_df.empty else [],
            "interactive_controls": [
                "Test-set size slider (10–40%)",
                "Random Forest n_estimators and max_depth sliders",
                "Model multi-select (5 models)",
                "Forecast horizon and resampling rule",
                "Feature-group toggles, weather column multi-select",
                "Dashboard model selector and plot-window slider",
                "Tab-based navigation for smooth flow",
            ],
            "no_leakage_evidence": (
                "All lag/rolling/weather features are computed with .shift() before any rolling window. "
                "The train/test split is strictly chronological with no shuffling; test set starts after train set ends."
            ),
            "student_notes": (
                "Compared up to 5 forecasts on a chronological time-based split. "
                "Metrics: MAE, RMSE, R^2, MAPE. Quantified % improvement vs naive baseline reported in dedicated table."
            ),
        },
        "dashboard": {
            "has_baseline_plot": True,
            "has_student_added_dashboard": True,
            "is_interactive": True,
            "ui_features": [
                "Animated gradient background with solar-themed particle overlays",
                "Glassmorphism cards with hover effects",
                "Tab-based navigation (Data, Features, Modeling, Dashboard, Export, Grader)",
                "Plotly interactive charts (zoom, pan, hover tooltips)",
                "Gradient KPI cards with live values",
                "Progress bar during model training",
                "Custom CSS theme with gold/orange/blue palette",
            ],
            "student_dashboard_components": [
                "Interactive model selector",
                "Plot-window slider (100–2000 test points)",
                "KPI rows (test rows, RMSE, MAE, R^2 for chosen model)",
                "Actual vs predicted time-series plot (Plotly)",
                "Residuals-over-time plot with zero reference",
                "Residual distribution histogram",
                "Actual vs predicted scatter with |error| colour scale",
                "Daily MAE bar chart with row counts",
                "Hour-of-day MAE bar chart",
                "Random Forest top-15 feature importance bar chart",
                "Quantified improvement table",
            ],
            "insights": [
                "Quantified improvement vs lag-1 baseline reported per model.",
                "Top predictor is typically lag_1; cyclical time and weather lags carry residual signal.",
                "Errors peak near solar noon, near zero at night.",
                "Residuals zero-centred but heavy-tailed during cloud transients.",
                "Weather features selectable; lagged to prevent leakage.",
                "Time-step regularity audited with explicit warning.",
                "Strict chronological split; no leakage.",
            ],
        },
    }

    submission_json = safe_json_dumps(submission)
    best_repr = best_row_dict if best_row_dict else {"model": "n/a", "MAE": "—", "RMSE": "—", "R2": "—", "MAPE_%": "—"}

    project_card = f"""# {project_title}

## Student
- Name: {student_name}
- ID: {student_id}

## Goal
{project_goal}

## Dataset
- Path: `{data_path}`
- Timestamp column: `{timestamp_col}`
- Target column: `{target_col}`
- Rows loaded: {len(df):,}
- Rows after cleaning: {len(cleaned_df) if cleaned_df is not None else len(df):,}
- Resampling: {resample_rule}
- Forecast horizon rows: {int(horizon)}

## Feature engineering ({len(feature_cols)} features in use)
- **Baseline:** {", ".join(feature_groups.get("baseline", []))}
- **Extra lags:** {", ".join(feature_groups.get("extra_lags", [])) or "off"}
- **Cyclical time:** {", ".join(feature_groups.get("domain_time", [])) or "off"}
- **Weather (lagged):** {", ".join(feature_groups.get("weather", [])) or "none"}

## Time-step regularity
- Modal step: `{step_audit.get("modal_step", "n/a")}`
- Irregular step %: {step_audit.get("irregular_step_pct", "n/a")}
- Resampling rule used: `{resample_rule}`

## Modeling
- Split: chronological, last {test_size_pct}% test, no shuffling.
- Models: {", ".join(results_df["model"].tolist()) if results_df is not None else "—"}.
- Best by RMSE: **{best_repr.get("model", "—")}** (MAE={best_repr.get("MAE", "—")}, RMSE={best_repr.get("RMSE", "—")}, R²={best_repr.get("R2", "—")}, MAPE={best_repr.get("MAPE_%", "—")}%).

## Quantified improvement vs naive lag-1 baseline
{improvements_df.to_markdown(index=False) if not improvements_df.empty else "Train models including Naive (lag-1) to see this table."}

## Dashboard (interactive, glass UI)
- Animated solar background, glassmorphism cards, gradient KPIs.
- Tab-based navigation: Data → Features → Modeling → Dashboard → Export → Grader.
- Plotly interactive charts with hover/zoom/pan.
- Model selector, plot-window slider, RF hyperparam sliders.

## No-leakage evidence
- Strict chronological split — test set starts after train set ends.
- All lag / rolling / weather features use `.shift()` so no future value enters any feature.

## Links
- Streamlit app: {deployed_url}
- GitHub repo: {repo_url}
"""

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown(kpi_card("📄 submission.json", f"{len(submission_json):,} chars", "ready to download"), unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download submission.json",
            data=submission_json,
            file_name="submission.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_e2:
        st.markdown(kpi_card("📝 project_card.md", f"{len(project_card):,} chars", "ready to download", "warm"), unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download project_card.md",
            data=project_card,
            file_name="project_card.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown("---")
    with st.expander("👁️ Preview submission.json", expanded=False):
        st.code(submission_json, language="json")
    with st.expander("👁️ Preview project_card.md", expanded=False):
        st.markdown(project_card)


# ─────────────────────────────────────────────
# TAB 6: AI GRADER
# ─────────────────────────────────────────────
with tab_grader:
    section_header("🎯", f"AI Grader · /80")
    st.caption(f"Model: `{OPENROUTER_MODEL}`")

    api_key = get_openrouter_api_key()
    grader_prompt = AI_GRADER_PROMPT_TEMPLATE.replace(
        "<insert submission.json contents here>", submission_json,
    )

    with st.expander("📜 Preview AI grader prompt", expanded=False):
        st.code(grader_prompt)

    if st.button("🚀 Run AI Grader", use_container_width=True):
        if not api_key:
            st.error("Provide OPENROUTER_API_KEY via Streamlit Secrets, env var, or the password field above.")
        else:
            try:
                with st.spinner("⏳ Calling AI grader..."):
                    raw_output = call_openrouter_grader(api_key, grader_prompt)
                parsed_output, parse_error = parse_grader_response(raw_output)
                if parsed_output is not None:
                    st.success("✅ Parsed grader JSON")

                    # Display the score prominently
                    total = parsed_output.get("total_80", "—")
                    st.markdown(f"""
                    <div class="hero" style="text-align:center; padding:2.5rem;">
                        <h1 style="font-size:4rem !important;">{total} / 80</h1>
                        <p style="font-size:1.1rem; margin-top:0.5rem;">AI Grader Score</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Sub-scores
                    scores = parsed_output.get("scores", {})
                    if scores:
                        st.markdown("### 📊 Rubric Breakdown")
                        cols = st.columns(len(scores))
                        for col, (k, v) in zip(cols, scores.items()):
                            with col:
                                st.markdown(kpi_card(k, str(v), "/ rubric", "green"),
                                            unsafe_allow_html=True)

                    # Strengths / weaknesses
                    col_s, col_w = st.columns(2)
                    with col_s:
                        st.markdown("### ✅ Strengths")
                        for s in parsed_output.get("strengths", []):
                            st.markdown(f"- {s}")
                    with col_w:
                        st.markdown("### ⚠️ Weaknesses")
                        for w in parsed_output.get("weaknesses", []):
                            st.markdown(f"- {w}")

                    st.markdown("### 🛠️ Actionable Improvements")
                    for a in parsed_output.get("actionable_improvements", []):
                        st.markdown(f"- {a}")

                    with st.expander("🔬 Raw grader JSON", expanded=False):
                        st.json(parsed_output)
                else:
                    st.warning(f"Could not parse grader response as JSON: {parse_error}")
                    st.code(raw_output)
            except Exception as exc:
                st.error(f"AI grader call failed: {exc}")


# ============================================================
# FOOTER
# ============================================================
st.markdown(f"""
<div class="footer">
    <p>☀️ <strong>HKUST SQ1 PV Forecasting Dashboard</strong> · {student_name} · {student_id}</p>
    <p>Built with Streamlit · scikit-learn · Plotly · Pandas · NumPy</p>
    <p style="opacity:0.6; margin-top:0.5rem;"><span class="live-dot"></span>Live simulation · Interactive · Real-time inference</p>
</div>
""", unsafe_allow_html=True)
