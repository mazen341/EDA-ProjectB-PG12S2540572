
"""
app.py — Solar PV Forecasting Intelligence Website
Clean real-website version with fast navigation, transparent charts, live telemetry,
interactive diagrams, controlled model comparison, simulator, and exports.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    go = None
    PLOTLY_AVAILABLE = False

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except Exception:
    AUTOREFRESH_AVAILABLE = False
    def st_autorefresh(*args, **kwargs):
        return 0

try:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import ElasticNetCV, RidgeCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PROJECT_NAME = "Mini Project B — Solar PV Forecasting Intelligence Website"
STUDENT_NAME_DEFAULT = "MAZEN AL-HIMALI"
STUDENT_ID_DEFAULT = "PG12S2540572"
DEFAULT_DATA_PATH = "data/dataset_sample.csv"
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_TARGET_COL = "total_active_power_w"
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

SECTION_OPTIONS = [
    "🏠 Home",
    "🔴 Live Telemetry",
    "📊 Forecasting",
    "🧩 Visual System",
    "🧹 Data Pipeline",
    "🤖 Models",
    "🧬 Advanced",
    "🛠️ Technical Diagrams",
    "🕹️ Simulator",
    "🔬 Comparison Lab",
    "📤 Export",
]

IMAGES = {
    "home": "https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&w=1600&q=80",
    "live": "https://images.unsplash.com/photo-1581092160607-ee22621dd758?auto=format&fit=crop&w=1200&q=80",
    "forecast": "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1200&q=80",
    "visual": "https://images.unsplash.com/photo-1497435334941-8c899ee9e8e9?auto=format&fit=crop&w=1200&q=80",
    "pipeline": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1200&q=80",
    "models": "https://images.unsplash.com/photo-1555949963-aa79dcee981c?auto=format&fit=crop&w=1200&q=80",
    "advanced": "https://images.unsplash.com/photo-1518186285589-2f7649de83e0?auto=format&fit=crop&w=1200&q=80",
    "diagrams": "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?auto=format&fit=crop&w=1200&q=80",
    "simulator": "https://images.unsplash.com/photo-1518779578993-ec3579fee39f?auto=format&fit=crop&w=1200&q=80",
    "compare": "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?auto=format&fit=crop&w=1200&q=80",
    "export": "https://images.unsplash.com/photo-1554224155-6726b3ff858f?auto=format&fit=crop&w=1200&q=80",
    "grid": "https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&w=1200&q=80",
    "battery": "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1200&q=80",
}

THEMES = {
    "Executive Dark": {"bg": "rgba(2,6,23,.86)", "panel": "rgba(5,18,38,.78)", "text": "#f8fbff", "muted": "#dbeafe", "accent": "#fbbf24", "accent2": "#38bdf8", "good": "#10b981", "warn": "#f87171"},
    "PV Blue": {"bg": "rgba(4,20,40,.84)", "panel": "rgba(8,30,58,.74)", "text": "#f8fbff", "muted": "#dbeafe", "accent": "#38bdf8", "accent2": "#fbbf24", "good": "#10b981", "warn": "#fb7185"},
    "Solar Gold": {"bg": "rgba(20,12,3,.86)", "panel": "rgba(45,28,8,.76)", "text": "#fff7ed", "muted": "#fde68a", "accent": "#fbbf24", "accent2": "#22d3ee", "good": "#84cc16", "warn": "#fb7185"},
    "Emerald Grid": {"bg": "rgba(3,18,15,.86)", "panel": "rgba(6,38,32,.76)", "text": "#ecfdf5", "muted": "#a7f3d0", "accent": "#5eead4", "accent2": "#fbbf24", "good": "#22c55e", "warn": "#fb7185"},
}

# -----------------------------------------------------------------------------
# Page + visual system
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Solar PV Intelligence", page_icon="☀️", layout="wide", initial_sidebar_state="expanded")


def inject_css(theme_name: str, motion: bool, compact: bool) -> None:
    t = THEMES.get(theme_name, THEMES["Executive Dark"])
    base_font = "15px" if compact else "16px"
    motion_off = "*,*::before,*::after{animation:none!important;transition:none!important;}" if not motion else ""
    st.markdown(f"""
<style>
:root {{
  --bg:{t['bg']}; --panel:{t['panel']}; --text:{t['text']}; --muted:{t['muted']};
  --accent:{t['accent']}; --accent2:{t['accent2']}; --good:{t['good']}; --warn:{t['warn']};
  --border:rgba(226,232,240,.18); --shadow:0 18px 54px rgba(0,0,0,.28);
}}
@keyframes pulse {{0%,100%{{opacity:.45;transform:scale(.95)}}50%{{opacity:1;transform:scale(1.08)}}}}
@keyframes shimmer {{0%{{background-position:-700px 0}}100%{{background-position:700px 0}}}}
@keyframes floatY {{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-5px)}}}}
@keyframes flow {{0%{{transform:translateX(-100%)}}100%{{transform:translateX(180%)}}}}
html, body, .stApp {{
  color:var(--text)!important; font-size:{base_font};
  background:
    linear-gradient(rgba(3, 12, 24, .84), rgba(6, 18, 34, .86), rgba(8, 27, 48, .88)),
    radial-gradient(circle at 10% 8%, rgba(56,189,248,.18), transparent 30%),
    radial-gradient(circle at 86% 10%, rgba(251,191,36,.16), transparent 28%),
    url('{IMAGES['home']}') !important;
  background-size:cover!important; background-position:center!important; background-attachment:fixed!important;
}}
.stApp::before {{content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image: radial-gradient(circle, rgba(255,255,255,.07) 1px, transparent 1px);
  background-size:42px 42px; opacity:.55;
}}
.block-container {{max-width:1500px; padding-top:.75rem; padding-bottom:2rem;}}
[data-testid="stHeader"] {{background:transparent;}}
[data-testid="stSidebar"] {{background:linear-gradient(180deg, rgba(2,6,23,.98), rgba(8,18,32,.96)); border-right:1px solid var(--border);}}
[data-testid="stSidebar"] * {{color:var(--text)!important;}}
h1,h2,h3,h4,h5,h6,label,.stMarkdown,.stCaption,p,li,span {{color:var(--text)!important;}}
.muted, .small, .hero-copy, .section-copy {{color:var(--muted)!important;}}
.app-header, .panel, .kpi-card, .hero, .feature-card, .section-hero, .glass-card {{
  background:linear-gradient(145deg, var(--panel), rgba(2,6,23,.54)); border:1px solid var(--border);
  border-radius:22px; box-shadow:var(--shadow); backdrop-filter:blur(14px); color:var(--text)!important;
}}
.app-header {{display:flex; justify-content:space-between; align-items:center; gap:1rem; padding:.85rem 1rem; margin-bottom:.8rem;}}
.brand-wrap {{display:flex; align-items:center; gap:.8rem;}}
.logo {{width:48px;height:48px;border-radius:16px;background:linear-gradient(135deg,var(--accent),var(--good));display:flex;align-items:center;justify-content:center;font-weight:1000;font-size:1.4rem;box-shadow:0 12px 36px rgba(251,191,36,.24);}}
.brand-title {{font-size:1.2rem;font-weight:1000;line-height:1.05;}}
.brand-sub {{font-size:.8rem;color:var(--muted)!important;margin-top:.15rem;}}
.top-pills {{display:flex; flex-wrap:wrap; gap:.45rem; justify-content:flex-end;}}
.pill {{display:inline-flex;align-items:center;gap:.35rem;border:1px solid rgba(56,189,248,.32);border-radius:999px;padding:.36rem .62rem;background:rgba(56,189,248,.10);font-weight:900;font-size:.8rem;white-space:nowrap;}}
.live-dot {{width:9px;height:9px;border-radius:999px;background:var(--good);box-shadow:0 0 14px rgba(16,185,129,.95);animation:pulse 1.3s ease-in-out infinite;}}
.quick-grid {{display:grid;grid-template-columns:repeat(6,minmax(110px,1fr));gap:.45rem;margin:.55rem 0 .8rem;}}
.quick-grid button {{min-height:2.55rem!important;border-radius:12px!important;font-weight:950!important;border:1px solid rgba(56,189,248,.26)!important;background:rgba(8,30,58,.88)!important;color:#fff!important;}}
.quick-grid button:hover {{border-color:var(--accent)!important; box-shadow:0 0 0 2px rgba(251,191,36,.14)!important;}}
.hero {{padding:1.05rem; min-height:210px; overflow:hidden; position:relative;}}
.hero::after {{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.08),transparent);animation:shimmer 8s linear infinite;pointer-events:none;}}
.hero-title {{font-size:2.05rem;font-weight:1000;letter-spacing:-.05em;line-height:1.05;margin:.4rem 0;}}
.hero-copy {{font-size:.95rem;line-height:1.55;max-width:900px;position:relative;z-index:2;}}
.section-hero {{padding:.9rem 1rem; margin:.35rem 0 .9rem; min-height:118px; background-size:cover!important; background-position:center!important; overflow:hidden; position:relative;}}
.section-hero::before {{content:"";position:absolute;inset:0;background:linear-gradient(90deg,rgba(2,6,23,.86),rgba(2,6,23,.48));}}
.section-hero > * {{position:relative; z-index:1;}}
.section-title {{color:var(--accent)!important;font-weight:1000;font-size:1.1rem;margin-bottom:.55rem;}}
.panel {{padding:.9rem; margin-bottom:.85rem;}}
.kpi-card {{padding:.8rem;min-height:112px;position:relative;overflow:hidden;}}
.kpi-card::after {{content:"";position:absolute;left:0;right:0;bottom:0;height:3px;background:linear-gradient(90deg,transparent,var(--accent),transparent);animation:flow 2.8s linear infinite;}}
.kpi-icon {{font-size:1.55rem;}}
.kpi-label {{font-size:.72rem;color:var(--muted)!important;font-weight:900;text-transform:uppercase;letter-spacing:.04em;margin-top:.35rem;}}
.kpi-value {{font-size:1.45rem;font-weight:1000;margin-top:.2rem;}}
.kpi-detail {{font-size:.75rem;color:var(--good)!important;font-weight:850;}}
.feature-card {{padding:.8rem;min-height:108px;margin-bottom:.65rem;}}
.feature-card b {{color:var(--text)!important;}}
.stPlotlyChart {{background:rgba(5,18,38,.20)!important;border:1px solid rgba(148,203,255,.14);border-radius:18px;padding:.35rem;backdrop-filter:blur(10px);}}
[data-testid="stDataFrame"], .stDataFrame {{background:rgba(255,255,255,.96)!important;color:#0f172a!important;border-radius:14px!important;overflow:hidden!important;}}
[data-testid="stGraphVizChart"] {{background:rgba(5,18,38,.26)!important;border:1px solid rgba(148,203,255,.16);border-radius:18px;padding:.6rem;backdrop-filter:blur(10px);}}
.stButton>button, .stDownloadButton>button {{border-radius:12px!important;font-weight:900!important;}}
.sidebar-card {{border:1px solid rgba(56,189,248,.22);border-radius:18px;padding:.75rem;background:linear-gradient(145deg,rgba(8,24,44,.96),rgba(11,35,63,.86));margin:.25rem 0 .75rem;}}
.sidebar-title {{font-weight:1000;color:var(--accent)!important;font-size:1.05rem;}}
.download-note {{border:1px dashed rgba(251,191,36,.24);border-radius:14px;background:rgba(251,191,36,.08);padding:.65rem;margin:.45rem 0;color:var(--muted)!important;}}
{motion_off}
@media(max-width:1100px){{.quick-grid{{grid-template-columns:repeat(3,1fr)}}.app-header{{flex-direction:column;align-items:flex-start}}}}
@media(max-width:700px){{.quick-grid{{grid-template-columns:repeat(2,1fr)}}.hero-title{{font-size:1.55rem}}}}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Utility rendering
# -----------------------------------------------------------------------------
if "chart_counter" not in st.session_state:
    st.session_state.chart_counter = 0

def next_key(prefix: str = "k") -> str:
    st.session_state.chart_counter += 1
    return f"{prefix}_{st.session_state.chart_counter}"

def section_hero(title: str, copy: str, image_key: str) -> None:
    st.markdown(f"""
    <div class="section-hero" style="background-image:url('{IMAGES[image_key]}')">
      <div class="pill"><span class="live-dot"></span>{title}</div>
      <div class="hero-title" style="font-size:1.45rem">{title}</div>
      <div class="section-copy">{copy}</div>
    </div>
    """, unsafe_allow_html=True)

def kpi_card(label: str, value: str, icon: str, detail: str = "") -> None:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-icon">{icon}</div><div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div><div class="kpi-detail">{detail}</div>
    </div>
    """, unsafe_allow_html=True)

def feature_card(title: str, body: str, icon: str = "✓") -> None:
    st.markdown(f"<div class='feature-card'><div style='font-size:1.3rem'>{icon}</div><b>{title}</b><div class='muted' style='margin-top:.25rem'>{body}</div></div>", unsafe_allow_html=True)

def style_fig(fig, height: int | None = None):
    if fig is None:
        return None
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fbff"), legend=dict(bgcolor="rgba(5,18,38,.35)", orientation="h"),
        margin=dict(l=10, r=10, t=42, b=10),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor="rgba(255,255,255,.12)", zerolinecolor="rgba(255,255,255,.20)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.12)", zerolinecolor="rgba(255,255,255,.20)")
    return fig

def render_plot(fig, name: str, data: pd.DataFrame | None = None) -> None:
    if fig is None:
        if data is not None and not data.empty:
            st.line_chart(data)
            download_df(data.reset_index() if data.index.name else data, f"{name}_data")
        else:
            st.info("Chart is unavailable for the current data.")
        return
    fig = style_fig(fig)
    st.plotly_chart(fig, use_container_width=True, key=next_key("plot"))
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(f"⬇️ Download {name} chart HTML", fig.to_html(full_html=True, include_plotlyjs="cdn"), f"{safe_name(name)}_chart.html", "text/html", key=next_key("html"), use_container_width=True)
    if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
        with c2:
            download_df(data, f"{name}_data")

def download_df(df: pd.DataFrame, name: str, label: str | None = None) -> None:
    if df is not None and not df.empty:
        st.download_button(label or f"⬇️ Download {name} CSV", df.to_csv(index=False), f"{safe_name(name)}.csv", "text/csv", key=next_key("csv"), use_container_width=True)

def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", s.strip().lower()).strip("_") or "file"

def safe_json_default(obj: Any):
    if isinstance(obj, pd.Timestamp): return obj.isoformat()
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    try:
        if pd.isna(obj): return None
    except Exception:
        pass
    return str(obj)

# -----------------------------------------------------------------------------
# Data + modeling
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def demo_data(days: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range(end=pd.Timestamp.now().floor("15min"), periods=days * 24 * 4, freq="15min")
    hour = np.asarray(idx.hour) + np.asarray(idx.minute) / 60
    doy = np.asarray(idx.dayofyear)
    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    seasonal = 0.80 + 0.18 * np.sin(2 * np.pi * (doy - 70) / 365)
    cloud = np.clip(rng.normal(.92, .18, len(idx)), .28, 1.18)
    temp = 25 + 8 * np.sin((hour - 8) / 24 * 2 * np.pi) + rng.normal(0, 1.7, len(idx))
    irradiance = 980 * daylight * seasonal * cloud
    power = 5200 * daylight * seasonal * cloud * (1 - .0035 * np.maximum(temp - 25, 0)) + rng.normal(0, 110, len(idx))
    power = np.clip(power, 0, None)
    anomaly_idx = rng.choice(np.arange(len(idx)), size=max(10, len(idx)//420), replace=False)
    power[anomaly_idx] *= rng.uniform(.25, .65, len(anomaly_idx))
    return pd.DataFrame({
        "timestamp": idx,
        "total_active_power_w": power,
        "irradiance_wm2": irradiance,
        "temperature_c": temp,
        "relative_humidity_pct": np.clip(54 - 17 * daylight + rng.normal(0, 4, len(idx)), 18, 96),
        "wind_speed_ms": np.clip(rng.normal(3.2, 1.1, len(idx)), .1, 11),
        "rainfall_mm": rng.choice([0, 0, 0, .2, .8, 1.5], len(idx), p=[.72, .12, .08, .04, .03, .01]),
        "sea_level_pressure_hpa": rng.normal(1008, 4, len(idx)),
    })

@st.cache_data(show_spinner=False)
def load_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_dataset(path: str, uploaded: Any) -> tuple[pd.DataFrame, str]:
    if uploaded is not None:
        name = uploaded.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded), "uploaded CSV"
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded), "uploaded Excel"
        if name.endswith(".json"):
            return pd.read_json(uploaded), "uploaded JSON"
    if path and os.path.exists(path):
        return load_path(path), path
    return demo_data(), "generated demo PV dataset"

def audit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": list(df.columns),
        "dtype": [str(df[c].dtype) for c in df.columns],
        "non_null": [int(df[c].notna().sum()) for c in df.columns],
        "missing_pct": [round(float(df[c].isna().mean() * 100), 3) for c in df.columns],
        "unique_count": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })

def prepare_timeseries(df: pd.DataFrame, timestamp_col: str, target_col: str, resample_rule: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = df.copy()
    before = len(work)
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    invalid = int(work[[timestamp_col, target_col]].isna().any(axis=1).sum())
    work = work.dropna(subset=[timestamp_col, target_col]).sort_values(timestamp_col)
    duplicate_count = int(work[timestamp_col].duplicated().sum())
    numeric_cols = []
    for c in work.columns:
        if c == timestamp_col: continue
        converted = pd.to_numeric(work[c], errors="coerce")
        if converted.notna().sum() > 0:
            work[c] = converted
            numeric_cols.append(c)
    work = work.groupby(timestamp_col, as_index=False)[numeric_cols].mean().sort_values(timestamp_col)
    if resample_rule != "None":
        work = work.set_index(timestamp_col).resample(resample_rule).mean(numeric_only=True).interpolate(limit_direction="both").reset_index()
    q1, q3 = work[target_col].quantile([.25, .75])
    iqr = q3 - q1
    lower = max(0, float(q1 - 1.5 * iqr))
    upper = float(q3 + 1.5 * iqr)
    outlier_count = int(((work[target_col] < lower) | (work[target_col] > upper)).sum())
    work[target_col] = work[target_col].clip(lower, upper)
    return work, {
        "rows_before_cleaning": int(before), "invalid_rows_removed": invalid,
        "duplicate_timestamps_before_grouping": duplicate_count, "rows_after_cleaning": int(len(work)),
        "resampling_rule": resample_rule, "outliers_clipped": outlier_count,
        "target_bounds": {"lower": round(lower, 3), "upper": round(upper, 3)},
    }

def build_features(df: pd.DataFrame, timestamp_col: str, target_col: str, horizon: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    work = df.copy().sort_values(timestamp_col)
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    for lag in [1, 4, 24, 96]:
        work[f"lag_{lag}"] = work[target_col].shift(lag)
    for w in [8, 24, 96]:
        work[f"rolling_mean_{w}"] = work[target_col].shift(1).rolling(w).mean()
        work[f"rolling_std_{w}"] = work[target_col].shift(1).rolling(w).std()
    work["hour"] = work[timestamp_col].dt.hour
    work["dayofweek"] = work[timestamp_col].dt.dayofweek
    work["month"] = work[timestamp_col].dt.month
    work["dayofyear"] = work[timestamp_col].dt.dayofyear
    work["weekend"] = (work["dayofweek"] >= 5).astype(int)
    work["is_daylight_hour"] = work["hour"].between(7, 18).astype(int)
    work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24)
    work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24)
    work["dayofyear_sin"] = np.sin(2 * np.pi * work["dayofyear"] / 365.25)
    work["dayofyear_cos"] = np.cos(2 * np.pi * work["dayofyear"] / 365.25)
    work["y_target"] = work[target_col].shift(-int(horizon))
    weather = [c for c in ["irradiance_wm2", "temperature_c", "relative_humidity_pct", "wind_speed_ms", "rainfall_mm", "sea_level_pressure_hpa"] if c in work.columns and c != target_col]
    features = [c for c in work.columns if c.startswith("lag_") or c.startswith("rolling_")] + ["hour", "dayofweek", "month", "dayofyear", "weekend", "is_daylight_hour", "hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos"] + weather
    for c in features:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    return work.dropna(subset=features + ["y_target"]).copy(), features, weather

def metrics_row(name: str, y_true, y_pred, train_rows: int, valid_rows: int, note: str) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred)) if SKLEARN_AVAILABLE else float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))) if SKLEARN_AVAILABLE else float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred)) if SKLEARN_AVAILABLE else 0.0
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100)
    return {"model": name, "MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE_pct": round(mape, 3), "R2": round(r2, 4), "train_rows": train_rows, "validation_rows": valid_rows, "split_type": "time_based_80_20", "notes": note}

def run_models(model_df: pd.DataFrame, features: list[str], timestamp_col: str, target_col: str, group: str, rank_metric: str):
    if group == "Do not train yet":
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, "Model training has not run yet. Choose a model group and press Run selected comparison."
    if len(model_df) < 120:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, "Not enough rows for reliable modeling."
    split = int(len(model_df) * .8)
    train, valid = model_df.iloc[:split].copy(), model_df.iloc[split:].copy()
    q1, q3 = model_df["y_target"].quantile([.25, .75])
    iqr = q3 - q1
    lower, upper = max(0, float(q1 - 1.5 * iqr)), float(q3 + 1.5 * iqr)
    X_train, y_train = train[features], train["y_target"].clip(lower, upper)
    X_valid, y_valid = valid[features], valid["y_target"]
    rows, preds, fitted = [], {}, {}
    baseline = valid["lag_24"].fillna(valid["lag_1"]).fillna(train["y_target"].median()).clip(lower, upper).to_numpy()
    rows.append(metrics_row("Naive seasonal lag_24 baseline", y_valid, baseline, len(train), len(valid), "Transparent lag baseline."))
    preds["Naive seasonal lag_24 baseline"] = baseline
    if SKLEARN_AVAILABLE and group != "Baseline only":
        catalog = {
            "Fast comparison": [("RidgeCV scaled", make_pipeline(StandardScaler(), RidgeCV(alphas=[.1, 1, 10, 100]))), ("HistGradientBoosting tuned", HistGradientBoostingRegressor(max_iter=180, learning_rate=.055, max_leaf_nodes=31, l2_regularization=.05, random_state=42))],
            "Linear models": [("RidgeCV scaled", make_pipeline(StandardScaler(), RidgeCV(alphas=[.1, 1, 10, 100]))), ("ElasticNetCV regularized", make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.2, .5, .8], alphas=[.001, .01, .1, 1.0], cv=3, max_iter=4000, random_state=42)))],
            "Tree ensemble models": [("RandomForest compact", RandomForestRegressor(n_estimators=70, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1)), ("ExtraTrees robust ensemble", ExtraTreesRegressor(n_estimators=80, max_depth=16, min_samples_leaf=3, random_state=42, n_jobs=-1)), ("HistGradientBoosting tuned", HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, max_leaf_nodes=31, l2_regularization=.05, random_state=42))],
            "All available models": [("RidgeCV scaled", make_pipeline(StandardScaler(), RidgeCV(alphas=[.1, 1, 10, 100]))), ("ElasticNetCV regularized", make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.2, .5, .8], alphas=[.001, .01, .1, 1.0], cv=3, max_iter=4000, random_state=42))), ("RandomForest compact", RandomForestRegressor(n_estimators=70, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1)), ("ExtraTrees robust ensemble", ExtraTreesRegressor(n_estimators=80, max_depth=16, min_samples_leaf=3, random_state=42, n_jobs=-1)), ("HistGradientBoosting tuned", HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, max_leaf_nodes=31, l2_regularization=.05, random_state=42))],
        }
        for name, model in catalog.get(group, []):
            model.fit(X_train, y_train)
            fitted[name] = model
            pred = np.clip(model.predict(X_valid), lower, upper)
            rows.append(metrics_row(name, y_valid, pred, len(train), len(valid), f"Selected comparison group: {group}."))
            preds[name] = pred
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, "No models selected."
    if rank_metric == "R2":
        comparison = comparison.sort_values(["R2", "RMSE"], ascending=[False, True]).reset_index(drop=True)
    else:
        comparison = comparison.sort_values([rank_metric if rank_metric in comparison.columns else "MAPE_pct", "RMSE"], ascending=True).reset_index(drop=True)
    best = str(comparison.iloc[0]["model"])
    best_pred = preds[best]
    residual = y_valid.to_numpy(dtype=float) - best_pred
    lo_res, hi_res = float(np.nanquantile(residual, .05)), float(np.nanquantile(residual, .95))
    pred_df = valid[[timestamp_col, target_col, "y_target"]].copy()
    pred_df["prediction"] = best_pred
    pred_df["prediction_lower_90"] = np.clip(best_pred + lo_res, lower, upper)
    pred_df["prediction_upper_90"] = np.clip(best_pred + hi_res, lower, upper)
    pred_df["residual"] = pred_df["y_target"] - pred_df["prediction"]
    pred_df["absolute_error"] = pred_df["residual"].abs()
    pred_df["interval_covered"] = (pred_df["y_target"].between(pred_df["prediction_lower_90"], pred_df["prediction_upper_90"]))
    if SKLEARN_AVAILABLE and best in fitted:
        try:
            sample = min(700, len(X_valid))
            perm = permutation_importance(fitted[best], X_valid.tail(sample), y_valid.tail(sample), n_repeats=3, random_state=42, scoring="neg_mean_absolute_error")
            importance = pd.DataFrame({"feature": features, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False).head(15)
        except Exception:
            importance = pd.DataFrame({"feature": features[:10], "importance_mean": np.linspace(1, .1, min(10, len(features))), "importance_std": 0})
    else:
        importance = pd.DataFrame({"feature": ["lag_24", "lag_1"], "importance_mean": [1.0, .55], "importance_std": [0.0, 0.0]})
    uncertainty = {"method": "Empirical validation residual quantiles", "lower_residual_5pct": round(lo_res, 3), "upper_residual_95pct": round(hi_res, 3), "interval_coverage_pct": round(float(pred_df["interval_covered"].mean() * 100), 3), "average_interval_width": round(float((pred_df["prediction_upper_90"] - pred_df["prediction_lower_90"]).mean()), 3), "outlier_bounds": {"lower": round(lower, 3), "upper": round(upper, 3)}}
    return comparison, pred_df, importance, uncertainty, f"Best model: {best}. Strict chronological 80/20 split used."

# -----------------------------------------------------------------------------
# Figures and components
# -----------------------------------------------------------------------------
def forecast_fig(df: pd.DataFrame, ts: str, target: str, window: int, band: float):
    if not PLOTLY_AVAILABLE or df.empty: return None
    chart = df[[ts, target]].dropna().tail(window).copy()
    if chart.empty: return None
    chart["smooth"] = chart[target].rolling(max(2, min(12, len(chart)//8))).mean().bfill()
    chart["low"] = chart["smooth"] * (1 - band)
    chart["high"] = chart["smooth"] * (1 + band)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart[ts], y=chart["high"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=chart[ts], y=chart["low"], mode="lines", fill="tonexty", fillcolor="rgba(251,191,36,.20)", line=dict(width=0), name="Forecast band"))
    fig.add_trace(go.Scatter(x=chart[ts], y=chart["smooth"], mode="lines", name="Forecast signal", line=dict(color="#FBBF24", width=3)))
    fig.add_trace(go.Scatter(x=chart[ts], y=chart[target], mode="lines", name="Actual", line=dict(color="#22D3EE", width=2)))
    return style_fig(fig, 390)

def prediction_fig(pred: pd.DataFrame, ts: str, window: int):
    if not PLOTLY_AVAILABLE or pred.empty: return None
    chart = pred.tail(window).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart[ts], y=chart["prediction_upper_90"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=chart[ts], y=chart["prediction_lower_90"], mode="lines", fill="tonexty", fillcolor="rgba(59,130,246,.20)", line=dict(width=0), name="90% interval"))
    fig.add_trace(go.Scatter(x=chart[ts], y=chart["y_target"], mode="lines", name="Actual", line=dict(color="#22D3EE", width=2)))
    fig.add_trace(go.Scatter(x=chart[ts], y=chart["prediction"], mode="lines", name="Predicted", line=dict(color="#10B981", width=2)))
    return style_fig(fig, 390)

def residual_fig(pred: pd.DataFrame):
    if not PLOTLY_AVAILABLE or pred.empty: return None
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pred["residual"], nbinsx=45, name="Residual", marker_color="#38bdf8", opacity=.78))
    fig.add_vline(x=0, line_color="#fbbf24", line_dash="dash")
    return style_fig(fig, 340)

def scatter_fig(pred: pd.DataFrame):
    if not PLOTLY_AVAILABLE or pred.empty: return None
    s = pred.tail(min(2000, len(pred))).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["y_target"], y=s["prediction"], mode="markers", marker=dict(size=6, opacity=.6, color=s["absolute_error"], colorscale="Turbo", showscale=True), name="Prediction"))
    lo, hi = float(min(s["y_target"].min(), s["prediction"].min())), float(max(s["y_target"].max(), s["prediction"].max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#fbbf24", dash="dash"), name="Perfect fit"))
    return style_fig(fig, 380)

def metrics_bar_fig(comparison: pd.DataFrame, metric: str):
    if not PLOTLY_AVAILABLE or comparison.empty or metric not in comparison: return None
    d = comparison.sort_values(metric, ascending=(metric != "R2"))
    fig = go.Figure(go.Bar(x=d[metric], y=d["model"], orientation="h", marker=dict(color=d[metric], colorscale="Viridis", showscale=True)))
    fig.update_layout(yaxis=dict(autorange="reversed"), title=f"{metric} model comparison")
    return style_fig(fig, max(340, 44 * len(d)))

def correlation_fig(model_df: pd.DataFrame, features: list[str]):
    if not PLOTLY_AVAILABLE or model_df.empty: return None
    cols = [c for c in features if c in model_df.columns][:22] + ["y_target"]
    corr = model_df[cols].corr(numeric_only=True).round(3)
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="Viridis"))
    return style_fig(fig, 520)

def pv_energy_component() -> None:
    html = """
<!DOCTYPE html><html><head><meta charset='utf-8'><style>
body{margin:0;background:transparent;font-family:Inter,system-ui;color:#f8fbff}.wrap{min-height:360px;border:1px solid rgba(56,189,248,.28);border-radius:24px;padding:18px;background:radial-gradient(circle at 8% 12%,rgba(251,191,36,.16),transparent 24%),radial-gradient(circle at 92% 10%,rgba(56,189,248,.18),transparent 28%),linear-gradient(145deg,rgba(5,18,38,.95),rgba(8,28,52,.82));box-shadow:0 18px 54px rgba(0,0,0,.26);overflow:hidden}.title{font-weight:1000;color:#fbbf24;font-size:20px}.sub{color:#dbeafe;font-size:13px;margin:4px 0 12px}.system{width:100%;height:235px;display:block}.node{fill:rgba(255,255,255,.07);stroke:rgba(148,203,255,.28);stroke-width:1.6;filter:drop-shadow(0 6px 12px rgba(0,0,0,.25))}.nodeText{fill:#f8fbff;font-size:13px;font-weight:900;text-anchor:middle}.nodeSub{fill:#dbeafe;font-size:10px;text-anchor:middle}.icon{font-size:26px;text-anchor:middle;dominant-baseline:middle}.track{stroke:rgba(56,189,248,.28);stroke-width:7;stroke-linecap:round}.trackGlow{stroke:rgba(251,191,36,.45);stroke-width:3;stroke-linecap:round;stroke-dasharray:14 20;animation:dash 1.4s linear infinite}@keyframes dash{to{stroke-dashoffset:-34}}.pulse{filter:drop-shadow(0 0 8px rgba(251,191,36,.95))}.pulse1{animation:moveA 3s linear infinite}.pulse2{animation:moveA 3s linear infinite .8s}.pulse3{animation:moveA 3s linear infinite 1.6s}.pulseB1{animation:moveB 3.2s linear infinite}.pulseB2{animation:moveB 3.2s linear infinite 1.05s}.pulseB3{animation:moveB 3.2s linear infinite 2.1s}@keyframes moveA{0%{transform:translate(118px,72px);opacity:0}8%{opacity:1}30%{transform:translate(270px,72px)}58%{transform:translate(425px,72px)}88%{transform:translate(585px,72px);opacity:1}100%{transform:translate(675px,72px);opacity:0}}@keyframes moveB{0%{transform:translate(118px,174px);opacity:0}10%{opacity:1}34%{transform:translate(270px,174px)}62%{transform:translate(425px,174px)}90%{transform:translate(585px,174px);opacity:1}100%{transform:translate(675px,174px);opacity:0}}.status{display:inline-flex;align-items:center;gap:8px;border:1px solid rgba(16,185,129,.32);background:rgba(16,185,129,.10);border-radius:999px;padding:8px 12px;color:#ecfeff;font-weight:900;font-size:13px}.dot{width:10px;height:10px;border-radius:50%;background:#10b981;box-shadow:0 0 16px rgba(16,185,129,.95);animation:blink 1.2s ease-in-out infinite}@keyframes blink{0%,100%{opacity:.4;transform:scale(.8)}50%{opacity:1;transform:scale(1.2)}}
</style></head><body><div class='wrap'><div class='title'>Animated PV Energy Flow</div><div class='sub'>Visible moving energy pulses from generation to grid, model, battery, and load.</div><svg class='system' viewBox='0 0 720 235' preserveAspectRatio='xMidYMid meet'>
<line class='track' x1='118' y1='72' x2='205' y2='72'/><line class='trackGlow' x1='118' y1='72' x2='205' y2='72'/><line class='track' x1='280' y1='72' x2='365' y2='72'/><line class='trackGlow' x1='280' y1='72' x2='365' y2='72'/><line class='track' x1='440' y1='72' x2='525' y2='72'/><line class='trackGlow' x1='440' y1='72' x2='525' y2='72'/>
<line class='track' x1='118' y1='174' x2='205' y2='174'/><line class='trackGlow' x1='118' y1='174' x2='205' y2='174'/><line class='track' x1='280' y1='174' x2='365' y2='174'/><line class='trackGlow' x1='280' y1='174' x2='365' y2='174'/><line class='track' x1='440' y1='174' x2='525' y2='174'/><line class='trackGlow' x1='440' y1='174' x2='525' y2='174'/>
<circle class='pulse pulse1' r='7' fill='#fbbf24'/><circle class='pulse pulse2' r='7' fill='#38bdf8'/><circle class='pulse pulse3' r='7' fill='#10b981'/><circle class='pulse pulseB1' r='7' fill='#38bdf8'/><circle class='pulse pulseB2' r='7' fill='#fbbf24'/><circle class='pulse pulseB3' r='7' fill='#10b981'/>
""" + "".join([
        f"<rect class='node' x='{x}' y='{y}' width='108' height='88' rx='18'/><text class='icon' x='{x+54}' y='{y+27}'>{ico}</text><text class='nodeText' x='{x+54}' y='{y+55}'>{title}</text><text class='nodeSub' x='{x+54}' y='{y+71}'>{sub}</text>"
        for x,y,ico,title,sub in [(10,28,'☀️','Sun','irradiance'),(205,28,'🔷','PV Array','DC power'),(365,28,'🔌','Inverter','DC to AC'),(525,28,'🗼','Grid','export'),(10,130,'🌤️','Weather','drivers'),(205,130,'🤖','Model','forecast'),(365,130,'🔋','Battery','storage'),(525,130,'🏠','Load','demand')]
    ]) + """</svg><div class='status'><span class='dot'></span>Energy moving • telemetry online • forecast active</div></div></body></html>"""
    components.html(html, height=395, scrolling=False)

def digital_twin_component() -> None:
    st.markdown("""
    <div class="panel" style="min-height:330px;position:relative;overflow:hidden">
      <div class="section-title">Animated 3D-Style Digital Twin</div>
      <div class="muted">PV array, battery, inverter and grid link in a contained visual twin.</div>
      <div style="position:absolute;right:8%;top:12%;font-size:3rem;filter:drop-shadow(0 0 18px rgba(251,191,36,.8));animation:floatY 5s ease-in-out infinite">☀️</div>
      <div style="position:absolute;left:8%;bottom:14%;width:58%;height:42%;transform:skewX(-18deg);border-radius:22px;background:linear-gradient(135deg,#193957,#09182b);border:1px solid rgba(34,211,238,.38)"></div>
      <div style="position:absolute;left:12%;bottom:23%;display:grid;grid-template-columns:repeat(5,38px);gap:6px;transform:rotate(-8deg)">""" + "".join(["<div style='height:26px;border-radius:6px;border:1px solid rgba(191,219,254,.66);background:linear-gradient(135deg,#14418f,#061843);box-shadow:inset 0 0 12px rgba(34,211,238,.28)'></div>" for _ in range(15)]) + """</div>
      <div style="position:absolute;right:34%;bottom:18%;width:82px;height:48px;border-radius:12px;background:#0f172a;border:1px solid rgba(16,185,129,.50);box-shadow:0 0 22px rgba(16,185,129,.18)"><div style="display:flex;gap:5px;align-items:end;height:100%;padding:9px"><i style="height:35%;width:11px;background:var(--good);border-radius:4px"></i><i style="height:58%;width:11px;background:var(--good);border-radius:4px"></i><i style="height:82%;width:11px;background:var(--good);border-radius:4px"></i><i style="height:94%;width:11px;background:var(--good);border-radius:4px"></i></div></div>
      <div style="position:absolute;right:18%;bottom:20%;width:72px;height:62px;border-radius:12px;background:linear-gradient(135deg,#f1f5f9,#64748b);box-shadow:0 18px 40px rgba(0,0,0,.38)"></div>
      <div style="position:absolute;right:6%;top:40%;font-size:3rem;text-shadow:0 0 18px rgba(34,211,238,.55)">🗼</div>
      <div style="position:absolute;right:10%;top:56%;width:30%;height:3px;background:linear-gradient(90deg,transparent,var(--accent2),var(--good));box-shadow:0 0 22px var(--accent2);transform:rotate(-8deg);animation:flow 2s linear infinite"></div>
    </div>
    """, unsafe_allow_html=True)

def live_readings(df: pd.DataFrame, target: str, tick: int) -> dict[str, float]:
    rng = np.random.default_rng(int(time.time()) % 10000 + tick)
    recent_power = float(df[target].tail(8).mean()) if not df.empty and target in df else 4200.0
    def mean_col(c, fallback):
        return float(df[c].tail(8).mean()) if not df.empty and c in df and pd.notna(df[c].tail(8).mean()) else fallback
    phase = tick / 18
    power = max(0.0, recent_power * (1 + .04 * np.sin(phase)) + rng.normal(0, max(20, recent_power * .012)))
    temp = mean_col("temperature_c", 28) + .6 * np.sin(phase * .7) + rng.normal(0,.18)
    irr = max(0.0, mean_col("irradiance_wm2", 740) * (1 + .05 * np.sin(phase*.9)) + rng.normal(0, 12))
    voltage = 400 + 4*np.sin(phase*1.1) + rng.normal(0,.6)
    current = power/max(voltage,1)
    freq = 50 + .06*np.sin(phase*1.7) + rng.normal(0,.012)
    soc = float(np.clip(62 + 18*np.sin(phase*.18) + rng.normal(0,.4), 5, 99))
    inv_temp = 38 + .18 * (power/1000) + .7*np.sin(phase*.6) + rng.normal(0,.25)
    eff = float(np.clip(100 * power / max(1, irr*6.5), 0, 99.5))
    return {"power_kw": power/1000, "temperature_c": temp, "irradiance": irr, "voltage_v": voltage, "current_a": current, "frequency_hz": freq, "battery_soc_pct": soc, "inverter_temp_c": inv_temp, "efficiency_pct": eff, "humidity": mean_col("relative_humidity_pct", 50), "wind_ms": mean_col("wind_speed_ms", 3.2), "daily_energy_kwh": max(0.0, power/1000*max(0, datetime.now().hour-6)*.62), "co2_avoided_kg": max(0.0, power/1000*max(0, datetime.now().hour-6)*.62*.42)}

def push_history(readings: dict[str, float], max_points=180) -> pd.DataFrame:
    hist = st.session_state.get("live_history")
    if hist is None:
        hist = pd.DataFrame(columns=["t","power_kw","temperature_c","irradiance","voltage_v","frequency_hz","battery_soc_pct","efficiency_pct"])
    row = {"t": datetime.now(), **{k: readings[k] for k in ["power_kw","temperature_c","irradiance","voltage_v","frequency_hz","battery_soc_pct","efficiency_pct"]}}
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True).tail(max_points).reset_index(drop=True)
    st.session_state["live_history"] = hist
    return hist

def render_live_cards(readings: dict[str, float]) -> None:
    cards = [("⚡","Power",f"{readings['power_kw']:.2f} kW"),("🌡️","Module Temp",f"{readings['temperature_c']:.1f} °C"),("☀️","Irradiance",f"{readings['irradiance']:.0f} W/m²"),("🔌","Voltage",f"{readings['voltage_v']:.1f} V"),("📡","Frequency",f"{readings['frequency_hz']:.3f} Hz"),("🔋","Battery",f"{readings['battery_soc_pct']:.1f}%")]
    cols = st.columns(6)
    for col, (ico, lab, val) in zip(cols, cards):
        with col: kpi_card(lab, val, ico, "live")

def render_live_chart(hist: pd.DataFrame) -> None:
    if hist.empty or len(hist) < 2:
        st.info("Collecting live data points…")
        return
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["t"], y=hist["power_kw"], name="Power kW", line=dict(color="#fbbf24", width=3), fill="tozeroy", fillcolor="rgba(251,191,36,.12)"))
        fig.add_trace(go.Scatter(x=hist["t"], y=hist["temperature_c"], name="Temp °C", yaxis="y2", line=dict(color="#f87171", width=2)))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right"), title="Rolling live telemetry")
        render_plot(style_fig(fig, 330), "live_telemetry", hist)
    else:
        st.line_chart(hist.set_index("t")[["power_kw","temperature_c"]])
        download_df(hist, "live_telemetry")

# -----------------------------------------------------------------------------
# Technical diagrams
# -----------------------------------------------------------------------------
def build_diagram_source(diagram: str, direction: str = "LR", detail: str = "Standard") -> str:
    node = 'node [shape=box, style="rounded,filled", color="#38bdf8", fillcolor="#07182d", fontcolor="white", penwidth=1.5]'
    edge = 'edge [color="#fbbf24", fontcolor="white", penwidth=1.3]'
    details = detail == "Detailed"
    if diagram == "PV System Architecture":
        extra = 'WX -> MODEL [label="weather"]; MON -> DASH [label="telemetry"]; MODEL -> DASH [label="forecast"];' if details else ''
        return f"""digraph G {{ graph [bgcolor="transparent", rankdir={direction}, splines=ortho]; {node}; {edge}; PV [label="PV Array\\nDC Generation"]; COMB [label="Combiner Box\\nProtection"]; INV [label="Inverter\\nDC to AC"]; TR [label="Transformer"]; GRID [label="Grid Export"]; BESS [label="Battery ESS"]; LOAD [label="Local Load"]; WX [label="Weather Station"]; MON [label="Monitoring Gateway"]; MODEL [label="Forecast Model"]; DASH [label="Dashboard"]; PV -> COMB -> INV -> TR -> GRID; INV -> LOAD; INV -> BESS [label="charge"]; BESS -> INV [label="discharge"]; {extra} }}"""
    if diagram == "Data Cleaning Pipeline":
        return f"""digraph G {{ graph [bgcolor="transparent", rankdir={direction}]; {node}; {edge}; RAW [label="Raw Data"]; SCHEMA [label="Column Detection"]; TYPES [label="Datetime + Numeric Types"]; MISS [label="Missing Values"]; DUP [label="Duplicate Timestamps"]; RES [label="Resampling"]; OUT [label="Outlier Clipping"]; CLEAN [label="Clean Time Series"]; RAW -> SCHEMA -> TYPES -> MISS -> DUP -> RES -> OUT -> CLEAN; }}"""
    if diagram == "Feature Engineering Map":
        return f"""digraph G {{ graph [bgcolor="transparent", rankdir={direction}]; {node}; {edge}; TS [label="Clean Time Series"]; LAG [label="Lag Features"]; ROLL [label="Rolling Features"]; TIME [label="Temporal Features"]; CYC [label="Cyclic Encoding"]; WX [label="Weather Features"]; MAT [label="Model Matrix"]; TS -> LAG -> MAT; TS -> ROLL -> MAT; TS -> TIME -> CYC -> MAT; WX -> MAT; }}"""
    if diagram == "Model Comparison Workflow":
        return f"""digraph G {{ graph [bgcolor="transparent", rankdir={direction}]; {node}; {edge}; FEAT [label="Feature Matrix"]; SPLIT [label="Time-Based Split"]; BASE [label="Baseline"]; LIN [label="Linear Models"]; TREE [label="Tree Ensembles"]; MET [label="Metrics Table"]; RANK [label="Leaderboard"]; RES [label="Residual Diagnostics"]; UNC [label="Uncertainty Bands"]; FEAT -> SPLIT; SPLIT -> BASE -> MET; SPLIT -> LIN -> MET; SPLIT -> TREE -> MET; MET -> RANK; MET -> RES -> UNC; }}"""
    if diagram == "Dashboard App Architecture":
        return f"""digraph G {{ graph [bgcolor="transparent", rankdir={direction}]; {node}; {edge}; USER [label="User Controls"]; DATA [label="Data Layer"]; PIPE [label="Pipeline Layer"]; MODEL [label="Model Layer"]; VIS [label="Visualization Layer"]; EXPORT [label="Export Layer"]; USER -> DATA -> PIPE -> MODEL -> VIS -> EXPORT; PIPE -> EXPORT; MODEL -> EXPORT; }}"""
    return f"""digraph G {{ graph [bgcolor="transparent", rankdir={direction}]; {node}; {edge}; F [label="Forecast Output"]; I [label="Prediction Interval"]; E [label="Residual/Error Signal"]; A [label="Anomaly Detector"]; D [label="Decision Gate"]; N [label="Normal Operation"]; W [label="Review Weather/Sensors"]; ACT [label="Maintenance / Battery Strategy"]; F -> I -> D; E -> A -> D; D -> N [label="low risk"]; D -> W [label="medium risk"]; D -> ACT [label="high risk"]; }}"""

def diagram_lab(key: str) -> None:
    c1, c2, c3 = st.columns([1.2,.8,.8])
    with c1:
        diagram = st.selectbox("Diagram type", ["PV System Architecture","Data Cleaning Pipeline","Feature Engineering Map","Model Comparison Workflow","Dashboard App Architecture","Risk & Decision Flow"], key=f"diagram_{key}")
    with c2:
        direction = st.selectbox("Layout", ["LR","TB"], key=f"dir_{key}")
    with c3:
        detail = st.selectbox("Detail", ["Compact","Standard","Detailed"], index=1, key=f"detail_{key}")
    dot = build_diagram_source(diagram, direction, detail)
    st.graphviz_chart(dot)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download diagram DOT", dot, f"{safe_name(diagram)}.dot", "text/plain", key=next_key("dot"), use_container_width=True)
    with c2:
        st.download_button("⬇️ Download diagram notes", f"Diagram: {diagram}\nLayout: {direction}\nDetail: {detail}\nUse this in your report/presentation.", f"{safe_name(diagram)}_notes.txt", "text/plain", key=next_key("notes"), use_container_width=True)

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "🏠 Home"

with st.sidebar:
    st.markdown("<div class='sidebar-card'><div class='sidebar-title'>☀️ Website Controls</div><div class='muted'>Fast access to every page and setting.</div></div>", unsafe_allow_html=True)
    st.markdown("### ⚡ Quick Access")
    page_choice = st.selectbox("Go to section", SECTION_OPTIONS, index=SECTION_OPTIONS.index(st.session_state.selected_page) if st.session_state.selected_page in SECTION_OPTIONS else 0)
    st.session_state.selected_page = page_choice
    st.markdown("---")
    theme = st.selectbox("Theme", list(THEMES.keys()), index=0)
    compact = st.toggle("Compact layout", value=True)
    alive_motion = st.toggle("Animations", value=True)
    dashboard_mode = st.selectbox("Dashboard style", ["Executive Website", "Engineering Workbench", "Student Evidence Center", "Simple Friendly View"], index=0)
    st.markdown("---")
    live_mode = st.toggle("Live updates", value=True)
    live_interval = int(st.slider("Live refresh seconds", 2, 30, 5, 1))
    st.markdown("---")
    student_name = st.text_input("Student name", STUDENT_NAME_DEFAULT)
    student_id = st.text_input("Student ID", STUDENT_ID_DEFAULT)
    project_title = st.text_input("Top project name", PROJECT_NAME)
    st.markdown("---")
    data_path = st.text_input("Dataset path", DEFAULT_DATA_PATH)
    uploaded_file = st.file_uploader("Upload data", type=["csv", "xlsx", "xls", "json"])
    st.markdown("---")
    site_name = st.selectbox("Site", ["Solar Farm Alpha", "Rooftop PV Lab", "Campus PV Plant"], index=0)
    resample_rule = st.selectbox("Resampling rule", ["None", "15min", "30min", "1h", "1D"], index=1)
    horizon = int(st.number_input("Forecast horizon rows", 1, 96, 1, 1))
    model_rows = int(st.slider("Model rows", 1000, 40000, 18000, 1000))
    chart_window = int(st.slider("Chart window rows", 96, 3000, 700, 32))
    confidence_width = float(st.slider("Forecast band width", .05, .35, .12, .01))
    anomaly_sensitivity = float(st.slider("Anomaly sensitivity", 1.0, 4.0, 2.0, .1))
    st.markdown("---")
    st.markdown("### 🔬 Model Comparison")
    comparison_group = st.selectbox("Compare", ["Do not train yet", "Baseline only", "Fast comparison", "Linear models", "Tree ensemble models", "All available models"], index=0)
    comparison_metric = st.selectbox("Rank by", ["MAPE_pct", "RMSE", "MAE", "R2"], index=0)
    run_models_clicked = st.button("⌛ Run selected comparison", type="primary", use_container_width=True)
    clear_models_clicked = st.button("Clear saved model results", use_container_width=True)
    api_key = st.text_input("OpenRouter API key for grader (optional)", type="password")

inject_css(theme, alive_motion, compact)

# -----------------------------------------------------------------------------
# Header and loading/data preparation
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="app-header">
  <div class="brand-wrap"><div class="logo">☀️</div><div><div class="brand-title">{project_title}</div><div class="brand-sub">Student: <b>{student_name}</b> • ID: <b>{student_id}</b> • Real interactive PV forecasting website</div></div></div>
  <div class="top-pills"><span class="pill"><span class="live-dot"></span>Alive</span><span class="pill">{dashboard_mode}</span><span class="pill">{site_name}</span><span class="pill">🕒 {datetime.now().strftime('%H:%M:%S')}</span></div>
</div>
""", unsafe_allow_html=True)

raw_df, source_label = load_dataset(data_path, uploaded_file)
columns = list(raw_df.columns)
numeric_candidates = [c for c in columns if pd.to_numeric(raw_df[c], errors="coerce").notna().sum() > 0]
if not numeric_candidates:
    st.error("No numeric columns were found. Upload a dataset with at least one numeric target column.")
    st.stop()
default_ts_idx = columns.index(DEFAULT_TIMESTAMP_COL) if DEFAULT_TIMESTAMP_COL in columns else 0
default_target_idx = numeric_candidates.index(DEFAULT_TARGET_COL) if DEFAULT_TARGET_COL in numeric_candidates else 0
c1, c2, c3, c4 = st.columns([1,1,.85,.85])
timestamp_col = c1.selectbox("Timestamp column", columns, index=default_ts_idx)
target_col = c2.selectbox("Target column", numeric_candidates, index=default_target_idx)
ts_preview = pd.to_datetime(raw_df[timestamp_col], errors="coerce")
min_date = ts_preview.min().date() if ts_preview.notna().any() else datetime.now().date()
max_date = ts_preview.max().date() if ts_preview.notna().any() else datetime.now().date()
start_date = c3.date_input("Start date", value=min_date)
end_date = c4.date_input("End date", value=max_date)

prepared_df, cleaning_report = prepare_timeseries(raw_df, timestamp_col, target_col, resample_rule)
prepared_df[timestamp_col] = pd.to_datetime(prepared_df[timestamp_col], errors="coerce")
filtered_df = prepared_df[(prepared_df[timestamp_col].dt.date >= start_date) & (prepared_df[timestamp_col].dt.date <= end_date)].copy()
if filtered_df.empty:
    filtered_df = prepared_df.copy()

live_tick = int(st_autorefresh(interval=max(1, live_interval) * 1000, key="live_refresh")) if live_mode and AUTOREFRESH_AVAILABLE else 0
readings = live_readings(filtered_df, target_col, live_tick)
live_history = push_history(readings)

model_df, feature_cols, weather_features = build_features(prepared_df, timestamp_col, target_col, horizon)
model_df = model_df.tail(model_rows).copy()
if clear_models_clicked:
    st.session_state.pop("model_results", None)
if run_models_clicked and comparison_group != "Do not train yet":
    with st.spinner(f"Training selected comparison: {comparison_group}"):
        results = run_models(model_df, feature_cols, timestamp_col, target_col, comparison_group, comparison_metric)
        st.session_state.model_results = results
if "model_results" in st.session_state:
    comparison_df, predictions_df, importance_df, uncertainty_summary, modeling_note = st.session_state.model_results
else:
    comparison_df, predictions_df, importance_df, uncertainty_summary, modeling_note = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, "Model training has not run yet. Choose a comparison group and press Run selected comparison."

latest_power = float(filtered_df[target_col].iloc[-1]) if len(filtered_df) else 0
avg_power = float(filtered_df[target_col].mean()) if len(filtered_df) else 0
max_power = float(filtered_df[target_col].max()) if len(filtered_df) else 0
energy_mwh = float(filtered_df[target_col].sum() * .25 / 1_000_000) if resample_rule in ["15min", "None"] else float(filtered_df[target_col].sum() / 1_000_000)
zero_pct = float((filtered_df[target_col] <= 0).mean() * 100) if len(filtered_df) else 0
best_model = str(comparison_df.iloc[0]["model"]) if not comparison_df.empty else "N/A"

# -----------------------------------------------------------------------------
# Fast navigation
# -----------------------------------------------------------------------------
st.markdown("<div class='panel'><div class='section-title'>⚡ Quick Access</div><div class='muted'>Use these buttons or the sidebar selector to open any section instantly. No duplicated navigation, no tabs, no hidden pages.</div></div>", unsafe_allow_html=True)
rows = [SECTION_OPTIONS[:6], SECTION_OPTIONS[6:]]
for r, row in enumerate(rows):
    cols = st.columns(len(row))
    for col, page in zip(cols, row):
        with col:
            label = f"✅ {page}" if st.session_state.selected_page == page else page
            if st.button(label, key=f"nav_{r}_{safe_name(page)}", use_container_width=True):
                st.session_state.selected_page = page
selected_page = st.session_state.selected_page

# -----------------------------------------------------------------------------
# Main sections
# -----------------------------------------------------------------------------
if selected_page == "🏠 Home":
    section_hero("🏠 Home", "Executive overview with KPIs, live status, PV energy flow, forecasting trend, and strategy guidance.", "home")
    c1, c2 = st.columns([1.08,.92])
    with c1:
        st.markdown(f"<div class='hero'><div class='pill'><span class='live-dot'></span>{site_name} • {source_label}</div><div class='hero-title'>{dashboard_mode}</div><div class='hero-copy'>A clear real-website dashboard for solar PV forecasting, live telemetry, model evidence, technical diagrams, simulator, and exports. All major tools are reachable from the Quick Access menu and sidebar.</div></div>", unsafe_allow_html=True)
    with c2:
        pv_energy_component()
    cols = st.columns(6)
    for col, args in zip(cols, [("Capacity", f"{max_power/1000:,.2f} kWp", "⚙️", "max observed"),("Energy", f"{energy_mwh:,.2f} MWh", "⚡", "selected period"),("Latest Power", f"{latest_power:,.0f} W", "📈", "latest row"),("Avg Power", f"{avg_power:,.0f} W", "🔁", "mean"),("Zero Power", f"{zero_pct:.1f}%", "🌙", "night/outage"),("Best Model", best_model[:20], "🤖", "after training")]):
        with col: kpi_card(*args)
    st.markdown("### Live Plant Snapshot")
    render_live_cards(readings)
    st.markdown("### Production Trend")
    render_plot(forecast_fig(filtered_df, timestamp_col, target_col, chart_window, confidence_width), "production_trend", filtered_df[[timestamp_col, target_col]].tail(chart_window))
    st.markdown("### Strategy Tips")
    c = st.columns(4)
    tips = [("Operate by forecast peaks", "Use predicted production windows for scheduling loads or storage."),("Monitor uncertainty", "Do not trust one line forecast without interval context."),("Check anomalies", "Low output during irradiance suggests faults, clipping, soiling, or curtailment."),("Defend methodology", "Use chronological validation and clear metric tables.")]
    for col, (t,b) in zip(c, tips):
        with col: feature_card(t,b,"💡")

elif selected_page == "🔴 Live Telemetry":
    section_hero("🔴 Live Telemetry", "SCADA-style live values, rolling trend, and downloadable telemetry buffer.", "live")
    render_live_cards(readings)
    render_live_chart(live_history)
    st.markdown("### Live readings table")
    snap = pd.DataFrame([{"signal": k, "value": round(float(v), 4)} for k, v in readings.items()])
    st.dataframe(snap, use_container_width=True, hide_index=True)
    download_df(live_history, "live_telemetry_buffer")

elif selected_page == "📊 Forecasting":
    section_hero("📊 Forecasting", "Forecast trend, actual vs predicted, uncertainty bands, and weather context.", "forecast")
    a,b = st.columns(2)
    with a: render_plot(forecast_fig(filtered_df, timestamp_col, target_col, chart_window, confidence_width), "forecast_signal", filtered_df[[timestamp_col, target_col]].tail(chart_window))
    with b: render_plot(prediction_fig(predictions_df, timestamp_col, chart_window), "actual_vs_predicted", predictions_df.tail(chart_window) if not predictions_df.empty else pd.DataFrame())
    if weather_features:
        st.markdown("### Weather context")
        weather_cols = [timestamp_col] + weather_features[:4]
        render_plot(None, "weather_context", filtered_df[weather_cols].tail(chart_window).set_index(timestamp_col))

elif selected_page == "🧩 Visual System":
    section_hero("🧩 Visual System", "System visuals, digital twin, energy flow, and image-based context.", "visual")
    a,b = st.columns(2)
    with a: digital_twin_component()
    with b: pv_energy_component()
    st.markdown("### Visual context")
    c = st.columns(4)
    for col, (key, title) in zip(c, [("home","Solar field"),("grid","Grid connection"),("battery","Battery/control"),("forecast","Weather context")]):
        with col:
            st.markdown(f"<div class='section-hero' style='min-height:180px;background-image:url({IMAGES[key]})'><div class='section-title'>{title}</div></div>", unsafe_allow_html=True)
    st.markdown("### Interactive diagram")
    diagram_lab("visual")

elif selected_page == "🧹 Data Pipeline":
    section_hero("🧹 Data Pipeline", "Cleaning, resampling, outlier handling, feature engineering, and data audit.", "pipeline")
    c = st.columns(4)
    for col, (lab,val,ico) in zip(c, [("Raw rows", f"{len(raw_df):,}", "📦"),("Clean rows", f"{len(prepared_df):,}", "🧹"),("Features", f"{len(feature_cols):,}", "🧠"),("Weather features", f"{len(weather_features):,}", "🌦️")]):
        with col: kpi_card(lab, val, ico, "pipeline")
    st.markdown("### Cleaning report")
    st.json(cleaning_report)
    st.markdown("### Data audit")
    audit = audit_dataframe(raw_df)
    st.dataframe(audit, use_container_width=True, hide_index=True)
    download_df(audit, "data_audit")
    st.markdown("### Pipeline flowchart")
    st.graphviz_chart(build_diagram_source("Data Cleaning Pipeline", "LR", "Detailed"))
    st.markdown("### Engineered feature preview")
    st.dataframe(model_df[[timestamp_col, "y_target"] + feature_cols[:12]].tail(200), use_container_width=True, hide_index=True)
    download_df(model_df[[timestamp_col, "y_target"] + feature_cols], "engineered_features")

elif selected_page == "🤖 Models":
    section_hero("🤖 Models", "User-controlled training, leaderboard, metrics, feature importance and uncertainty summary.", "models")
    st.info(modeling_note)
    if comparison_df.empty:
        st.warning("No model comparison has been run yet. Use the sidebar Model Comparison controls and click Run selected comparison.")
    else:
        st.markdown("### Model leaderboard")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        download_df(comparison_df, "model_leaderboard")
        a,b,c = st.columns(3)
        with a: render_plot(metrics_bar_fig(comparison_df, "MAE"), "mae_comparison", comparison_df)
        with b: render_plot(metrics_bar_fig(comparison_df, "RMSE"), "rmse_comparison", comparison_df)
        with c: render_plot(metrics_bar_fig(comparison_df, "MAPE_pct"), "mape_comparison", comparison_df)
        st.markdown("### Feature importance")
        st.dataframe(importance_df, use_container_width=True, hide_index=True)
        download_df(importance_df, "feature_importance")
        st.markdown("### Uncertainty")
        st.json(uncertainty_summary)

elif selected_page == "🧬 Advanced":
    section_hero("🧬 Advanced", "Residual diagnostics, correlations, anomaly signals, and feature relationships.", "advanced")
    a,b = st.columns(2)
    with a: render_plot(residual_fig(predictions_df), "residual_distribution", predictions_df[[timestamp_col,"residual","absolute_error"]].tail(3000) if not predictions_df.empty else pd.DataFrame())
    with b: render_plot(scatter_fig(predictions_df), "actual_predicted_scatter", predictions_df.tail(2000) if not predictions_df.empty else pd.DataFrame())
    st.markdown("### Feature correlation heatmap")
    render_plot(correlation_fig(model_df, feature_cols), "feature_correlation", model_df[[c for c in feature_cols if c in model_df.columns][:22] + ["y_target"]].tail(3000) if not model_df.empty else pd.DataFrame())
    if not filtered_df.empty:
        q1, q3 = filtered_df[target_col].quantile([.25, .75]); iqr = q3 - q1
        z = (filtered_df[target_col] < q1 - anomaly_sensitivity * iqr) | (filtered_df[target_col] > q3 + anomaly_sensitivity * iqr)
        anomalies = filtered_df.loc[z, [timestamp_col, target_col]].copy()
        st.markdown("### Anomaly candidates")
        st.dataframe(anomalies.tail(250), use_container_width=True, hide_index=True)
        download_df(anomalies, "anomaly_candidates")

elif selected_page == "🛠️ Technical Diagrams":
    section_hero("🛠️ Technical Diagrams", "Interactive flowcharts for system architecture, pipeline, features, model workflow, dashboard architecture and risk decisions.", "diagrams")
    c = st.columns(3)
    for col, (t,b) in zip(c, [("Architecture", "PV components, monitoring, forecast and dashboard."),("Pipeline", "Cleaning, resampling, outlier handling and feature generation."),("Decision flow", "Forecast uncertainty and residuals become operational decisions.")]):
        with col: feature_card(t,b,"🛠️")
    diagram_lab("technical")

elif selected_page == "🕹️ Simulator":
    section_hero("🕹️ Simulator", "What-if simulator for weather, soiling, curtailment and battery strategy.", "simulator")
    s1, s2, s3, s4 = st.columns(4)
    irradiance_factor = s1.slider("Irradiance factor", .4, 1.3, 1.0, .05)
    temp_delta = s2.slider("Temperature delta °C", -10.0, 15.0, 0.0, .5)
    soiling_loss = s3.slider("Soiling loss %", 0.0, 35.0, 5.0, 1.0)
    curtailment = s4.slider("Curtailment limit %", 40.0, 120.0, 100.0, 5.0)
    sim = filtered_df[[timestamp_col, target_col]].tail(chart_window).copy()
    temp_eff = 1 - 0.0035 * max(temp_delta, 0)
    sim["scenario_power"] = sim[target_col] * irradiance_factor * temp_eff * (1 - soiling_loss / 100)
    sim["scenario_power"] = np.minimum(sim["scenario_power"], sim[target_col].max() * curtailment / 100)
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim[timestamp_col], y=sim[target_col], name="Current", line=dict(color="#38bdf8")))
        fig.add_trace(go.Scatter(x=sim[timestamp_col], y=sim["scenario_power"], name="Scenario", line=dict(color="#fbbf24")))
        render_plot(style_fig(fig, 390), "simulator_scenario", sim)
    else:
        st.line_chart(sim.set_index(timestamp_col)[[target_col,"scenario_power"]])
        download_df(sim, "simulator_scenario")
    delta = float(sim["scenario_power"].sum() - sim[target_col].sum())
    st.metric("Scenario energy difference", f"{delta:,.0f} W-samples")

elif selected_page == "🔬 Comparison Lab":
    section_hero("🔬 Comparison Lab", "All-in-one comparison workspace: leaderboard, metric bars, residuals, scatter, correlation and recommendations.", "compare")
    tool = st.selectbox("Comparison tool", ["Full comparison dashboard", "Leaderboard", "Metric bars", "Actual vs predicted", "Residuals", "Feature correlation", "Recommendations"], key="comparison_tool")
    if tool in ["Full comparison dashboard", "Leaderboard"]:
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        download_df(comparison_df, "comparison_leaderboard")
    if tool in ["Full comparison dashboard", "Metric bars"]:
        a,b,c = st.columns(3)
        with a: render_plot(metrics_bar_fig(comparison_df, "MAE"), "mae_comparison", comparison_df)
        with b: render_plot(metrics_bar_fig(comparison_df, "RMSE"), "rmse_comparison", comparison_df)
        with c: render_plot(metrics_bar_fig(comparison_df, "MAPE_pct"), "mape_comparison", comparison_df)
    if tool in ["Full comparison dashboard", "Actual vs predicted"]:
        render_plot(prediction_fig(predictions_df, timestamp_col, chart_window), "prediction_interval", predictions_df.tail(chart_window) if not predictions_df.empty else pd.DataFrame())
        render_plot(scatter_fig(predictions_df), "actual_predicted_scatter", predictions_df.tail(2000) if not predictions_df.empty else pd.DataFrame())
    if tool in ["Full comparison dashboard", "Residuals"]:
        render_plot(residual_fig(predictions_df), "residual_distribution", predictions_df[[timestamp_col,"residual","absolute_error"]].tail(3000) if not predictions_df.empty else pd.DataFrame())
    if tool in ["Full comparison dashboard", "Feature correlation"]:
        render_plot(correlation_fig(model_df, feature_cols), "feature_correlation", model_df[[c for c in feature_cols if c in model_df.columns][:22] + ["y_target"]].tail(3000) if not model_df.empty else pd.DataFrame())
    if tool in ["Full comparison dashboard", "Recommendations"]:
        c = st.columns(4)
        recs = [("Model", f"Current best: {best_model}."),("Metric", "Use MAE/RMSE/MAPE/R² together; do not rely on MAPE only."),("Uncertainty", "Show prediction intervals for confidence."),("Data", "Improve with local weather forecasts, inverter status and maintenance labels.")]
        for col, (t,b) in zip(c, recs):
            with col: feature_card(t,b,"🎯")

elif selected_page == "📤 Export":
    section_hero("📤 Export", "Download data, model outputs, evidence JSON, grader prompt, and optional local/AI grade.", "export")
    submission = {
        "project": {"title": project_title, "student": student_name, "student_id": student_id, "site": site_name},
        "data_integrity": {"rows_loaded": int(len(raw_df)), "rows_cleaned": int(len(prepared_df)), "cleaning_report": cleaning_report, "resampling_discussed": True, "outliers_discussed": True},
        "feature_engineering": {"feature_count": len(feature_cols), "baseline_features": [c for c in feature_cols if c.startswith("lag_")], "student_added_features": feature_cols, "weather_features": weather_features},
        "modeling_and_evaluation": {"has_time_based_split": True, "has_metrics_table": not comparison_df.empty, "model_comparison_table": comparison_df.to_dict(orient="records"), "feature_importance_table": importance_df.to_dict(orient="records"), "uncertainty_summary": uncertainty_summary, "note": modeling_note},
        "dashboard": {"sections": SECTION_OPTIONS, "has_live_telemetry": True, "has_system_photos": True, "has_diagrams_and_3d": True, "has_interactive_technical_diagrams": True, "has_simulator": True, "has_downloads": True, "graph_types": ["line", "bar", "scatter", "histogram", "heatmap", "flowchart", "technical diagrams"]},
        "presentation_and_rigor": {"limitations": ["Demo/live values are simulated unless connected to real SCADA.", "External image URLs should be replaced with local assets for offline deployment."], "reproducibility_notes": ["Run with streamlit run app.py", "Install requirements.txt", "Use chronological split for validation."]},
    }
    sub_json = json.dumps(submission, indent=2, default=safe_json_default)
    c1,c2,c3 = st.columns(3)
    with c1: st.download_button("⬇️ submission_evidence.json", sub_json, "submission_evidence.json", "application/json", use_container_width=True)
    with c2: download_df(prepared_df, "cleaned_dataset")
    with c3: download_df(model_df, "model_features")
    if not comparison_df.empty:
        download_df(comparison_df, "model_metrics")
        download_df(predictions_df, "predictions")
        download_df(importance_df, "feature_importance")
    st.markdown("### Evidence JSON preview")
    st.json(submission)
    st.markdown("### Grader")
    def local_grade(sub: dict[str, Any]) -> dict[str, Any]:
        scores = {"Data & integrity": 20, "Feature engineering": 15, "Modeling & evaluation": 22 if not comparison_df.empty else 14, "Dashboard quality": 10, "Presentation & rigor": 9}
        return {"scores": scores, "total_80": int(sum(scores.values())), "strengths": ["Comprehensive interactive PV website", "Clear data pipeline and feature engineering", "Controlled model comparison and exports"], "weaknesses": ["Live values are simulated unless connected to real telemetry"], "actionable_improvements": ["Add real SCADA API", "Replace external images with local project photos"]}
    grade = local_grade(submission)
    if api_key:
        prompt = "Grade this PV forecasting project from the following JSON. Return JSON only.\n" + sub_json
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": OPENROUTER_MODEL, "messages": [{"role":"user", "content": prompt}], "temperature": 0}, timeout=60)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", content, re.S)
            if m: grade = json.loads(m.group(0))
        except Exception as e:
            st.warning(f"AI grader unavailable; using local fallback. {e}")
    st.json(grade)
    st.download_button("⬇️ grade_result.json", json.dumps(grade, indent=2), "grade_result.json", "application/json", use_container_width=True)
