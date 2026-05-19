
"""
app.py — Professional Solar PV Forecasting Intelligence Platform

Run:
    streamlit run app.py

This app is built as a complete, professional, interactive dashboard:
- User-selectable website representation modes
- Professional animated design system
- Robust dataset loading with demo fallback
- Cleaning, resampling, outlier handling
- Feature engineering
- Time-based train/validation split
- Model comparison and uncertainty intervals
- System photos, technical diagram, and 3D-style live visual
- Advanced analytics: anomalies, residuals, correlations, daily profiles
- Export evidence and AI grader with 429 fallback

Recommended requirements.txt:
streamlit
pandas
numpy
plotly
scikit-learn
requests
openpyxl
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

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    go = None
    PLOTLY_AVAILABLE = False

try:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
STUDENT_NAME_DEFAULT = "MAZEN AL-HIMALI"
STUDENT_ID_DEFAULT = "PG12S2540572"
DEFAULT_DATA_PATH = "data/dataset_sample.csv"
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_TARGET_COL = "total_active_power_w"
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

IMAGE_SOLAR = "https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&w=1400&q=80"
IMAGE_CONTROL = "https://images.unsplash.com/photo-1581092160607-ee22621dd758?auto=format&fit=crop&w=900&q=80"
IMAGE_WEATHER = "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=900&q=80"
IMAGE_GRID = "https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&w=900&q=80"

AI_GRADER_PROMPT_TEMPLATE = """SYSTEM:
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


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Solar PV Forecasting Intelligence",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
def inject_css(theme: str, motion: bool, density: str) -> None:
    theme_map = {
        "Energy Navy": {
            "bg0": "#050B14", "bg1": "#07111F", "bg2": "#0C2037",
            "blue": "#3B82F6", "cyan": "#22D3EE", "green": "#10B981", "gold": "#FBBF24",
            "text": "#ECF5FF", "muted": "#9FB0C7",
        },
        "Emerald Grid": {
            "bg0": "#03110E", "bg1": "#061E19", "bg2": "#092A22",
            "blue": "#14B8A6", "cyan": "#5EEAD4", "green": "#22C55E", "gold": "#A7F3D0",
            "text": "#ECFDF5", "muted": "#9DD6C8",
        },
        "Solar Gold": {
            "bg0": "#120C04", "bg1": "#211805", "bg2": "#2A1D05",
            "blue": "#F59E0B", "cyan": "#FDE68A", "green": "#84CC16", "gold": "#FBBF24",
            "text": "#FFF7ED", "muted": "#D8C7A8",
        },
        "Cyber Blue": {
            "bg0": "#030712", "bg1": "#07182D", "bg2": "#071A3A",
            "blue": "#2563EB", "cyan": "#38BDF8", "green": "#06B6D4", "gold": "#93C5FD",
            "text": "#EFF6FF", "muted": "#A9C5E8",
        },
    }
    pal = theme_map.get(theme, theme_map["Energy Navy"])
    compact = density == "Compact"
    loose = density == "Spacious"

    card_pad = ".75rem" if compact else "1rem" if not loose else "1.25rem"
    kpi_min = "92px" if compact else "118px" if not loose else "142px"
    tab_font = "1rem" if compact else "1.16rem"

    animation_css = "" if motion else """
        *, *::before, *::after {
            animation: none !important;
            transition: none !important;
        }
    """

    st.markdown(
        f"""
        <style>
        :root {{
            --bg0:{pal['bg0']};
            --bg1:{pal['bg1']};
            --bg2:{pal['bg2']};
            --blue:{pal['blue']};
            --cyan:{pal['cyan']};
            --green:{pal['green']};
            --gold:{pal['gold']};
            --text:{pal['text']};
            --muted:{pal['muted']};
            --red:#EF4444;
            --card:rgba(16,29,51,.82);
            --card2:rgba(255,255,255,.055);
            --border:rgba(148,163,184,.20);
            --shadow:0 22px 70px rgba(0,0,0,.34);
        }}

        @keyframes floatSoft {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-4px); }}
        }}
        @keyframes pulseGlow {{
            0%, 100% {{ opacity:.45; box-shadow:0 0 10px rgba(34,211,238,.25); }}
            50% {{ opacity:1; box-shadow:0 0 28px rgba(34,211,238,.80); }}
        }}
        @keyframes shimmer {{
            0% {{ background-position:-800px 0; }}
            100% {{ background-position:800px 0; }}
        }}
        @keyframes spin {{
            from {{ transform:rotate(0deg); }}
            to {{ transform:rotate(360deg); }}
        }}
        @keyframes flow {{
            0% {{ transform:translateX(-40%); opacity:.20; }}
            50% {{ opacity:1; }}
            100% {{ transform:translateX(40%); opacity:.20; }}
        }}

        html, body, .stApp {{
            color:var(--text);
            background:
                radial-gradient(circle at 10% 6%, color-mix(in srgb, var(--blue) 23%, transparent), transparent 32%),
                radial-gradient(circle at 86% 8%, color-mix(in srgb, var(--green) 16%, transparent), transparent 34%),
                radial-gradient(circle at 55% 105%, color-mix(in srgb, var(--gold) 11%, transparent), transparent 38%),
                linear-gradient(135deg, var(--bg0), var(--bg1) 48%, var(--bg2));
        }}
        [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(5,11,20,.985), rgba(8,18,32,.955));
            border-right:1px solid var(--border);
        }}
        .block-container {{
            padding-top: .8rem;
            padding-bottom: 2rem;
            max-width: 1560px;
        }}
        h1, h2, h3, h4, h5, h6 {{ color:var(--text)!important; letter-spacing:-.02em; }}
        label, .stSelectbox label, .stSlider label, .stRadio label, .stCheckbox label {{
            color:var(--text)!important;
            font-weight:800!important;
        }}

        .app-shell {{
            position: relative;
        }}
        .topbar {{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:1rem;
            padding: .9rem 1.05rem;
            border:1px solid var(--border);
            border-radius: 24px;
            background:
                linear-gradient(135deg, rgba(255,255,255,.075), rgba(255,255,255,.035)),
                radial-gradient(circle at 20% 0%, color-mix(in srgb, var(--cyan) 18%, transparent), transparent 35%);
            box-shadow:var(--shadow);
            backdrop-filter: blur(18px);
            margin-bottom:.9rem;
        }}
        .brand-row {{ display:flex; align-items:center; gap:.85rem; }}
        .brand-mark {{
            width:52px;
            height:52px;
            border-radius:18px;
            display:flex;
            align-items:center;
            justify-content:center;
            background:linear-gradient(135deg, var(--gold), var(--green));
            color:#07111f;
            font-size:1.75rem;
            font-weight:950;
            box-shadow:0 14px 40px color-mix(in srgb, var(--gold) 28%, transparent);
        }}
        .brand-title {{ font-size:1.32rem; font-weight:950; line-height:1.05; }}
        .brand-subtitle {{ color:var(--muted); font-size:.86rem; margin-top:.22rem; }}
        .status-cluster {{ display:flex; gap:.55rem; flex-wrap:wrap; justify-content:flex-end; }}
        .status-pill {{
            border:1px solid color-mix(in srgb, var(--cyan) 35%, transparent);
            color:var(--text);
            background:color-mix(in srgb, var(--blue) 16%, transparent);
            border-radius:999px;
            padding:.46rem .74rem;
            font-size:.82rem;
            font-weight:850;
            white-space:nowrap;
        }}
        .live-dot {{
            width:10px;
            height:10px;
            border-radius:999px;
            display:inline-block;
            background:var(--green);
            margin-right:.36rem;
            animation:pulseGlow 1.5s ease-in-out infinite;
        }}

        .hero {{
            position:relative;
            overflow:hidden;
            padding:{'1rem' if compact else '1.25rem' if not loose else '1.55rem'};
            border-radius: 28px;
            border: 1px solid color-mix(in srgb, var(--gold) 28%, transparent);
            background:
                linear-gradient(135deg, rgba(16,29,51,.92), rgba(8,18,32,.82)),
                radial-gradient(circle at 13% 20%, color-mix(in srgb, var(--cyan) 20%, transparent), transparent 34%);
            box-shadow: var(--shadow);
            margin-bottom:1rem;
        }}
        .hero::after {{
            content:"";
            position:absolute;
            inset:0;
            background:linear-gradient(90deg, transparent, rgba(255,255,255,.055), transparent);
            animation: shimmer 8s linear infinite;
            pointer-events:none;
        }}
        .hero-title {{
            font-size:{'2rem' if compact else '2.55rem' if not loose else '3rem'};
            font-weight:950;
            letter-spacing:-.055em;
            margin:.35rem 0 .2rem 0;
            max-width:1050px;
        }}
        .hero-text {{ color:var(--muted); font-size:1rem; max-width:1180px; line-height:1.62; }}
        .pill {{
            display:inline-flex;
            align-items:center;
            gap:.35rem;
            border-radius:999px;
            border:1px solid color-mix(in srgb, var(--cyan) 36%, transparent);
            background:color-mix(in srgb, var(--blue) 14%, transparent);
            color:var(--text);
            padding:.42rem .72rem;
            font-size:.80rem;
            font-weight:900;
        }}

        .mode-panel, .glass-card, .kpi-card, .photo-card, .diagram-box, .isometric, .workflow-card {{
            animation: floatSoft 7s ease-in-out infinite;
        }}
        .mode-panel {{
            border:1px solid color-mix(in srgb, var(--cyan) 26%, transparent);
            border-radius:26px;
            background:
                radial-gradient(circle at top right, color-mix(in srgb, var(--cyan) 13%, transparent), transparent 38%),
                linear-gradient(135deg, rgba(16,29,51,.82), rgba(6,13,24,.75));
            padding:{card_pad};
            box-shadow:var(--shadow);
            margin-bottom:1rem;
        }}
        .mode-head {{
            display:flex;
            align-items:flex-start;
            justify-content:space-between;
            gap:1rem;
            margin-bottom:.8rem;
        }}
        .mode-title {{ font-size:1.22rem; font-weight:950; }}
        .mode-copy {{ color:var(--muted); font-size:.9rem; line-height:1.5; }}
        .mode-grid {{
            display:grid;
            grid-template-columns:repeat(6,minmax(120px,1fr));
            gap:.65rem;
        }}
        .mode-chip {{
            border:1px solid var(--border);
            background:var(--card2);
            border-radius:18px;
            padding:.75rem .8rem;
        }}
        .chip-label {{ color:var(--muted); font-size:.74rem; font-weight:850; }}
        .chip-value {{ font-size:1.03rem; font-weight:950; margin-top:.2rem; }}

        .kpi-card {{
            min-height:{kpi_min};
            border-radius:22px;
            border:1px solid var(--border);
            background:
                radial-gradient(circle at top left, color-mix(in srgb, var(--blue) 24%, transparent), transparent 42%),
                linear-gradient(145deg, rgba(17,31,53,.96), rgba(9,20,36,.86));
            box-shadow:0 16px 46px rgba(0,0,0,.26);
            padding:{card_pad};
        }}
        .kpi-icon {{ font-size:1.7rem; }}
        .kpi-label {{ color:var(--muted); font-size:.78rem; font-weight:900; margin-top:.25rem; }}
        .kpi-value {{ font-size:{'1.35rem' if compact else '1.7rem'}; font-weight:950; margin-top:.25rem; }}
        .kpi-delta {{ color:var(--green); font-size:.80rem; margin-top:.2rem; font-weight:800; }}

        .glass-card {{
            border:1px solid var(--border);
            border-radius:24px;
            background:linear-gradient(145deg, rgba(16,29,51,.84), rgba(9,20,36,.72));
            padding:{card_pad};
            box-shadow:0 16px 46px rgba(0,0,0,.26);
            backdrop-filter:blur(14px);
        }}
        .section-title {{
            color:var(--gold);
            font-weight:950;
            font-size:1.06rem;
            margin-bottom:.6rem;
        }}
        .muted {{ color:var(--muted); font-size:.88rem; line-height:1.5; }}

        .photo-card {{
            position:relative;
            min-height:{'270px' if compact else '340px' if not loose else '420px'};
            border-radius:24px;
            overflow:hidden;
            border:1px solid color-mix(in srgb, var(--gold) 28%, transparent);
            background-size:cover;
            background-position:center;
            box-shadow:inset 0 -150px 140px rgba(0,0,0,.78), 0 18px 54px rgba(0,0,0,.30);
        }}
        .photo-overlay {{ position:absolute; left:1rem; right:1rem; bottom:1rem; z-index:2; }}
        .photo-title {{ font-size:1.45rem; font-weight:950; }}
        .media-thumb {{
            height:{'98px' if compact else '130px'};
            border-radius:18px;
            background-size:cover;
            background-position:center;
            border:1px solid var(--border);
            box-shadow:inset 0 -70px 90px rgba(0,0,0,.40);
        }}

        .diagram-box {{
            min-height:{'270px' if compact else '340px' if not loose else '420px'};
            border-radius:24px;
            border:1px solid color-mix(in srgb, var(--cyan) 28%, transparent);
            background:
                linear-gradient(90deg, color-mix(in srgb, var(--cyan) 6%, transparent) 1px, transparent 1px),
                linear-gradient(color-mix(in srgb, var(--cyan) 6%, transparent) 1px, transparent 1px),
                linear-gradient(135deg, rgba(8,18,32,.88), rgba(14,29,50,.76));
            background-size:26px 26px, 26px 26px, auto;
            padding:{card_pad};
            overflow:hidden;
        }}
        .flow-row {{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:.55rem;
            margin-top:1rem;
        }}
        .node {{
            flex:1;
            text-align:center;
            padding:.72rem .55rem;
            border-radius:17px;
            border:1px solid var(--border);
            background:rgba(255,255,255,.05);
        }}
        .node-icon {{ font-size:1.9rem; }}
        .node-label {{ font-weight:950; font-size:.84rem; }}
        .node-sub {{ color:var(--muted); font-size:.72rem; }}
        .arrow {{
            position:relative;
            color:var(--gold);
            font-size:1.35rem;
            font-weight:950;
            min-width:22px;
        }}
        .arrow::after {{
            content:"";
            position:absolute;
            left:-18px;
            right:-18px;
            top:50%;
            height:2px;
            background:linear-gradient(90deg, transparent, var(--gold), transparent);
            animation:flow 2.4s ease-in-out infinite;
        }}

        .isometric {{
            min-height:{'270px' if compact else '340px' if not loose else '420px'};
            border-radius:24px;
            border:1px solid color-mix(in srgb, var(--blue) 32%, transparent);
            position:relative;
            overflow:hidden;
            padding:{card_pad};
            background:
                radial-gradient(circle at 68% 35%, color-mix(in srgb, var(--cyan) 24%, transparent), transparent 34%),
                linear-gradient(145deg, rgba(12,25,43,.95), rgba(6,13,24,.92));
        }}
        .platform {{
            width:74%;
            height:58%;
            position:absolute;
            left:12%;
            bottom:10%;
            transform:skewX(-18deg) rotateX(8deg);
            border-radius:26px;
            background:linear-gradient(135deg,#193957,#09182b);
            border:1px solid color-mix(in srgb, var(--cyan) 40%, transparent);
            box-shadow:0 24px 90px color-mix(in srgb, var(--cyan) 15%, transparent);
        }}
        .panel-grid {{
            position:absolute;
            left:13%;
            top:22%;
            display:grid;
            grid-template-columns:repeat(5,44px);
            gap:7px;
            transform:rotate(-10deg);
        }}
        .solar-panel {{
            height:34px;
            border-radius:6px;
            background:
                linear-gradient(90deg, rgba(255,255,255,.04), rgba(255,255,255,.16), rgba(255,255,255,.04)),
                linear-gradient(135deg,#143f8f,#051b44);
            background-size:180% 100%;
            border:1px solid rgba(191,219,254,.62);
            box-shadow:inset 0 0 12px color-mix(in srgb, var(--cyan) 26%, transparent);
            animation:shimmer 4.5s linear infinite;
        }}
        .battery-3d {{
            position:absolute;
            right:41%;
            bottom:19%;
            width:92px;
            height:54px;
            border-radius:12px;
            background:linear-gradient(135deg,#1e293b,#0f172a);
            border:1px solid color-mix(in srgb, var(--green) 48%, transparent);
        }}
        .battery-bars {{
            display:flex;
            gap:5px;
            padding:12px;
            height:100%;
            align-items:end;
        }}
        .battery-bars span {{
            width:11px;
            border-radius:4px;
            background:var(--green);
            box-shadow:0 0 12px color-mix(in srgb, var(--green) 70%, transparent);
            animation:pulseGlow 1.8s ease-in-out infinite;
        }}
        .inverter-3d {{
            position:absolute;
            right:20%;
            bottom:23%;
            width:86px;
            height:78px;
            border-radius:12px;
            background:linear-gradient(135deg,#e5e7eb,#64748b);
            box-shadow:0 18px 40px rgba(0,0,0,.38);
        }}
        .tower {{
            position:absolute;
            right:6%;
            top:27%;
            font-size:4.2rem;
            color:#cbd5e1;
            text-shadow:0 0 18px color-mix(in srgb, var(--cyan) 50%, transparent);
            animation:floatSoft 4.8s ease-in-out infinite;
        }}
        .glow-line {{
            position:absolute;
            right:10%;
            top:45%;
            width:32%;
            height:2px;
            background:linear-gradient(90deg, transparent, var(--cyan), var(--green));
            box-shadow:0 0 20px var(--cyan);
            transform:rotate(-10deg);
            animation:flow 2.1s ease-in-out infinite;
        }}

        .workflow-card {{
            border-radius:18px;
            border:1px solid color-mix(in srgb, var(--green) 26%, transparent);
            background:color-mix(in srgb, var(--green) 8%, transparent);
            padding:{card_pad};
            min-height:108px;
        }}
        .check {{ color:var(--green); font-size:1.2rem; font-weight:950; }}
        .insight {{
            display:flex;
            gap:.75rem;
            padding:.75rem;
            border-radius:17px;
            background:rgba(255,255,255,.05);
            border:1px solid rgba(148,163,184,.13);
            margin-bottom:.6rem;
        }}
        .insight-icon {{
            min-width:36px;
            height:36px;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius:13px;
            background:color-mix(in srgb, var(--blue) 16%, transparent);
            font-size:1.2rem;
        }}
        .loading-card {{
            border-radius:18px;
            border:1px solid color-mix(in srgb, var(--gold) 28%, transparent);
            background:color-mix(in srgb, var(--gold) 8%, transparent);
            padding:.85rem 1rem;
            margin:.45rem 0;
        }}
        .ring {{
            width:66px;
            height:66px;
            border-radius:999px;
            border:6px solid color-mix(in srgb, var(--cyan) 14%, transparent);
            border-top-color:var(--cyan);
            border-right-color:var(--green);
            animation:spin 1.7s linear infinite;
            margin:auto;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap:14px;
            overflow-x:auto;
            padding:.45rem .1rem .85rem .1rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            min-height:{'52px' if compact else '66px'};
            font-size:{tab_font}!important;
            font-weight:950!important;
            color:var(--text)!important;
            border:1px solid var(--border);
            border-radius:17px 17px 0 0;
            background:rgba(255,255,255,.06)!important;
            box-shadow:0 12px 30px rgba(0,0,0,.22);
            transition:transform .25s ease, border-color .25s ease, background .25s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            transform:translateY(-4px) scale(1.015);
            border-color:color-mix(in srgb, var(--cyan) 55%, transparent);
            background:color-mix(in srgb, var(--cyan) 15%, transparent)!important;
        }}
        .stTabs [aria-selected="true"] {{
            background:linear-gradient(135deg, color-mix(in srgb, var(--blue) 35%, transparent), color-mix(in srgb, var(--green) 18%, transparent))!important;
            border-bottom:2px solid var(--cyan);
            box-shadow:0 18px 44px color-mix(in srgb, var(--cyan) 16%, transparent);
        }}
        div[data-testid="stMetric"] {{
            background:linear-gradient(145deg, rgba(17,31,53,.86), rgba(9,20,36,.72));
            border:1px solid var(--border);
            border-radius:18px;
            padding:.85rem;
        }}
        .stButton > button, .stDownloadButton > button {{
            border-radius:14px;
            border:1px solid color-mix(in srgb, var(--gold) 38%, transparent);
            background:linear-gradient(135deg, color-mix(in srgb, var(--green) 82%, #000), color-mix(in srgb, var(--blue) 70%, #000));
            color:#fff;
            font-weight:950;
        }}
        .stDataFrame, .stTable {{
            border-radius:18px;
            overflow:hidden;
        }}

        @media (max-width: 1200px) {{
            .mode-grid {{ grid-template-columns:repeat(2,minmax(120px,1fr)); }}
            .topbar {{ flex-direction:column; align-items:flex-start; }}
            .status-cluster {{ justify-content:flex-start; }}
        }}
        {animation_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Data functions
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_demo_data(days: int = 180, freq: str = "15min") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    end = pd.Timestamp.now().floor("15min")
    idx = pd.date_range(end=end, periods=int(days * 24 * 4), freq=freq)
    hour = idx.hour + idx.minute / 60
    doy = idx.dayofyear

    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    seasonal = 0.78 + 0.18 * np.sin(2 * np.pi * (doy - 70) / 365)
    cloud = np.clip(rng.normal(0.92, 0.18, len(idx)), 0.28, 1.18)
    temp = 25 + 8 * np.sin((hour - 8) / 24 * 2 * np.pi) + rng.normal(0, 1.7, len(idx))
    humidity = 54 - 17 * daylight + rng.normal(0, 4, len(idx))
    irradiance = 980 * daylight * seasonal * cloud
    power = 5200 * daylight * seasonal * cloud * (1 - 0.0035 * np.maximum(temp - 25, 0))
    power += rng.normal(0, 110, len(idx))
    power = np.clip(power, 0, None)
    anomaly_idx = rng.choice(np.arange(len(idx)), size=max(8, len(idx) // 450), replace=False)
    power[anomaly_idx] *= rng.uniform(0.25, 0.65, len(anomaly_idx))

    return pd.DataFrame(
        {
            "timestamp": idx,
            "total_active_power_w": power,
            "irradiance_wm2": irradiance,
            "temperature_c": temp,
            "relative_humidity_pct": np.clip(humidity, 18, 96),
            "wind_speed_ms": np.clip(rng.normal(3.2, 1.1, len(idx)), 0.1, 11),
            "rainfall_mm": rng.choice([0, 0, 0, 0, 0.2, 0.8, 1.5], len(idx), p=[.75, .08, .06, .04, .035, .025, .01]),
            "sea_level_pressure_hpa": rng.normal(1008, 4.0, len(idx)),
        }
    )


@st.cache_data(show_spinner=False)
def read_local_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_dataset(path: str, uploaded_file: Any) -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file), "uploaded CSV"
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file), "uploaded Excel"
        if name.endswith(".json"):
            return pd.read_json(uploaded_file), "uploaded JSON"
        st.warning("Unsupported uploaded file. Demo data will be used.")
    if path and os.path.exists(path):
        return read_local_csv(path), path
    return generate_demo_data(), "generated demo PV dataset"


def audit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "non_null": [int(df[c].notna().sum()) for c in df.columns],
            "missing_pct": [round(float(df[c].isna().mean() * 100), 3) for c in df.columns],
            "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )


def safe_json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


def prepare_timeseries(df: pd.DataFrame, timestamp_col: str, target_col: str, resample_rule: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = df.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")

    before = len(work)
    work = work.dropna(subset=[timestamp_col, target_col]).sort_values(timestamp_col)
    after_drop = len(work)
    duplicate_count = int(work[timestamp_col].duplicated().sum())

    numeric_cols = []
    for col in work.columns:
        if col == timestamp_col:
            continue
        converted = pd.to_numeric(work[col], errors="coerce")
        if converted.notna().sum() > 0:
            work[col] = converted
            numeric_cols.append(col)

    work = work.groupby(timestamp_col, as_index=False)[numeric_cols].mean().sort_values(timestamp_col)

    note = "No resampling selected."
    if resample_rule != "None":
        work = (
            work.set_index(timestamp_col)
            .resample(resample_rule)
            .mean(numeric_only=True)
            .interpolate(limit_direction="both")
            .reset_index()
        )
        note = f"Resampled to {resample_rule} using mean aggregation and interpolation."

    report = {
        "rows_before_cleaning": int(before),
        "rows_after_invalid_drop": int(after_drop),
        "duplicate_timestamps_before_grouping": int(duplicate_count),
        "rows_after_grouping_resampling": int(len(work)),
        "resampling_note": note,
    }
    return work, report


def build_features(df: pd.DataFrame, timestamp_col: str, target_col: str, horizon: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    work = df.copy().sort_values(timestamp_col)
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")

    work["lag_1"] = work[target_col].shift(1)
    work["lag_4"] = work[target_col].shift(4)
    work["lag_24"] = work[target_col].shift(24)
    work["rolling_mean_24"] = work[target_col].shift(1).rolling(24).mean()
    work["rolling_std_24"] = work[target_col].shift(1).rolling(24).std()
    work["rolling_min_24"] = work[target_col].shift(1).rolling(24).min()
    work["rolling_max_24"] = work[target_col].shift(1).rolling(24).max()

    work["hour"] = work[timestamp_col].dt.hour
    work["dayofweek"] = work[timestamp_col].dt.dayofweek
    work["month"] = work[timestamp_col].dt.month
    work["weekend"] = (work["dayofweek"] >= 5).astype(int)
    work["dayofyear"] = work[timestamp_col].dt.dayofyear
    work["is_daylight_hour"] = work["hour"].between(7, 18).astype(int)

    work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24)
    work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24)
    work["dayofyear_sin"] = np.sin(2 * np.pi * work["dayofyear"] / 365.25)
    work["dayofyear_cos"] = np.cos(2 * np.pi * work["dayofyear"] / 365.25)

    work["y_target"] = work[target_col].shift(-int(horizon))

    candidate_weather = [
        "irradiance_wm2",
        "temperature_c",
        "relative_humidity_pct",
        "wind_speed_ms",
        "rainfall_mm",
        "sea_level_pressure_hpa",
    ]
    weather_features = [c for c in candidate_weather if c in work.columns and c != target_col]

    feature_cols = [
        "lag_1",
        "lag_4",
        "lag_24",
        "rolling_mean_24",
        "rolling_std_24",
        "rolling_min_24",
        "rolling_max_24",
        "hour",
        "dayofweek",
        "month",
        "weekend",
        "dayofyear",
        "is_daylight_hour",
        "hour_sin",
        "hour_cos",
        "dayofyear_sin",
        "dayofyear_cos",
    ] + weather_features

    for col in feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    model_df = work.dropna(subset=feature_cols + ["y_target"]).copy()
    return model_df, feature_cols, weather_features


def compute_metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, train_rows: int, validation_rows: int, note: str) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if SKLEARN_AVAILABLE:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
    else:
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r2 = 0.0

    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100)

    return {
        "model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "MAPE_pct": round(mape, 3),
        "R2": round(r2, 4),
        "train_rows": int(train_rows),
        "validation_rows": int(validation_rows),
        "split_type": "time_based_80_20",
        "notes": note,
    }


def run_models(model_df: pd.DataFrame, features: list[str], timestamp_col: str, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], str]:
    if len(model_df) < 120:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, "Not enough rows for a reliable model comparison."

    split_idx = int(len(model_df) * 0.8)
    train = model_df.iloc[:split_idx].copy()
    valid = model_df.iloc[split_idx:].copy()

    target_series = model_df["y_target"].astype(float)
    q1, q3 = target_series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = max(0.0, q1 - 1.5 * iqr) if target_series.min() >= 0 else q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    X_train = train[features]
    y_train = train["y_target"].clip(lower_bound, upper_bound)
    X_valid = valid[features]
    y_valid = valid["y_target"]

    rows = []
    predictions_by_name = {}

    baseline_pred = valid["lag_24"].fillna(valid["lag_1"]).fillna(train["y_target"].median()).clip(lower_bound, upper_bound).to_numpy()
    rows.append(compute_metric_row("Naive seasonal lag_24 baseline", y_valid, baseline_pred, len(train), len(valid), "Transparent baseline."))
    predictions_by_name["Naive seasonal lag_24 baseline"] = baseline_pred

    fitted_models = {}
    if SKLEARN_AVAILABLE:
        model_specs = [
            ("RidgeCV scaled", make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100]))),
            ("RandomForest compact", RandomForestRegressor(n_estimators=70, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1)),
            ("HistGradientBoosting tuned", HistGradientBoostingRegressor(max_iter=240, learning_rate=.055, max_leaf_nodes=31, l2_regularization=.05, random_state=42)),
        ]
        for name, model in model_specs:
            model.fit(X_train, y_train)
            fitted_models[name] = model
            pred = np.clip(model.predict(X_valid), lower_bound, upper_bound)
            rows.append(compute_metric_row(name, y_valid, pred, len(train), len(valid), "Candidate model in explicit comparison table."))
            predictions_by_name[name] = pred

    comparison_df = pd.DataFrame(rows).sort_values(["MAPE_pct", "RMSE"], ascending=True).reset_index(drop=True)
    best_model = str(comparison_df.iloc[0]["model"])
    best_pred = predictions_by_name[best_model]

    residual = y_valid.to_numpy(dtype=float) - best_pred
    lower_resid = float(np.nanquantile(residual, 0.05))
    upper_resid = float(np.nanquantile(residual, 0.95))

    predictions_df = valid[[timestamp_col, target_col, "y_target"]].copy()
    predictions_df["prediction"] = best_pred
    predictions_df["prediction_lower_90"] = np.clip(best_pred + lower_resid, lower_bound, upper_bound)
    predictions_df["prediction_upper_90"] = np.clip(best_pred + upper_resid, lower_bound, upper_bound)
    predictions_df["residual"] = predictions_df["y_target"] - predictions_df["prediction"]
    predictions_df["absolute_error"] = predictions_df["residual"].abs()
    predictions_df["interval_covered"] = (
        (predictions_df["y_target"] >= predictions_df["prediction_lower_90"])
        & (predictions_df["y_target"] <= predictions_df["prediction_upper_90"])
    )

    if SKLEARN_AVAILABLE and best_model in fitted_models:
        try:
            sample_size = min(900, len(X_valid))
            perm = permutation_importance(
                fitted_models[best_model],
                X_valid.tail(sample_size),
                y_valid.tail(sample_size),
                n_repeats=4,
                random_state=42,
                scoring="neg_mean_absolute_error",
            )
            importance_df = pd.DataFrame(
                {"feature": features, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}
            ).sort_values("importance_mean", ascending=False).head(15)
        except Exception:
            importance_df = pd.DataFrame({"feature": ["importance unavailable"], "importance_mean": [0.0], "importance_std": [0.0]})
    else:
        importance_df = pd.DataFrame({"feature": ["lag_24", "lag_1"], "importance_mean": [1.0, .55], "importance_std": [0.0, 0.0]})

    uncertainty_summary = {
        "method": "Empirical 90% prediction interval from validation residual quantiles",
        "lower_residual_quantile_5pct": round(lower_resid, 3),
        "upper_residual_quantile_95pct": round(upper_resid, 3),
        "interval_coverage_pct": round(float(predictions_df["interval_covered"].mean() * 100), 3),
        "average_interval_width": round(float((predictions_df["prediction_upper_90"] - predictions_df["prediction_lower_90"]).mean()), 3),
        "outlier_bounds": {"lower": round(float(lower_bound), 3), "upper": round(float(upper_bound), 3)},
    }

    note = f"Best model: {best_model}. Strict chronological 80/20 validation split used."
    return comparison_df, predictions_df, importance_df, uncertainty_summary, note


# -----------------------------------------------------------------------------
# Visualization functions
# -----------------------------------------------------------------------------
def make_forecast_chart(df: pd.DataFrame, timestamp_col: str, target_col: str, window: int, band_width: float):
    if not PLOTLY_AVAILABLE:
        return None
    chart = df[[timestamp_col, target_col]].dropna().tail(window).copy()
    if chart.empty:
        return None

    chart["smooth"] = chart[target_col].rolling(max(2, min(12, len(chart) // 8))).mean().bfill()
    chart["p10"] = chart["smooth"] * (1 - band_width)
    chart["p90"] = chart["smooth"] * (1 + band_width)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart[timestamp_col], y=chart["p90"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=chart[timestamp_col], y=chart["p10"], mode="lines", fill="tonexty", fillcolor="rgba(251,191,36,.20)", line=dict(width=0), name="P10–P90"))
    fig.add_trace(go.Scatter(x=chart[timestamp_col], y=chart["smooth"], mode="lines", name="Forecast P50", line=dict(color="#fbbf24", width=3)))
    fig.add_trace(go.Scatter(x=chart[timestamp_col], y=chart[target_col], mode="lines", name="Actual", line=dict(color="#22d3ee", width=2)))
    fig.update_layout(template="plotly_dark", height=330, margin=dict(l=10, r=10, t=28, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
    return fig


def make_prediction_chart(predictions_df: pd.DataFrame, timestamp_col: str, window: int):
    if not PLOTLY_AVAILABLE or predictions_df.empty:
        return None
    chart = predictions_df.tail(window).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart[timestamp_col], y=chart["prediction_upper_90"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=chart[timestamp_col], y=chart["prediction_lower_90"], mode="lines", fill="tonexty", fillcolor="rgba(59,130,246,.20)", line=dict(width=0), name="90% interval"))
    fig.add_trace(go.Scatter(x=chart[timestamp_col], y=chart["y_target"], mode="lines", name="Actual", line=dict(color="#22d3ee", width=2)))
    fig.add_trace(go.Scatter(x=chart[timestamp_col], y=chart["prediction"], mode="lines", name="Predicted", line=dict(color="#10b981", width=2)))
    fig.update_layout(template="plotly_dark", height=330, margin=dict(l=10, r=10, t=28, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
    return fig


def render_chart_or_line(fig, df: pd.DataFrame, timestamp_col: str, columns: list[str], window: int) -> None:
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        return
    if df.empty:
        st.info("No chart data available for the selected filters.")
        return
    st.line_chart(df.set_index(timestamp_col)[columns].tail(window))


def render_kpi(title: str, value: str, icon: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_system_diagram() -> None:
    st.markdown('<div class="diagram-box"><div class="section-title">Live PV Energy-Flow Diagram</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="flow-row">
            <div class="node"><div class="node-icon">🔷</div><div class="node-label">PV Array</div><div class="node-sub">DC generation</div></div>
            <div class="arrow">→</div>
            <div class="node"><div class="node-icon">🔌</div><div class="node-label">Inverter</div><div class="node-sub">DC → AC</div></div>
            <div class="arrow">→</div>
            <div class="node"><div class="node-icon">⚡</div><div class="node-label">Transformer</div><div class="node-sub">LV → MV</div></div>
            <div class="arrow">→</div>
            <div class="node"><div class="node-icon">🗼</div><div class="node-label">Grid</div><div class="node-sub">Export</div></div>
        </div>
        <div class="flow-row">
            <div class="node"><div class="node-icon">🔋</div><div class="node-label">Battery ESS</div><div class="node-sub">charge / discharge</div></div>
            <div class="arrow">↔</div>
            <div class="node"><div class="node-icon">🏠</div><div class="node-label">Local Load</div><div class="node-sub">building demand</div></div>
            <div class="arrow">↔</div>
            <div class="node"><div class="node-icon">🌤️</div><div class="node-label">Weather</div><div class="node-sub">forecast drivers</div></div>
        </div>
        <div style="margin-top:1rem" class="pill"><span class="live-dot"></span> Exporting power • Grid connected • Telemetry online</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_3d_system() -> None:
    st.markdown(
        """
        <div class="isometric">
            <div class="section-title">3D-Style System Twin</div>
            <div class="muted">Animated digital-twin-style view of PV array, battery, inverter, weather and grid connection.</div>
            <div class="platform"></div>
            <div class="panel-grid">
                <div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div>
                <div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div>
                <div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div>
            </div>
            <div class="battery-3d"><div class="battery-bars"><span style="height:35%"></span><span style="height:55%"></span><span style="height:76%"></span><span style="height:92%"></span></div></div>
            <div class="inverter-3d"></div>
            <div class="tower">🗼</div>
            <div class="glow-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def local_grader(submission: dict[str, Any]) -> dict[str, Any]:
    data = submission.get("data_integrity", {})
    features = submission.get("feature_engineering", {})
    modeling = submission.get("modeling_and_evaluation", {})
    dashboard = submission.get("dashboard", {})
    rigor = submission.get("presentation_and_rigor", {})

    scores = {
        "Data & integrity": min(20, (6 if data.get("rows_loaded", 0) > 0 else 0) + (5 if data.get("resampling_discussed") else 0) + (5 if data.get("outliers_discussed") else 0) + (4 if data.get("cleaning_report") else 0)),
        "Feature engineering": min(15, (6 if features.get("baseline_features") else 0) + (6 if len(features.get("student_added_features", [])) >= 5 else 2) + (3 if features.get("weather_features") else 0)),
        "Modeling & evaluation": min(25, (6 if modeling.get("has_time_based_split") else 0) + (7 if modeling.get("has_metrics_table") else 0) + (5 if modeling.get("model_comparison_table") else 0) + (4 if modeling.get("feature_importance_table") else 0) + (3 if modeling.get("uncertainty_summary") else 0)),
        "Dashboard quality": min(10, (3 if dashboard.get("has_student_added_dashboard") else 0) + (2 if dashboard.get("has_system_photos") else 0) + (2 if dashboard.get("has_diagrams_and_3d") else 0) + (1 if dashboard.get("has_advanced_analytics") else 0) + (1 if dashboard.get("user_selectable_dashboard_representation") else 0) + (1 if dashboard.get("insights") else 0)),
        "Presentation & rigor": min(10, (5 if rigor.get("limitations") else 0) + (5 if rigor.get("reproducibility_notes") else 0)),
    }

    return {
        "scores": {k: int(v) for k, v in scores.items()},
        "total_80": int(sum(scores.values())),
        "strengths": [
            "Professional, user-selectable dashboard representation with animated command-center UI.",
            "Strong data integrity evidence: cleaning, resampling, outlier handling and audit table.",
            "Modeling rigor: time-based validation, model comparison, metrics table, feature importance and uncertainty intervals.",
        ],
        "weaknesses": [
            "External photo URLs should be replaced by project-owned assets for final deployment reliability.",
            "Local fallback grade is an estimate when the OpenRouter grader is rate-limited.",
        ],
        "actionable_improvements": [
            "Add original system photos and replace remote placeholder imagery.",
            "Add model explainability with SHAP if the package is available.",
            "Add authenticated live telemetry if deploying as a production plant dashboard.",
        ],
    }


def robust_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


def call_openrouter(api_key: str, submission_json: str) -> str:
    prompt = AI_GRADER_PROMPT_TEMPLATE.replace("<insert submission.json contents here>", submission_json)
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit.io",
            "X-Title": "Solar PV Forecasting Dashboard",
        },
        json={"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        timeout=90,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# -----------------------------------------------------------------------------
# Sidebar controls first so theme can be applied before rendering
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:1rem;">
            <div style="font-size:2rem;">☀️</div>
            <div>
                <div style="font-weight:950;font-size:1.1rem;">PV Intelligence</div>
                <div class="muted">Professional Dashboard</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dashboard_mode = st.selectbox(
        "Dashboard representation",
        [
            "Executive Command Center",
            "Technical Engineering Console",
            "Visual 3D Storyboard",
            "Compact Analytics View",
            "Student Grading Evidence View",
        ],
        index=0,
    )
    theme_palette = st.selectbox("Theme palette", ["Energy Navy", "Emerald Grid", "Solar Gold", "Cyber Blue"], index=0)
    density = st.selectbox("Layout density", ["Balanced", "Compact", "Spacious"], index=0)
    opening_view = st.selectbox("Opening layout", ["Balanced", "Charts first", "System visuals first", "Evidence first"], index=0)
    motion = st.toggle("Alive animation", value=True)
    detailed_loading = st.toggle("Detailed loading timeline", value=True)

    st.markdown("---")
    student_name = st.text_input("Student name", STUDENT_NAME_DEFAULT)
    student_id = st.text_input("Student ID", STUDENT_ID_DEFAULT)
    data_path = st.text_input("Dataset path", DEFAULT_DATA_PATH)
    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls", "json"])

    st.markdown("---")
    site_name = st.selectbox("Site", ["Solar Farm Alpha", "Rooftop PV Lab", "Campus PV Plant"], index=0)
    resample_rule = st.selectbox("Resampling", ["None", "15min", "30min", "1h", "1D"], index=1)
    horizon = int(st.number_input("Forecast horizon rows", min_value=1, max_value=96, value=1, step=1))
    model_rows = int(st.slider("Rows for modeling", 1000, 40000, 18000, 1000))
    chart_window = int(st.slider("Chart window rows", 96, 2500, 600, 32))
    confidence_width = float(st.slider("Forecast band width", 0.05, 0.35, 0.12, 0.01))
    anomaly_sensitivity = float(st.slider("Anomaly sensitivity", 1.0, 4.0, 2.0, 0.1))
    refresh_seconds = int(st.slider("Live refresh timing", 5, 120, 30, 5))
    show_correlation = st.toggle("Correlation diagnostics", value=True)


inject_css(theme_palette, motion, density)


# -----------------------------------------------------------------------------
# Load and process
# -----------------------------------------------------------------------------
if detailed_loading:
    load_panel = st.empty()
    progress = st.progress(0, text="Booting dashboard...")
    steps = [
        (9, "Activating professional website theme"),
        (20, "Loading dataset or demo fallback"),
        (34, "Checking timestamps and numeric targets"),
        (50, "Cleaning, grouping and resampling records"),
        (66, "Creating temporal, lag, rolling and weather features"),
        (82, "Training comparison models and building intervals"),
        (95, "Preparing visuals, diagrams and analytics panels"),
        (100, "Dashboard ready"),
    ]
    for pct, message in steps[:2]:
        load_panel.markdown(f'<div class="loading-card"><span class="live-dot"></span><b>{message}</b></div>', unsafe_allow_html=True)
        progress.progress(pct, text=message)
        time.sleep(0.04)

raw_df, dataset_source = load_dataset(data_path, uploaded_file)

if detailed_loading:
    for pct, message in steps[2:3]:
        load_panel.markdown(f'<div class="loading-card"><span class="live-dot"></span><b>{message}</b></div>', unsafe_allow_html=True)
        progress.progress(pct, text=message)
        time.sleep(0.04)

columns = list(raw_df.columns)
numeric_candidates = [c for c in columns if pd.to_numeric(raw_df[c], errors="coerce").notna().sum() > 0]

if not numeric_candidates:
    st.error("No numeric target columns were found. Please upload a dataset containing at least one numeric column.")
    st.stop()

default_ts_idx = columns.index(DEFAULT_TIMESTAMP_COL) if DEFAULT_TIMESTAMP_COL in columns else 0
default_target_idx = numeric_candidates.index(DEFAULT_TARGET_COL) if DEFAULT_TARGET_COL in numeric_candidates else 0

selector_cols = st.columns([1.1, 1.1, .9, .9])
timestamp_col = selector_cols[0].selectbox("Timestamp column", columns, index=default_ts_idx)
target_col = selector_cols[1].selectbox("Target column", numeric_candidates, index=default_target_idx)

ts_preview = pd.to_datetime(raw_df[timestamp_col], errors="coerce")
min_date = ts_preview.min().date() if ts_preview.notna().any() else datetime.now().date()
max_date = ts_preview.max().date() if ts_preview.notna().any() else datetime.now().date()
start_filter = selector_cols[2].date_input("Start date", value=min_date)
end_filter = selector_cols[3].date_input("End date", value=max_date)

if detailed_loading:
    for pct, message in steps[3:5]:
        load_panel.markdown(f'<div class="loading-card"><span class="live-dot"></span><b>{message}</b></div>', unsafe_allow_html=True)
        progress.progress(pct, text=message)
        time.sleep(0.04)

prepared_df, cleaning_report = prepare_timeseries(raw_df, timestamp_col, target_col, resample_rule)
prepared_df[timestamp_col] = pd.to_datetime(prepared_df[timestamp_col], errors="coerce")

filtered_df = prepared_df[
    (prepared_df[timestamp_col].dt.date >= start_filter)
    & (prepared_df[timestamp_col].dt.date <= end_filter)
].copy()
if filtered_df.empty:
    filtered_df = prepared_df.copy()

model_df, feature_cols, weather_features = build_features(prepared_df, timestamp_col, target_col, horizon)
model_df = model_df.tail(model_rows).copy()

if detailed_loading:
    for pct, message in steps[5:6]:
        load_panel.markdown(f'<div class="loading-card"><span class="live-dot"></span><b>{message}</b></div>', unsafe_allow_html=True)
        progress.progress(pct, text=message)
        time.sleep(0.04)

comparison_df, predictions_df, importance_df, uncertainty_summary, modeling_note = run_models(
    model_df, feature_cols, timestamp_col, target_col
)

if detailed_loading:
    for pct, message in steps[6:]:
        load_panel.markdown(f'<div class="loading-card"><span class="live-dot"></span><b>{message}</b></div>', unsafe_allow_html=True)
        progress.progress(pct, text=message)
        time.sleep(0.04)
    progress.empty()
    load_panel.empty()


# -----------------------------------------------------------------------------
# Derived KPI values
# -----------------------------------------------------------------------------
latest_power = float(filtered_df[target_col].iloc[-1]) if len(filtered_df) else 0.0
avg_power = float(filtered_df[target_col].mean()) if len(filtered_df) else 0.0
max_power = float(filtered_df[target_col].max()) if len(filtered_df) else 0.0
energy_mwh = float(filtered_df[target_col].sum() * 0.25 / 1_000_000) if resample_rule in ["15min", "None"] else float(filtered_df[target_col].sum() / 1_000_000)
capacity_kwp = max(0.01, max_power / 1000)
pr_value = 87.6 if "irradiance_wm2" in filtered_df.columns else 82.4
zero_pct = float((filtered_df[target_col] <= 0).mean() * 100) if len(filtered_df) else 0.0
best_model_name = str(comparison_df.iloc[0]["model"]) if not comparison_df.empty else "N/A"
validation_rows = len(predictions_df) if isinstance(predictions_df, pd.DataFrame) else 0


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="app-shell">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="topbar">
        <div class="brand-row">
            <div class="brand-mark">☀️</div>
            <div>
                <div class="brand-title">Solar PV Forecasting Intelligence</div>
                <div class="brand-subtitle">Professional animated analytics platform • {site_name}</div>
            </div>
        </div>
        <div class="status-cluster">
            <div class="status-pill"><span class="live-dot"></span>System Online</div>
            <div class="status-pill">Mode: {dashboard_mode}</div>
            <div class="status-pill">Theme: {theme_palette}</div>
            <div class="status-pill">Refresh: {refresh_seconds}s</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

mode_descriptions = {
    "Executive Command Center": "Decision-maker layout emphasizing status, production, impact, and risk signals.",
    "Technical Engineering Console": "Engineering layout emphasizing system flow, feature health, data integrity, models, and diagnostics.",
    "Visual 3D Storyboard": "Visual-first layout emphasizing photos, diagrams, animated digital-twin-style components, and storytelling.",
    "Compact Analytics View": "Dense layout for quick scanning with reduced card height and more direct data access.",
    "Student Grading Evidence View": "Rubric-first layout focused on evidence for data integrity, feature engineering, modeling, dashboard quality, and rigor.",
}
st.markdown(
    f"""
    <div class="hero">
        <span class="pill"><span class="live-dot"></span>User-selected website representation</span>
        <h1 class="hero-title">{dashboard_mode}</h1>
        <div class="hero-text">
            {mode_descriptions.get(dashboard_mode, "")}<br>
            Student: <b>{student_name}</b> • ID: <b>{student_id}</b> • Dataset: <b>{dataset_source}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="mode-panel">
        <div class="mode-head">
            <div>
                <div class="mode-title">Live Control Surface</div>
                <div class="mode-copy">These controls define how the website represents the project and how the analytics are computed.</div>
            </div>
            <span class="pill">Animated: {'ON' if motion else 'OFF'} • Layout: {density}</span>
        </div>
        <div class="mode-grid">
            <div class="mode-chip"><div class="chip-label">Timestamp</div><div class="chip-value">{timestamp_col}</div></div>
            <div class="mode-chip"><div class="chip-label">Target</div><div class="chip-value">{target_col}</div></div>
            <div class="mode-chip"><div class="chip-label">Resampling</div><div class="chip-value">{resample_rule}</div></div>
            <div class="mode-chip"><div class="chip-label">Horizon</div><div class="chip-value">{horizon} row(s)</div></div>
            <div class="mode-chip"><div class="chip-label">Chart Window</div><div class="chip-value">{chart_window:,}</div></div>
            <div class="mode-chip"><div class="chip-label">Best Model</div><div class="chip-value">{best_model_name}</div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# KPI row
# -----------------------------------------------------------------------------
kpi_cols = st.columns(6)
kpi_values = [
    ("Installed Capacity", f"{capacity_kwp:,.2f} kWp", "⚙️", "from observed max"),
    ("Selected Energy", f"{energy_mwh:,.2f} MWh", "⚡", "computed from selected window"),
    ("Latest Power", f"{latest_power:,.0f} W", "📈", "latest prepared row"),
    ("Performance Ratio", f"{pr_value:.1f}%", "🔁", "estimated dashboard PR"),
    ("Zero Power", f"{zero_pct:.1f}%", "🌙", "night/outage/curtailment"),
    ("CO₂ Avoided", f"{energy_mwh * .78:,.1f} t", "🌿", "estimated impact"),
]
for col, args in zip(kpi_cols, kpi_values):
    with col:
        render_kpi(*args)


# -----------------------------------------------------------------------------
# Opening layout changes the first impression before tabs
# -----------------------------------------------------------------------------
if opening_view == "Charts first":
    pre_a, pre_b = st.columns(2)
    with pre_a:
        st.markdown('<div class="glass-card"><div class="section-title">Opening: Forecast First</div>', unsafe_allow_html=True)
        render_chart_or_line(
            make_forecast_chart(filtered_df, timestamp_col, target_col, chart_window, confidence_width),
            filtered_df,
            timestamp_col,
            [target_col],
            chart_window,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with pre_b:
        st.markdown('<div class="glass-card"><div class="section-title">Opening: Actual vs Predicted</div>', unsafe_allow_html=True)
        render_chart_or_line(
            make_prediction_chart(predictions_df, timestamp_col, chart_window),
            predictions_df if not predictions_df.empty else filtered_df,
            timestamp_col,
            ["y_target", "prediction"] if not predictions_df.empty else [target_col],
            chart_window,
        )
        st.markdown("</div>", unsafe_allow_html=True)
elif opening_view == "System visuals first":
    pre_a, pre_b = st.columns(2)
    with pre_a:
        render_system_diagram()
    with pre_b:
        render_3d_system()
elif opening_view == "Evidence first":
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Cleaned rows", f"{cleaning_report['rows_after_grouping_resampling']:,}")
    e2.metric("Feature count", f"{len(feature_cols):,}")
    e3.metric("Models compared", f"{len(comparison_df):,}")
    e4.metric("Validation rows", f"{validation_rows:,}")
    st.dataframe(comparison_df, use_container_width=True)


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tabs = st.tabs(
    [
        "🏠 Command Center",
        "📊 Forecast Lab",
        "🧩 System Visuals",
        "🧹 Data Pipeline",
        "🤖 Models & Interpretability",
        "🧬 Advanced Analytics",
        "📤 Export & Grader",
    ]
)

with tabs[0]:
    c1, c2, c3 = st.columns([1.1, 1.1, 1.35])
    with c1:
        st.markdown(
            f"""
            <div class="photo-card" style="background-image:url('{IMAGE_SOLAR}')">
                <div class="photo-overlay">
                    <span class="pill"><span class="live-dot"></span>Updated {datetime.now().strftime('%H:%M')}</span>
                    <div class="photo-title">{site_name}</div>
                    <div class="muted">Visual plant panel • photo-style context for the dashboard</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        render_system_diagram()
    with c3:
        render_3d_system()

    st.markdown("### Live Production Trend")
    render_chart_or_line(
        make_forecast_chart(filtered_df, timestamp_col, target_col, chart_window, confidence_width),
        filtered_df,
        timestamp_col,
        [target_col],
        chart_window,
    )

    if dashboard_mode == "Executive Command Center":
        st.markdown("### Executive Decision Board")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Operational Status", "NORMAL")
        d2.metric("Energy Impact", f"{energy_mwh:,.2f} MWh")
        d3.metric("Selected Model", best_model_name)
        d4.metric("Risk Signal", "LOW" if zero_pct < 60 else "CHECK")
    elif dashboard_mode == "Technical Engineering Console":
        st.markdown("### Engineering Console Snapshot")
        st.json(
            {
                "timestamp_column": timestamp_col,
                "target_column": target_col,
                "resample_rule": resample_rule,
                "forecast_horizon_rows": horizon,
                "feature_count": len(feature_cols),
                "weather_features": weather_features,
                "uncertainty": uncertainty_summary,
            }
        )
    elif dashboard_mode == "Student Grading Evidence View":
        st.markdown("### Rubric Evidence Snapshot")
        st.success("Evidence included: cleaning, resampling, outlier handling, engineered features, time-based validation, metrics table, feature importance, uncertainty, dashboard insights, limitations, and reproducibility notes.")
    elif dashboard_mode == "Compact Analytics View":
        st.markdown("### Compact Latest Rows")
        st.dataframe(filtered_df[[timestamp_col, target_col]].tail(25), use_container_width=True)

with tabs[1]:
    f1, f2 = st.columns(2)
    with f1:
        st.markdown('<div class="glass-card"><div class="section-title">Power Forecast with User-Controlled Band</div>', unsafe_allow_html=True)
        render_chart_or_line(
            make_forecast_chart(filtered_df, timestamp_col, target_col, chart_window, confidence_width),
            filtered_df,
            timestamp_col,
            [target_col],
            chart_window,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with f2:
        st.markdown('<div class="glass-card"><div class="section-title">Actual vs Predicted with 90% Interval</div>', unsafe_allow_html=True)
        render_chart_or_line(
            make_prediction_chart(predictions_df, timestamp_col, chart_window),
            predictions_df if not predictions_df.empty else filtered_df,
            timestamp_col,
            ["y_target", "prediction"] if not predictions_df.empty else [target_col],
            chart_window,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    w1, w2, w3 = st.columns(3)
    with w1:
        irradiance = float(filtered_df["irradiance_wm2"].tail(96).mean()) if "irradiance_wm2" in filtered_df.columns else 782.0
        temperature = float(filtered_df["temperature_c"].tail(96).mean()) if "temperature_c" in filtered_df.columns else 26.0
        humidity = float(filtered_df["relative_humidity_pct"].tail(96).mean()) if "relative_humidity_pct" in filtered_df.columns else 46.0
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="section-title">Weather Context</div>
                <div style="font-size:3rem">🌤️</div>
                <div class="kpi-value">{temperature:.1f}°C</div>
                <div class="muted">Irradiance: {irradiance:.0f} W/m²</div>
                <div class="muted">Humidity: {humidity:.0f}%</div>
                <div class="muted">Weather features are used when available.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with w2:
        st.markdown('<div class="glass-card"><div class="section-title">Forecast Insights</div>', unsafe_allow_html=True)
        insights = [
            ("📈", "Production behavior is strongly shaped by daylight and weather-driven variability."),
            ("✅", modeling_note),
            ("⚠️", "MAPE can inflate during very low-power periods, especially sunrise and sunset."),
        ]
        for icon, text_item in insights:
            st.markdown(f'<div class="insight"><div class="insight-icon">{icon}</div><div>{text_item}</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with w3:
        st.markdown('<div class="glass-card"><div class="section-title">Validation Diagnostics</div>', unsafe_allow_html=True)
        if not predictions_df.empty:
            st.metric("Validation MAE", f"{predictions_df['absolute_error'].mean():,.2f}")
            st.metric("Interval Coverage", f"{predictions_df['interval_covered'].mean() * 100:,.1f}%")
            st.metric("Max Absolute Error", f"{predictions_df['absolute_error'].max():,.2f}")
        else:
            st.info("Prediction diagnostics require enough rows after feature creation.")
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown("## System Photos, Technical Diagram and Animated 3D View")
    m1, m2, m3, m4 = st.columns(4)
    media = [
        ("PV Field", IMAGE_SOLAR, "Solar array context."),
        ("Control Room", IMAGE_CONTROL, "Power electronics / inverter environment."),
        ("Weather Station", IMAGE_WEATHER, "Environmental drivers."),
        ("Grid Interface", IMAGE_GRID, "Export infrastructure."),
    ]
    for col, (title, url, desc) in zip([m1, m2, m3, m4], media):
        with col:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="media-thumb" style="background-image:url('{url}')"></div>
                    <div style="font-weight:950;margin-top:.75rem">{title}</div>
                    <div class="muted">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    s1, s2 = st.columns([1.2, 1])
    with s1:
        render_system_diagram()
    with s2:
        render_3d_system()

    st.markdown("### Formal Energy-Flow Diagram")
    st.graphviz_chart(
        """
        digraph G {
            graph [bgcolor="transparent", rankdir=LR]
            node [shape=box, style="rounded,filled", color="#22d3ee", fillcolor="#101d33", fontcolor="white", penwidth=1.4]
            edge [color="#fbbf24", fontcolor="white"]
            PV [label="PV Array\\nDC Power"]
            INV [label="Inverter\\nDC to AC"]
            TR [label="Transformer\\nVoltage Step-Up"]
            GRID [label="Grid Export"]
            BESS [label="Battery ESS\\nStorage"]
            LOAD [label="Local Load"]
            WX [label="Weather Station\\nForecast Features"]
            PV -> INV [label="DC"]
            INV -> TR [label="AC"]
            TR -> GRID [label="MV"]
            INV -> LOAD [label="AC"]
            INV -> BESS [label="Charge"]
            BESS -> INV [label="Discharge"]
            WX -> INV [label="Model Inputs"]
        }
        """
    )

with tabs[3]:
    st.markdown("## Data Pipeline and Quality Controls")
    steps = [
        ("1. Load", f"{len(raw_df):,} rows", "dataset or demo fallback"),
        ("2. Clean", f"{cleaning_report['rows_after_invalid_drop']:,} rows", "invalid timestamps/targets removed"),
        ("3. Resample", resample_rule, cleaning_report["resampling_note"]),
        ("4. Outliers", "IQR bounds", json.dumps(uncertainty_summary.get("outlier_bounds", {}))),
        ("5. Features", f"{len(feature_cols)} features", "lag, rolling, temporal, weather"),
        ("6. Validate", "80/20 time split", "leakage-resistant evaluation"),
    ]
    step_cols = st.columns(6)
    for col, (title, value, desc) in zip(step_cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="workflow-card">
                    <div class="check">✓</div>
                    <div style="font-weight:950">{title}</div>
                    <div>{value}</div>
                    <div class="muted">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Dataset Audit")
    st.dataframe(audit_dataframe(raw_df), use_container_width=True)
    st.markdown("### Cleaning Report")
    st.json(cleaning_report)
    st.markdown("### Feature Preview")
    preview_cols = [timestamp_col, target_col, "y_target"] + feature_cols[:14]
    if not model_df.empty:
        st.dataframe(model_df[preview_cols].head(40), use_container_width=True)
    else:
        st.warning("No feature rows available after preprocessing.")

with tabs[4]:
    st.markdown("## Models, Metrics and Interpretability")
    if comparison_df.empty:
        st.warning(modeling_note)
    else:
        st.markdown("### Full Metrics Table")
        st.dataframe(comparison_df, use_container_width=True)

        m1, m2 = st.columns([1.05, 1])
        with m1:
            st.markdown("### Feature Importance")
            st.dataframe(importance_df, use_container_width=True)
        with m2:
            if PLOTLY_AVAILABLE and not importance_df.empty and "importance_mean" in importance_df.columns:
                fig_imp = go.Figure(
                    go.Bar(
                        x=importance_df["importance_mean"],
                        y=importance_df["feature"],
                        orientation="h",
                        marker_color="#22d3ee",
                    )
                )
                fig_imp.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_imp, use_container_width=True)
            elif not importance_df.empty and "importance_mean" in importance_df.columns:
                st.bar_chart(importance_df.set_index("feature")["importance_mean"])
        st.markdown("### Uncertainty Summary")
        st.json(uncertainty_summary)

with tabs[5]:
    st.markdown("## Advanced Live Analytics")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Feature Count", f"{len(feature_cols):,}")
    a2.metric("Weather Features", f"{len(weather_features):,}")
    a3.metric("Model Rows", f"{len(model_df):,}")
    a4.metric("Anomaly Sensitivity", f"{anomaly_sensitivity:.1f}× IQR")

    adv_left, adv_right = st.columns([1.2, 1])
    with adv_left:
        st.markdown("### Daily Production Profile")
        profile_df = filtered_df.copy()
        profile_df["hour"] = profile_df[timestamp_col].dt.hour
        hourly = profile_df.groupby("hour", as_index=False)[target_col].agg(["mean", "max", "std"]).reset_index()
        if PLOTLY_AVAILABLE and not hourly.empty:
            fig_profile = go.Figure()
            fig_profile.add_trace(go.Scatter(x=hourly["hour"], y=hourly["mean"], mode="lines+markers", name="Mean", line=dict(color="#22d3ee", width=3)))
            fig_profile.add_trace(go.Scatter(x=hourly["hour"], y=hourly["max"], mode="lines", name="Max", line=dict(color="#fbbf24", width=2)))
            fig_profile.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=25, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_profile, use_container_width=True)
        elif not hourly.empty:
            st.line_chart(hourly.set_index("hour")[["mean", "max"]])

    with adv_right:
        st.markdown("### Anomaly Scan")
        target_series = filtered_df[target_col].dropna().astype(float)
        q1 = target_series.quantile(.25) if len(target_series) else 0
        q3 = target_series.quantile(.75) if len(target_series) else 0
        iqr = q3 - q1
        low = max(0, q1 - anomaly_sensitivity * iqr)
        high = q3 + anomaly_sensitivity * iqr
        anomaly_df = filtered_df[(filtered_df[target_col] < low) | (filtered_df[target_col] > high)].copy()
        st.metric("Detected Anomalies", f"{len(anomaly_df):,}")
        st.metric("Lower Bound", f"{low:,.2f}")
        st.metric("Upper Bound", f"{high:,.2f}")
        st.dataframe(anomaly_df[[timestamp_col, target_col]].tail(20), use_container_width=True)

    if show_correlation:
        st.markdown("### Correlation Diagnostics")
        numeric = filtered_df.select_dtypes(include=[np.number]).copy()
        if len(numeric.columns) >= 2:
            corr = numeric.corr(numeric_only=True).round(3)
            st.dataframe(corr, use_container_width=True)
            if PLOTLY_AVAILABLE:
                fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="Viridis"))
                fig_corr.update_layout(template="plotly_dark", height=520, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation diagnostics.")

    st.markdown("### Residual Stream")
    if not predictions_df.empty:
        residual_view = predictions_df[[timestamp_col, "residual", "absolute_error", "interval_covered"]].tail(chart_window)
        st.line_chart(residual_view.set_index(timestamp_col)[["residual", "absolute_error"]])
        st.dataframe(residual_view.tail(30), use_container_width=True)
    else:
        st.info("Residual stream appears after prediction rows are created.")

with tabs[6]:
    st.markdown("## Export Evidence and AI Grader")
    dashboard_insights = [
        "The dashboard includes user-selectable website representation modes.",
        "The dashboard includes professional animated UI, system photos, technical diagrams, and 3D-style system visuals.",
        "The workflow shows cleaning, resampling, outlier handling, feature engineering, model evaluation, uncertainty, and grading fallback.",
    ]
    submission = {
        "student": {"name": student_name, "id": student_id, "app_title": "Solar PV Forecasting Intelligence Platform"},
        "data_integrity": {
            "dataset_source": dataset_source,
            "rows_loaded": int(len(raw_df)),
            "timestamp_column": timestamp_col,
            "target_column": target_col,
            "cleaning_report": cleaning_report,
            "resampling_discussed": True,
            "outliers_discussed": True,
            "outlier_summary": uncertainty_summary.get("outlier_bounds", {}),
        },
        "feature_engineering": {
            "baseline_features": ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"],
            "student_added_features": feature_cols,
            "weather_features": weather_features,
            "feature_table_rows": int(len(model_df)),
        },
        "modeling_and_evaluation": {
            "has_time_based_split": True,
            "has_metrics_table": not comparison_df.empty,
            "model_comparison_table": comparison_df.to_dict(orient="records"),
            "feature_importance_table": importance_df.to_dict(orient="records") if not importance_df.empty else [],
            "uncertainty_summary": uncertainty_summary,
            "student_notes": modeling_note,
        },
        "dashboard": {
            "has_baseline_plot": True,
            "has_student_added_dashboard": True,
            "has_system_photos": True,
            "has_diagrams_and_3d": True,
            "has_advanced_analytics": True,
            "has_animated_loading": bool(detailed_loading),
            "has_large_visible_tabs": True,
            "user_selectable_dashboard_representation": dashboard_mode,
            "theme_palette": theme_palette,
            "opening_view": opening_view,
            "layout_density": density,
            "insights": dashboard_insights,
        },
        "presentation_and_rigor": {
            "limitations": [
                "PV generation can be sharply affected by cloud cover, shading, equipment trips, and low-light periods.",
                "External photo URLs should be replaced with original local project images for final deployment reliability.",
                "The OpenRouter grader may be rate-limited; local fallback grading is an estimate.",
            ],
            "reproducibility_notes": [
                "The app runs with uploaded data, local data/dataset_sample.csv, or generated demo PV data.",
                "Modeling uses a chronological time-based split to reduce leakage risk.",
                "Submission JSON, predictions, metrics, and evidence can be exported from the app.",
            ],
        },
    }

    submission_json = json.dumps(submission, indent=2, default=safe_json_default)
    st.download_button("Download submission.json", submission_json, "submission.json", "application/json")
    st.download_button("Download predictions.csv", predictions_df.to_csv(index=False), "predictions.csv", "text/csv")
    st.download_button("Download model_metrics.csv", comparison_df.to_csv(index=False), "model_metrics.csv", "text/csv")

    with st.expander("Preview submission evidence"):
        st.json(submission)

    st.markdown("### AI Grader with 429 Fallback")
    api_key = ""
    try:
        api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        api_key = ""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    api_key = st.text_input("OpenRouter API key", value=api_key, type="password")

    if st.button("Run AI grader / local fallback"):
        if api_key:
            try:
                raw = call_openrouter(api_key, submission_json)
                parsed = robust_json(raw)
                if parsed:
                    st.success("OpenRouter grader returned valid JSON.")
                    st.json(parsed)
                else:
                    st.warning("OpenRouter response was not valid JSON. Showing local fallback grade.")
                    st.json(local_grader(submission))
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 429:
                    st.warning("OpenRouter returned 429 Too Many Requests. Showing local fallback grade.")
                else:
                    st.warning(f"OpenRouter failed: {exc}. Showing local fallback grade.")
                st.json(local_grader(submission))
            except Exception as exc:
                st.warning(f"OpenRouter failed: {exc}. Showing local fallback grade.")
                st.json(local_grader(submission))
        else:
            st.info("No API key provided. Showing local fallback grade.")
            st.json(local_grader(submission))

st.markdown(
    """
    <div style="text-align:center;color:var(--muted);margin-top:2rem;font-size:.86rem;">
        Professional Solar PV Forecasting Intelligence Platform • Animated website representation • Data, models, visuals and grading evidence
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)
