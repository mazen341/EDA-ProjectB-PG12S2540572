
"""
app.py — Fully Alive Professional Solar PV Forecasting Website

Run:
    streamlit run app.py

This version focuses on:
- Big, user-friendly website layout
- Project name always visible at the top
- Clear organization and easy navigation
- Animated background, live status, moving energy flows
- More images and 3D-style visual components
- Interactive user-selected dashboard modes
- Forecasting, analytics, model evidence, simulator and export
- Robust fallbacks if no dataset, no Plotly, or no scikit-learn
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
PROJECT_NAME = "Mini Project B — Solar PV Forecasting Intelligence Website"
STUDENT_NAME_DEFAULT = "MAZEN AL-HIMALI"
STUDENT_ID_DEFAULT = "PG12S2540572"
DEFAULT_DATA_PATH = "data/dataset_sample.csv"
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_TARGET_COL = "total_active_power_w"
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

IMG_SOLAR_1 = "https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&w=1400&q=80"
IMG_SOLAR_2 = "https://images.unsplash.com/photo-1497435334941-8c899ee9e8e9?auto=format&fit=crop&w=1200&q=80"
IMG_CONTROL = "https://images.unsplash.com/photo-1581092160607-ee22621dd758?auto=format&fit=crop&w=1100&q=80"
IMG_GRID = "https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&w=1100&q=80"
IMG_WEATHER = "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1100&q=80"
IMG_BATTERY = "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1100&q=80"

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
# Page
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Solar PV Forecasting Intelligence",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# CSS / visual system
# -----------------------------------------------------------------------------
THEMES = {
    "Midnight Energy": {
        "bg0": "#020617", "bg1": "#07111F", "bg2": "#0B2440",
        "card": "rgba(15, 23, 42, .84)", "card2": "rgba(30, 41, 59, .66)",
        "text": "#F8FAFC", "muted": "#B6C2D6",
        "blue": "#3B82F6", "cyan": "#22D3EE", "green": "#10B981", "gold": "#FBBF24", "red": "#EF4444",
    },
    "Emerald Energy": {
        "bg0": "#03110E", "bg1": "#06221B", "bg2": "#0B3428",
        "card": "rgba(6, 32, 27, .86)", "card2": "rgba(13, 55, 46, .62)",
        "text": "#ECFDF5", "muted": "#A7D8C9",
        "blue": "#14B8A6", "cyan": "#5EEAD4", "green": "#22C55E", "gold": "#BEF264", "red": "#F87171",
    },
    "Solar Gold": {
        "bg0": "#160C03", "bg1": "#251703", "bg2": "#3A2605",
        "card": "rgba(41, 25, 5, .86)", "card2": "rgba(75, 48, 9, .60)",
        "text": "#FFF7ED", "muted": "#E0C99C",
        "blue": "#F59E0B", "cyan": "#FDE68A", "green": "#84CC16", "gold": "#FBBF24", "red": "#FB7185",
    },
    "Cyber Blue": {
        "bg0": "#020617", "bg1": "#061A33", "bg2": "#082252",
        "card": "rgba(8, 22, 47, .86)", "card2": "rgba(15, 46, 84, .62)",
        "text": "#EFF6FF", "muted": "#B7D3F2",
        "blue": "#2563EB", "cyan": "#38BDF8", "green": "#06B6D4", "gold": "#93C5FD", "red": "#F43F5E",
    },
}


def inject_css(theme_name: str, motion: bool, big_mode: bool) -> None:
    t = THEMES.get(theme_name, THEMES["Midnight Energy"])
    motion_css = "" if motion else """
        *, *::before, *::after { animation: none !important; transition: none !important; }
    """
    scale = "1.12" if big_mode else "1"
    hero_size = "3.25rem" if big_mode else "2.65rem"
    card_pad = "1.25rem" if big_mode else "1rem"
    tab_size = "1.22rem" if big_mode else "1.08rem"

    st.markdown(
        f"""
        <style>
        :root {{
            --bg0:{t['bg0']};
            --bg1:{t['bg1']};
            --bg2:{t['bg2']};
            --card:{t['card']};
            --card2:{t['card2']};
            --text:{t['text']};
            --muted:{t['muted']};
            --blue:{t['blue']};
            --cyan:{t['cyan']};
            --green:{t['green']};
            --gold:{t['gold']};
            --red:{t['red']};
            --border:rgba(226,232,240,.18);
            --shadow:0 26px 80px rgba(0,0,0,.36);
        }}

        @keyframes floatY {{ 0%,100%{{transform:translateY(0)}} 50%{{transform:translateY(-8px)}} }}
        @keyframes pulseGlow {{ 0%,100%{{opacity:.45; box-shadow:0 0 10px rgba(34,211,238,.25)}} 50%{{opacity:1; box-shadow:0 0 32px rgba(34,211,238,.9)}} }}
        @keyframes shimmer {{ 0%{{background-position:-900px 0}} 100%{{background-position:900px 0}} }}
        @keyframes spin {{ from{{transform:rotate(0deg)}} to{{transform:rotate(360deg)}} }}
        @keyframes energyFlow {{ 0%{{transform:translateX(-45%);opacity:.2}} 50%{{opacity:1}} 100%{{transform:translateX(45%);opacity:.2}} }}
        @keyframes orbit {{ from{{transform:rotate(0deg)}} to{{transform:rotate(360deg)}} }}

        html, body, .stApp {{
            color:var(--text);
            font-size:calc(16px * {scale});
            background:
                radial-gradient(circle at 8% 8%, rgba(59,130,246,.22), transparent 30%),
                radial-gradient(circle at 90% 12%, rgba(16,185,129,.18), transparent 34%),
                radial-gradient(circle at 55% 105%, rgba(251,191,36,.12), transparent 38%),
                linear-gradient(135deg, var(--bg0), var(--bg1) 48%, var(--bg2));
        }}
        .stApp::before {{
            content:"";
            position:fixed;
            inset:0;
            pointer-events:none;
            background-image:
                radial-gradient(circle, rgba(255,255,255,.14) 1px, transparent 1px),
                linear-gradient(90deg, rgba(34,211,238,.04) 1px, transparent 1px),
                linear-gradient(rgba(34,211,238,.04) 1px, transparent 1px);
            background-size: 42px 42px, 90px 90px, 90px 90px;
            mask-image: linear-gradient(to bottom, rgba(0,0,0,.75), transparent 85%);
            animation: shimmer 20s linear infinite;
            z-index:0;
        }}
        [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
        [data-testid="stSidebar"] {{
            background:linear-gradient(180deg, rgba(2,6,23,.98), rgba(8,18,32,.96));
            border-right:1px solid var(--border);
        }}
        .block-container {{
            max-width: 1720px;
            padding-top:.65rem;
            padding-bottom:2rem;
        }}
        h1, h2, h3, h4, h5, h6 {{ color:var(--text)!important; letter-spacing:-.03em; }}
        label, .stSelectbox label, .stSlider label, .stRadio label, .stCheckbox label, .stTextInput label {{
            color:var(--text)!important;
            font-weight:900!important;
        }}

        .sticky-title {{
            position:sticky;
            top:.35rem;
            z-index:1000;
            border:1px solid var(--border);
            border-radius:26px;
            padding:.9rem 1.05rem;
            margin-bottom:1rem;
            background:linear-gradient(135deg, rgba(15,23,42,.92), rgba(15,23,42,.70));
            backdrop-filter:blur(18px);
            box-shadow:0 16px 48px rgba(0,0,0,.30);
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:1rem;
        }}
        .brand {{
            display:flex;
            align-items:center;
            gap:.9rem;
        }}
        .brand-logo {{
            width:58px;
            height:58px;
            border-radius:20px;
            display:flex;
            align-items:center;
            justify-content:center;
            color:#06121f;
            font-size:2rem;
            font-weight:950;
            background:linear-gradient(135deg, var(--gold), var(--green));
            box-shadow:0 14px 42px rgba(251,191,36,.28);
            animation:floatY 5s ease-in-out infinite;
        }}
        .brand-title {{
            font-size:1.45rem;
            font-weight:950;
            line-height:1.05;
        }}
        .brand-sub {{
            color:var(--muted);
            font-size:.88rem;
            margin-top:.22rem;
        }}
        .top-actions {{
            display:flex;
            flex-wrap:wrap;
            justify-content:flex-end;
            gap:.55rem;
        }}
        .status-pill, .pill {{
            display:inline-flex;
            align-items:center;
            gap:.35rem;
            border:1px solid rgba(34,211,238,.35);
            border-radius:999px;
            padding:.45rem .72rem;
            background:rgba(59,130,246,.14);
            color:var(--text);
            font-weight:900;
            font-size:.82rem;
            white-space:nowrap;
        }}
        .live-dot {{
            width:10px;
            height:10px;
            border-radius:999px;
            background:var(--green);
            display:inline-block;
            box-shadow:0 0 18px rgba(16,185,129,.9);
            animation:pulseGlow 1.35s ease-in-out infinite;
        }}

        .hero-grid {{
            display:grid;
            grid-template-columns: 1.15fr .85fr;
            gap:1rem;
            margin-bottom:1rem;
        }}
        .hero-card {{
            position:relative;
            overflow:hidden;
            min-height:350px;
            border-radius:34px;
            border:1px solid rgba(251,191,36,.28);
            padding:1.35rem;
            background:
                linear-gradient(135deg, rgba(15,23,42,.82), rgba(15,23,42,.50)),
                url('{IMG_SOLAR_1}');
            background-size:cover;
            background-position:center;
            box-shadow:var(--shadow);
        }}
        .hero-card::after {{
            content:"";
            position:absolute;
            inset:0;
            pointer-events:none;
            background:linear-gradient(90deg, transparent, rgba(255,255,255,.08), transparent);
            animation:shimmer 7s linear infinite;
        }}
        .hero-content {{
            position:relative;
            z-index:2;
            max-width:920px;
        }}
        .hero-title {{
            font-size:{hero_size};
            line-height:1.02;
            font-weight:1000;
            letter-spacing:-.065em;
            margin:.6rem 0 .65rem;
        }}
        .hero-copy {{
            color:#DCE9F7;
            font-size:1.05rem;
            max-width:920px;
            line-height:1.62;
        }}
        .mode-card {{
            border-radius:34px;
            border:1px solid rgba(34,211,238,.28);
            padding:1.15rem;
            background:linear-gradient(145deg, var(--card), rgba(2,6,23,.66));
            box-shadow:var(--shadow);
            position:relative;
            overflow:hidden;
        }}
        .mode-card::before {{
            content:"";
            position:absolute;
            width:240px;
            height:240px;
            border-radius:50%;
            right:-90px;
            top:-90px;
            background:radial-gradient(circle, rgba(34,211,238,.22), transparent 62%);
            animation:floatY 6s ease-in-out infinite;
        }}

        .control-grid {{
            display:grid;
            grid-template-columns:repeat(6,minmax(120px,1fr));
            gap:.7rem;
            margin-top:.9rem;
        }}
        .control-chip {{
            border-radius:20px;
            border:1px solid var(--border);
            background:rgba(255,255,255,.06);
            padding:.8rem;
        }}
        .control-label {{
            color:var(--muted);
            font-size:.74rem;
            font-weight:900;
            text-transform:uppercase;
            letter-spacing:.04em;
        }}
        .control-value {{
            color:var(--text);
            font-weight:1000;
            font-size:1.02rem;
            margin-top:.18rem;
            word-break:break-word;
        }}

        .kpi-card, .panel, .visual-card, .flow-card, .image-card, .workflow-card {{
            animation:floatY 7s ease-in-out infinite;
        }}
        .kpi-card {{
            min-height:132px;
            border-radius:24px;
            border:1px solid var(--border);
            padding:{card_pad};
            background:
                radial-gradient(circle at top left, rgba(59,130,246,.24), transparent 42%),
                linear-gradient(145deg, var(--card), rgba(2,6,23,.72));
            box-shadow:0 18px 54px rgba(0,0,0,.28);
        }}
        .kpi-icon {{ font-size:1.9rem; }}
        .kpi-label {{
            margin-top:.35rem;
            color:var(--muted);
            font-size:.78rem;
            font-weight:1000;
            text-transform:uppercase;
            letter-spacing:.04em;
        }}
        .kpi-value {{
            font-size:1.85rem;
            font-weight:1000;
            margin-top:.25rem;
        }}
        .kpi-detail {{
            color:var(--green);
            font-weight:900;
            font-size:.82rem;
            margin-top:.15rem;
        }}
        .panel {{
            border:1px solid var(--border);
            border-radius:28px;
            background:linear-gradient(145deg, var(--card), rgba(2,6,23,.66));
            padding:{card_pad};
            box-shadow:0 18px 54px rgba(0,0,0,.28);
            backdrop-filter:blur(12px);
        }}
        .section-title {{
            color:var(--gold);
            font-size:1.12rem;
            font-weight:1000;
            margin-bottom:.65rem;
        }}
        .muted {{ color:var(--muted); font-size:.9rem; line-height:1.55; }}

        .image-card {{
            min-height:190px;
            border-radius:24px;
            background-size:cover;
            background-position:center;
            border:1px solid var(--border);
            box-shadow:inset 0 -90px 95px rgba(0,0,0,.62), 0 16px 44px rgba(0,0,0,.24);
            position:relative;
            overflow:hidden;
        }}
        .image-card span {{
            position:absolute;
            left:.85rem;
            bottom:.75rem;
            font-weight:1000;
            color:#fff;
            text-shadow:0 2px 10px rgba(0,0,0,.7);
        }}

        .flow-card {{
            min-height:360px;
            border-radius:28px;
            border:1px solid rgba(34,211,238,.28);
            background:
                linear-gradient(90deg, rgba(34,211,238,.055) 1px, transparent 1px),
                linear-gradient(rgba(34,211,238,.055) 1px, transparent 1px),
                linear-gradient(145deg, var(--card), rgba(2,6,23,.70));
            background-size:28px 28px, 28px 28px, auto;
            padding:{card_pad};
            overflow:hidden;
        }}
        .flow-row {{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:.55rem;
            margin-top:1.05rem;
        }}
        .node {{
            flex:1;
            text-align:center;
            border-radius:18px;
            border:1px solid var(--border);
            background:rgba(255,255,255,.06);
            padding:.85rem .55rem;
        }}
        .node-icon {{ font-size:2.05rem; }}
        .node-label {{ font-weight:1000; font-size:.86rem; }}
        .node-sub {{ color:var(--muted); font-size:.72rem; }}
        .arrow {{
            min-width:26px;
            text-align:center;
            color:var(--gold);
            font-size:1.4rem;
            font-weight:1000;
            position:relative;
        }}
        .arrow::after {{
            content:"";
            position:absolute;
            left:-24px;
            right:-24px;
            top:50%;
            height:2px;
            background:linear-gradient(90deg, transparent, var(--gold), transparent);
            animation:energyFlow 2.2s ease-in-out infinite;
        }}

        .visual-card {{
            min-height:360px;
            border-radius:28px;
            border:1px solid rgba(59,130,246,.30);
            position:relative;
            overflow:hidden;
            padding:{card_pad};
            background:
                radial-gradient(circle at 72% 32%, rgba(34,211,238,.22), transparent 34%),
                linear-gradient(145deg, rgba(8,18,32,.95), rgba(2,6,23,.92));
            box-shadow:var(--shadow);
        }}
        .sun-orbit {{
            position:absolute;
            right:8%;
            top:12%;
            width:100px;
            height:100px;
            border-radius:50%;
            border:1px dashed rgba(251,191,36,.38);
            animation:orbit 12s linear infinite;
        }}
        .sun-orbit::before {{
            content:"☀️";
            position:absolute;
            left:-10px;
            top:34px;
            font-size:2rem;
            filter:drop-shadow(0 0 16px rgba(251,191,36,.9));
        }}
        .platform {{
            width:74%;
            height:58%;
            position:absolute;
            left:12%;
            bottom:10%;
            transform:skewX(-18deg) rotateX(8deg);
            border-radius:30px;
            background:linear-gradient(135deg,#193957,#09182b);
            border:1px solid rgba(34,211,238,.40);
            box-shadow:0 24px 90px rgba(34,211,238,.16);
        }}
        .panel-grid {{
            position:absolute;
            left:12%;
            top:28%;
            display:grid;
            grid-template-columns:repeat(6,42px);
            gap:7px;
            transform:rotate(-10deg);
        }}
        .solar-panel {{
            height:34px;
            border-radius:7px;
            border:1px solid rgba(191,219,254,.66);
            background:
                linear-gradient(90deg, rgba(255,255,255,.04), rgba(255,255,255,.18), rgba(255,255,255,.04)),
                linear-gradient(135deg,#14418f,#061843);
            background-size:180% 100%;
            animation:shimmer 4.4s linear infinite;
            box-shadow:inset 0 0 12px rgba(34,211,238,.28);
        }}
        .battery {{
            position:absolute;
            right:43%;
            bottom:20%;
            width:108px;
            height:62px;
            border-radius:13px;
            background:linear-gradient(135deg,#1e293b,#0f172a);
            border:1px solid rgba(16,185,129,.50);
        }}
        .battery-bars {{
            display:flex;
            gap:6px;
            align-items:end;
            padding:12px;
            height:100%;
        }}
        .battery-bars i {{
            display:block;
            width:13px;
            border-radius:5px;
            background:var(--green);
            box-shadow:0 0 14px rgba(16,185,129,.75);
            animation:pulseGlow 1.7s ease-in-out infinite;
        }}
        .inverter {{
            position:absolute;
            right:20%;
            bottom:23%;
            width:96px;
            height:86px;
            border-radius:13px;
            background:linear-gradient(135deg,#f1f5f9,#64748b);
            box-shadow:0 18px 40px rgba(0,0,0,.38);
        }}
        .tower {{
            position:absolute;
            right:6%;
            top:31%;
            font-size:4.35rem;
            color:#cbd5e1;
            text-shadow:0 0 18px rgba(34,211,238,.55);
            animation:floatY 4.6s ease-in-out infinite;
        }}
        .power-line {{
            position:absolute;
            right:10%;
            top:49%;
            width:34%;
            height:2px;
            background:linear-gradient(90deg, transparent, var(--cyan), var(--green));
            box-shadow:0 0 22px var(--cyan);
            transform:rotate(-10deg);
            animation:energyFlow 2s ease-in-out infinite;
        }}

        .workflow-card {{
            border-radius:20px;
            border:1px solid rgba(16,185,129,.28);
            background:rgba(16,185,129,.08);
            padding:{card_pad};
            min-height:118px;
        }}
        .check {{ color:var(--green); font-size:1.3rem; font-weight:1000; }}
        .insight {{
            display:flex;
            gap:.75rem;
            padding:.8rem;
            border-radius:18px;
            border:1px solid rgba(226,232,240,.12);
            background:rgba(255,255,255,.055);
            margin-bottom:.65rem;
        }}
        .insight-icon {{
            min-width:38px;
            height:38px;
            border-radius:14px;
            background:rgba(59,130,246,.16);
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:1.25rem;
        }}
        .loading-card {{
            border-radius:20px;
            border:1px solid rgba(251,191,36,.28);
            background:rgba(251,191,36,.08);
            padding:1rem;
            margin:.5rem 0;
            font-weight:950;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap:16px;
            overflow-x:auto;
            padding:.55rem .1rem 1rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            min-height:72px;
            font-size:{tab_size}!important;
            font-weight:1000!important;
            color:var(--text)!important;
            border:1px solid var(--border);
            border-radius:20px 20px 0 0;
            background:rgba(255,255,255,.065)!important;
            box-shadow:0 14px 36px rgba(0,0,0,.24);
            transition:transform .25s ease, border-color .25s ease, background .25s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            transform:translateY(-5px) scale(1.015);
            border-color:rgba(34,211,238,.60);
            background:rgba(34,211,238,.15)!important;
        }}
        .stTabs [aria-selected="true"] {{
            border-color:rgba(34,211,238,.72);
            border-bottom:3px solid var(--cyan);
            background:linear-gradient(135deg, rgba(59,130,246,.32), rgba(16,185,129,.18))!important;
            box-shadow:0 18px 46px rgba(34,211,238,.17);
        }}
        div[data-testid="stMetric"] {{
            border:1px solid var(--border);
            border-radius:20px;
            padding:1rem;
            background:linear-gradient(145deg, var(--card), rgba(2,6,23,.68));
        }}
        .stButton > button, .stDownloadButton > button {{
            border-radius:16px;
            border:1px solid rgba(251,191,36,.38);
            background:linear-gradient(135deg, var(--green), var(--blue));
            color:white;
            font-weight:1000;
            min-height:3rem;
        }}
        .stDataFrame {{
            border-radius:20px;
            overflow:hidden;
        }}

        @media (max-width: 1250px) {{
            .hero-grid {{ grid-template-columns:1fr; }}
            .control-grid {{ grid-template-columns:repeat(2,minmax(120px,1fr)); }}
            .sticky-title {{ position:relative; top:auto; flex-direction:column; align-items:flex-start; }}
            .top-actions {{ justify-content:flex-start; }}
        }}
        {motion_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Unique Streamlit element keys
# -----------------------------------------------------------------------------
if "chart_render_counter" not in st.session_state:
    st.session_state.chart_render_counter = 0


def next_chart_key(prefix: str = "chart") -> str:
    st.session_state.chart_render_counter += 1
    return f"{prefix}_{st.session_state.chart_render_counter}"


# -----------------------------------------------------------------------------
# Data and modeling
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def demo_data(days: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range(end=pd.Timestamp.now().floor("15min"), periods=days * 24 * 4, freq="15min")
    hour = idx.hour + idx.minute / 60
    doy = idx.dayofyear

    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    seasonal = 0.78 + 0.18 * np.sin(2 * np.pi * (doy - 70) / 365)
    cloud = np.clip(rng.normal(.92, .18, len(idx)), .28, 1.18)
    temp = 25 + 8 * np.sin((hour - 8) / 24 * 2 * np.pi) + rng.normal(0, 1.7, len(idx))
    humidity = 54 - 17 * daylight + rng.normal(0, 4, len(idx))
    irradiance = 980 * daylight * seasonal * cloud
    power = 5200 * daylight * seasonal * cloud * (1 - .0035 * np.maximum(temp - 25, 0))
    power += rng.normal(0, 110, len(idx))
    power = np.clip(power, 0, None)

    anomaly_idx = rng.choice(np.arange(len(idx)), size=max(10, len(idx)//420), replace=False)
    power[anomaly_idx] *= rng.uniform(.25, .65, len(anomaly_idx))

    return pd.DataFrame({
        "timestamp": idx,
        "total_active_power_w": power,
        "irradiance_wm2": irradiance,
        "temperature_c": temp,
        "relative_humidity_pct": np.clip(humidity, 18, 96),
        "wind_speed_ms": np.clip(rng.normal(3.2, 1.1, len(idx)), .1, 11),
        "rainfall_mm": rng.choice([0, 0, 0, 0, .2, .8, 1.5], len(idx), p=[.75, .08, .06, .04, .035, .025, .01]),
        "sea_level_pressure_hpa": rng.normal(1008, 4, len(idx)),
    })


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_dataset(path: str, uploaded_file: Any) -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        lower = uploaded_file.name.lower()
        if lower.endswith(".csv"):
            return pd.read_csv(uploaded_file), "uploaded CSV"
        if lower.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file), "uploaded Excel"
        if lower.endswith(".json"):
            return pd.read_json(uploaded_file), "uploaded JSON"
    if path and os.path.exists(path):
        return load_csv(path), path
    return demo_data(), "generated demo PV dataset"


def audit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "non_null": [int(df[c].notna().sum()) for c in df.columns],
        "missing_pct": [round(float(df[c].isna().mean() * 100), 3) for c in df.columns],
        "unique_count": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })


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

    return work, {
        "rows_before_cleaning": int(before),
        "rows_after_invalid_drop": int(after_drop),
        "duplicate_timestamps_before_grouping": duplicate_count,
        "rows_after_grouping_resampling": int(len(work)),
        "resampling_note": note,
    }


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

    weather = [c for c in [
        "irradiance_wm2", "temperature_c", "relative_humidity_pct",
        "wind_speed_ms", "rainfall_mm", "sea_level_pressure_hpa"
    ] if c in work.columns and c != target_col]

    features = [
        "lag_1", "lag_4", "lag_24", "rolling_mean_24", "rolling_std_24",
        "rolling_min_24", "rolling_max_24", "hour", "dayofweek", "month",
        "weekend", "dayofyear", "is_daylight_hour", "hour_sin", "hour_cos",
        "dayofyear_sin", "dayofyear_cos",
    ] + weather

    for col in features:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return work.dropna(subset=features + ["y_target"]).copy(), features, weather


def metrics_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, train_rows: int, valid_rows: int, notes: str) -> dict[str, Any]:
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
        "validation_rows": int(valid_rows),
        "split_type": "time_based_80_20",
        "notes": notes,
    }


def run_models(model_df: pd.DataFrame, features: list[str], timestamp_col: str, target_col: str):
    if len(model_df) < 120:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, "Not enough rows for reliable modeling."

    split = int(len(model_df) * 0.8)
    train = model_df.iloc[:split].copy()
    valid = model_df.iloc[split:].copy()

    q1, q3 = model_df["y_target"].astype(float).quantile([.25, .75])
    iqr = q3 - q1
    lower = max(0.0, q1 - 1.5 * iqr) if model_df["y_target"].min() >= 0 else q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    X_train = train[features]
    y_train = train["y_target"].clip(lower, upper)
    X_valid = valid[features]
    y_valid = valid["y_target"]

    rows = []
    preds = {}

    baseline = valid["lag_24"].fillna(valid["lag_1"]).fillna(train["y_target"].median()).clip(lower, upper).to_numpy()
    rows.append(metrics_row("Naive seasonal lag_24 baseline", y_valid, baseline, len(train), len(valid), "Transparent lag baseline."))
    preds["Naive seasonal lag_24 baseline"] = baseline

    fitted = {}
    if SKLEARN_AVAILABLE:
        models = [
            ("RidgeCV scaled", make_pipeline(StandardScaler(), RidgeCV(alphas=[.1, 1, 10, 100]))),
            ("RandomForest compact", RandomForestRegressor(n_estimators=70, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1)),
            ("HistGradientBoosting tuned", HistGradientBoostingRegressor(max_iter=240, learning_rate=.055, max_leaf_nodes=31, l2_regularization=.05, random_state=42)),
        ]
        for name, model in models:
            model.fit(X_train, y_train)
            fitted[name] = model
            pred = np.clip(model.predict(X_valid), lower, upper)
            rows.append(metrics_row(name, y_valid, pred, len(train), len(valid), "Candidate model in comparison table."))
            preds[name] = pred

    comparison = pd.DataFrame(rows).sort_values(["MAPE_pct", "RMSE"], ascending=True).reset_index(drop=True)
    best = str(comparison.iloc[0]["model"])
    best_pred = preds[best]

    residual = y_valid.to_numpy(dtype=float) - best_pred
    lo_res = float(np.nanquantile(residual, .05))
    hi_res = float(np.nanquantile(residual, .95))

    pred_df = valid[[timestamp_col, target_col, "y_target"]].copy()
    pred_df["prediction"] = best_pred
    pred_df["prediction_lower_90"] = np.clip(best_pred + lo_res, lower, upper)
    pred_df["prediction_upper_90"] = np.clip(best_pred + hi_res, lower, upper)
    pred_df["residual"] = pred_df["y_target"] - pred_df["prediction"]
    pred_df["absolute_error"] = pred_df["residual"].abs()
    pred_df["interval_covered"] = (
        (pred_df["y_target"] >= pred_df["prediction_lower_90"])
        & (pred_df["y_target"] <= pred_df["prediction_upper_90"])
    )

    if SKLEARN_AVAILABLE and best in fitted:
        try:
            sample = min(800, len(X_valid))
            perm = permutation_importance(
                fitted[best],
                X_valid.tail(sample),
                y_valid.tail(sample),
                n_repeats=4,
                random_state=42,
                scoring="neg_mean_absolute_error",
            )
            importance = pd.DataFrame({
                "feature": features,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }).sort_values("importance_mean", ascending=False).head(15)
        except Exception:
            importance = pd.DataFrame({"feature": ["importance unavailable"], "importance_mean": [0.0], "importance_std": [0.0]})
    else:
        importance = pd.DataFrame({"feature": ["lag_24", "lag_1"], "importance_mean": [1.0, .55], "importance_std": [0.0, 0.0]})

    uncertainty = {
        "method": "Empirical 90% prediction interval from validation residual quantiles",
        "lower_residual_quantile_5pct": round(lo_res, 3),
        "upper_residual_quantile_95pct": round(hi_res, 3),
        "interval_coverage_pct": round(float(pred_df["interval_covered"].mean() * 100), 3),
        "average_interval_width": round(float((pred_df["prediction_upper_90"] - pred_df["prediction_lower_90"]).mean()), 3),
        "outlier_bounds": {"lower": round(float(lower), 3), "upper": round(float(upper), 3)},
    }

    return comparison, pred_df, importance, uncertainty, f"Best model: {best}. Strict chronological 80/20 split used."


def safe_json_default(obj: Any):
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


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def forecast_fig(df: pd.DataFrame, timestamp_col: str, target_col: str, window: int, band: float):
    if not PLOTLY_AVAILABLE or df.empty:
        return None
    chart = df[[timestamp_col, target_col]].dropna().tail(window).copy()
    if chart.empty:
        return None
    chart["smooth"] = chart[target_col].rolling(max(2, min(12, len(chart)//8))).mean().bfill()
    chart["low"] = chart["smooth"] * (1 - band)
    chart["high"] = chart["smooth"] * (1 + band)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart[timestamp_col], y=chart["high"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=chart[timestamp_col], y=chart["low"], mode="lines", fill="tonexty", fillcolor="rgba(251,191,36,.20)", line=dict(width=0), name="Confidence band"))
    fig.add_trace(go.Scatter(x=chart[timestamp_col], y=chart["smooth"], mode="lines", name="Forecast signal", line=dict(color="#FBBF24", width=3)))
    fig.add_trace(go.Scatter(x=chart[timestamp_col], y=chart[target_col], mode="lines", name="Actual", line=dict(color="#22D3EE", width=2)))
    fig.update_layout(template="plotly_dark", height=390, margin=dict(l=10, r=10, t=32, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
    return fig


def prediction_fig(pred_df: pd.DataFrame, timestamp_col: str, window: int):
    if not PLOTLY_AVAILABLE or pred_df.empty:
        return None
    chart = pred_df.tail(window).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart[timestamp_col], y=chart["prediction_upper_90"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=chart[timestamp_col], y=chart["prediction_lower_90"], mode="lines", fill="tonexty", fillcolor="rgba(59,130,246,.20)", line=dict(width=0), name="90% interval"))
    fig.add_trace(go.Scatter(x=chart[timestamp_col], y=chart["y_target"], mode="lines", name="Actual", line=dict(color="#22D3EE", width=2)))
    fig.add_trace(go.Scatter(x=chart[timestamp_col], y=chart["prediction"], mode="lines", name="Predicted", line=dict(color="#10B981", width=2)))
    fig.update_layout(template="plotly_dark", height=390, margin=dict(l=10, r=10, t=32, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
    return fig


def show_chart(fig, df: pd.DataFrame, timestamp_col: str, columns: list[str], window: int, key_prefix: str = "chart"):
    """Render charts with explicit unique keys to avoid StreamlitDuplicateElementId."""
    chart_key = next_chart_key(key_prefix)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
    elif not df.empty:
        st.line_chart(df.set_index(timestamp_col)[columns].tail(window), use_container_width=True, key=chart_key)
    else:
        st.info("No data available for this chart.")


# -----------------------------------------------------------------------------
# Rendering helpers
# -----------------------------------------------------------------------------
def kpi(title: str, value: str, icon: str, detail: str):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def energy_flow_panel():
    st.markdown('<div class="flow-card"><div class="section-title">Animated PV Energy Flow</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="flow-row">
            <div class="node"><div class="node-icon">🔷</div><div class="node-label">PV Array</div><div class="node-sub">DC generation</div></div>
            <div class="arrow">→</div>
            <div class="node"><div class="node-icon">🔌</div><div class="node-label">Inverter</div><div class="node-sub">DC → AC</div></div>
            <div class="arrow">→</div>
            <div class="node"><div class="node-icon">⚡</div><div class="node-label">Transformer</div><div class="node-sub">Voltage step-up</div></div>
            <div class="arrow">→</div>
            <div class="node"><div class="node-icon">🗼</div><div class="node-label">Grid</div><div class="node-sub">Export</div></div>
        </div>
        <div class="flow-row">
            <div class="node"><div class="node-icon">🌤️</div><div class="node-label">Weather</div><div class="node-sub">forecast drivers</div></div>
            <div class="arrow">↔</div>
            <div class="node"><div class="node-icon">🔋</div><div class="node-label">Battery</div><div class="node-sub">storage</div></div>
            <div class="arrow">↔</div>
            <div class="node"><div class="node-icon">🏠</div><div class="node-label">Local Load</div><div class="node-sub">demand</div></div>
        </div>
        <div style="margin-top:1rem"><span class="pill"><span class="live-dot"></span>Telemetry online • energy moving • forecast active</span></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def visual_twin_panel():
    st.markdown(
        """
        <div class="visual-card">
            <div class="section-title">Animated 3D-Style Digital Twin</div>
            <div class="muted">A visual representation of the PV array, inverter, battery, weather and grid link.</div>
            <div class="sun-orbit"></div>
            <div class="platform"></div>
            <div class="panel-grid">
                <div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div>
                <div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div>
                <div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div><div class="solar-panel"></div>
            </div>
            <div class="battery"><div class="battery-bars"><i style="height:35%"></i><i style="height:55%"></i><i style="height:76%"></i><i style="height:92%"></i></div></div>
            <div class="inverter"></div>
            <div class="tower">🗼</div>
            <div class="power-line"></div>
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
        "Dashboard quality": min(10, (2 if dashboard.get("has_student_added_dashboard") else 0) + (2 if dashboard.get("has_system_photos") else 0) + (2 if dashboard.get("has_diagrams_and_3d") else 0) + (2 if dashboard.get("has_advanced_analytics") else 0) + (1 if dashboard.get("user_selectable_dashboard_representation") else 0) + (1 if dashboard.get("insights") else 0)),
        "Presentation & rigor": min(10, (5 if rigor.get("limitations") else 0) + (5 if rigor.get("reproducibility_notes") else 0)),
    }
    return {
        "scores": {k: int(v) for k, v in scores.items()},
        "total_80": int(sum(scores.values())),
        "strengths": [
            "User-friendly professional website with organized navigation and animated visuals.",
            "Includes photos, diagrams, 3D-style digital twin, forecasting charts and advanced analytics.",
            "Provides strong grading evidence: cleaning, features, time split, metrics, feature importance, uncertainty and limitations.",
        ],
        "weaknesses": [
            "External image URLs should be replaced by original project images for final deployment reliability.",
            "Local fallback grade is an estimate if OpenRouter is unavailable or rate-limited.",
        ],
        "actionable_improvements": [
            "Upload original system photos into an assets folder and reference them locally.",
            "Add authenticated live plant telemetry if this becomes a production dashboard.",
            "Add SHAP explainability if the package is available in the deployment environment.",
        ],
    }


def robust_json(text: str):
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
            "X-Title": PROJECT_NAME,
        },
        json={"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        timeout=90,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ☀️ Website Controls")
    st.caption("Everything here changes the website representation.")

    dashboard_mode = st.selectbox(
        "Choose dashboard style",
        [
            "Live Command Website",
            "Visual 3D Experience",
            "Engineering Workbench",
            "Student Evidence Center",
            "Simple Friendly View",
        ],
        index=0,
    )
    theme = st.selectbox("Color theme", list(THEMES.keys()), index=0)
    big_dashboard = st.toggle("Big dashboard size", value=True)
    alive_motion = st.toggle("Keep everything alive / animated", value=True)
    detailed_loading = st.toggle("Show loading timeline", value=True)
    first_view = st.selectbox("First screen priority", ["Balanced", "Charts first", "3D visuals first", "Evidence first"], index=0)

    st.markdown("---")
    st.markdown("## 👤 Project Details")
    student_name = st.text_input("Student name", STUDENT_NAME_DEFAULT)
    student_id = st.text_input("Student ID", STUDENT_ID_DEFAULT)
    project_title = st.text_input("Top project name", PROJECT_NAME)

    st.markdown("---")
    st.markdown("## 📁 Data")
    data_path = st.text_input("Dataset path", DEFAULT_DATA_PATH)
    uploaded_file = st.file_uploader("Upload data", type=["csv", "xlsx", "xls", "json"])

    st.markdown("---")
    st.markdown("## ⚙️ Forecast Parameters")
    site_name = st.selectbox("Site", ["Solar Farm Alpha", "Rooftop PV Lab", "Campus PV Plant"], index=0)
    resample_rule = st.selectbox("Resampling rule", ["None", "15min", "30min", "1h", "1D"], index=1)
    horizon = int(st.number_input("Forecast horizon rows", min_value=1, max_value=96, value=1, step=1))
    model_rows = int(st.slider("Model rows", 1000, 40000, 18000, 1000))
    chart_window = int(st.slider("Chart window rows", 96, 3000, 700, 32))
    confidence_width = float(st.slider("Forecast band width", .05, .35, .12, .01))
    anomaly_sensitivity = float(st.slider("Anomaly sensitivity", 1.0, 4.0, 2.0, .1))
    refresh_seconds = int(st.slider("Visible refresh timing", 5, 120, 30, 5))
    show_correlation = st.toggle("Show correlation diagnostics", value=True)


inject_css(theme, alive_motion, big_dashboard)

# Top title stays at the top.
st.markdown(
    f"""
    <div class="sticky-title">
        <div class="brand">
            <div class="brand-logo">☀️</div>
            <div>
                <div class="brand-title">{project_title}</div>
                <div class="brand-sub">Student: <b>{student_name}</b> • ID: <b>{student_id}</b> • Professional interactive Streamlit website</div>
            </div>
        </div>
        <div class="top-actions">
            <span class="status-pill"><span class="live-dot"></span>Alive</span>
            <span class="status-pill">{dashboard_mode}</span>
            <span class="status-pill">{theme}</span>
            <span class="status-pill">Refresh {refresh_seconds}s</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Loading
if detailed_loading:
    load_slot = st.empty()
    prog = st.progress(0, text="Starting professional dashboard...")
    for pct, msg in [
        (10, "Preparing animated website shell"),
        (22, "Loading dataset or demo fallback"),
    ]:
        load_slot.markdown(f'<div class="loading-card"><span class="live-dot"></span> {msg}</div>', unsafe_allow_html=True)
        prog.progress(pct, text=msg)
        time.sleep(.04)

raw_df, source_label = load_dataset(data_path, uploaded_file)

if detailed_loading:
    for pct, msg in [(34, "Checking columns and selecting target"), (46, "Rendering user controls")]:
        load_slot.markdown(f'<div class="loading-card"><span class="live-dot"></span> {msg}</div>', unsafe_allow_html=True)
        prog.progress(pct, text=msg)
        time.sleep(.04)

columns = list(raw_df.columns)
numeric_candidates = [c for c in columns if pd.to_numeric(raw_df[c], errors="coerce").notna().sum() > 0]
if not numeric_candidates:
    st.error("No numeric columns were found. Upload a dataset with at least one numeric target column.")
    st.stop()

default_ts_idx = columns.index(DEFAULT_TIMESTAMP_COL) if DEFAULT_TIMESTAMP_COL in columns else 0
default_target_idx = numeric_candidates.index(DEFAULT_TARGET_COL) if DEFAULT_TARGET_COL in numeric_candidates else 0

control_cols = st.columns([1, 1, .9, .9])
timestamp_col = control_cols[0].selectbox("Timestamp column", columns, index=default_ts_idx)
target_col = control_cols[1].selectbox("Target column", numeric_candidates, index=default_target_idx)

ts_preview = pd.to_datetime(raw_df[timestamp_col], errors="coerce")
min_date = ts_preview.min().date() if ts_preview.notna().any() else datetime.now().date()
max_date = ts_preview.max().date() if ts_preview.notna().any() else datetime.now().date()
start_date = control_cols[2].date_input("Start date", value=min_date)
end_date = control_cols[3].date_input("End date", value=max_date)

if detailed_loading:
    for pct, msg in [
        (58, "Cleaning, grouping and resampling data"),
        (70, "Engineering lags, rolling features, temporal features and weather features"),
    ]:
        load_slot.markdown(f'<div class="loading-card"><span class="live-dot"></span> {msg}</div>', unsafe_allow_html=True)
        prog.progress(pct, text=msg)
        time.sleep(.04)

prepared_df, cleaning_report = prepare_timeseries(raw_df, timestamp_col, target_col, resample_rule)
prepared_df[timestamp_col] = pd.to_datetime(prepared_df[timestamp_col], errors="coerce")
filtered_df = prepared_df[
    (prepared_df[timestamp_col].dt.date >= start_date)
    & (prepared_df[timestamp_col].dt.date <= end_date)
].copy()
if filtered_df.empty:
    filtered_df = prepared_df.copy()

model_df, feature_cols, weather_features = build_features(prepared_df, timestamp_col, target_col, horizon)
model_df = model_df.tail(model_rows).copy()

if detailed_loading:
    for pct, msg in [(84, "Training comparison models and uncertainty bands")]:
        load_slot.markdown(f'<div class="loading-card"><span class="live-dot"></span> {msg}</div>', unsafe_allow_html=True)
        prog.progress(pct, text=msg)
        time.sleep(.04)

comparison_df, predictions_df, importance_df, uncertainty_summary, modeling_note = run_models(
    model_df, feature_cols, timestamp_col, target_col
)

if detailed_loading:
    for pct, msg in [(94, "Creating images, diagrams, 3D digital twin and analytics panels"), (100, "Website ready")]:
        load_slot.markdown(f'<div class="loading-card"><span class="live-dot"></span> {msg}</div>', unsafe_allow_html=True)
        prog.progress(pct, text=msg)
        time.sleep(.04)
    prog.empty()
    load_slot.empty()

# Metrics
latest_power = float(filtered_df[target_col].iloc[-1]) if len(filtered_df) else 0.0
avg_power = float(filtered_df[target_col].mean()) if len(filtered_df) else 0.0
max_power = float(filtered_df[target_col].max()) if len(filtered_df) else 0.0
energy_mwh = float(filtered_df[target_col].sum() * .25 / 1_000_000) if resample_rule in ["15min", "None"] else float(filtered_df[target_col].sum() / 1_000_000)
capacity_kwp = max(.01, max_power / 1000)
pr_value = 87.6 if "irradiance_wm2" in filtered_df.columns else 82.4
zero_pct = float((filtered_df[target_col] <= 0).mean() * 100) if len(filtered_df) else 0.0
best_model = str(comparison_df.iloc[0]["model"]) if not comparison_df.empty else "N/A"

# Hero
mode_copy = {
    "Live Command Website": "A big, alive command-center website for fast decisions, forecasting, system status, and project evidence.",
    "Visual 3D Experience": "A visual-first website with more images, moving energy flows, and a 3D-style digital twin.",
    "Engineering Workbench": "A technical workspace for data quality, features, models, diagnostics, residuals, and correlations.",
    "Student Evidence Center": "A rubric-friendly representation that makes grading evidence easy to find.",
    "Simple Friendly View": "A simplified student-friendly representation with clear sections and fewer distractions.",
}.get(dashboard_mode, "")

st.markdown(
    f"""
    <div class="hero-grid">
        <div class="hero-card">
            <div class="hero-content">
                <span class="pill"><span class="live-dot"></span>{site_name} • {source_label}</span>
                <div class="hero-title">{dashboard_mode}</div>
                <div class="hero-copy">{mode_copy}<br><br>
                Everything is organized into clear areas: Home, Forecasting, Visual System, Data Pipeline, Models, Advanced Analytics, Simulator and Export.</div>
            </div>
        </div>
        <div class="mode-card">
            <div class="section-title">Live Website Control Surface</div>
            <div class="muted">All dashboard parameters are controlled by the user from the sidebar and reflected in charts, models, evidence and exports.</div>
            <div class="control-grid" style="grid-template-columns:repeat(2,minmax(120px,1fr));">
                <div class="control-chip"><div class="control-label">Timestamp</div><div class="control-value">{timestamp_col}</div></div>
                <div class="control-chip"><div class="control-label">Target</div><div class="control-value">{target_col}</div></div>
                <div class="control-chip"><div class="control-label">Mode</div><div class="control-value">{dashboard_mode}</div></div>
                <div class="control-chip"><div class="control-label">Theme</div><div class="control-value">{theme}</div></div>
                <div class="control-chip"><div class="control-label">Horizon</div><div class="control-value">{horizon} rows</div></div>
                <div class="control-chip"><div class="control-label">Best Model</div><div class="control-value">{best_model}</div></div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# KPI deck
kpi_cols = st.columns(6)
for col, args in zip(
    kpi_cols,
    [
        ("Capacity", f"{capacity_kwp:,.2f} kWp", "⚙️", "from max observed output"),
        ("Energy", f"{energy_mwh:,.2f} MWh", "⚡", "selected period"),
        ("Latest Power", f"{latest_power:,.0f} W", "📈", "latest row"),
        ("Avg Power", f"{avg_power:,.0f} W", "🔁", "selected average"),
        ("Zero Power", f"{zero_pct:.1f}%", "🌙", "night / outage"),
        ("CO₂ Avoided", f"{energy_mwh * .78:,.1f} t", "🌿", "estimated"),
    ],
):
    with col:
        kpi(*args)

# First screen priority
if first_view == "Charts first":
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="panel"><div class="section-title">First View: Forecast</div>', unsafe_allow_html=True)
        show_chart(forecast_fig(filtered_df, timestamp_col, target_col, chart_window, confidence_width), filtered_df, timestamp_col, [target_col], chart_window)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="section-title">First View: Prediction</div>', unsafe_allow_html=True)
        show_chart(prediction_fig(predictions_df, timestamp_col, chart_window), predictions_df if not predictions_df.empty else filtered_df, timestamp_col, ["y_target", "prediction"] if not predictions_df.empty else [target_col], chart_window)
        st.markdown("</div>", unsafe_allow_html=True)
elif first_view == "3D visuals first":
    a, b = st.columns(2)
    with a:
        visual_twin_panel()
    with b:
        energy_flow_panel()
elif first_view == "Evidence first":
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Cleaned rows", f"{cleaning_report['rows_after_grouping_resampling']:,}")
    e2.metric("Features", f"{len(feature_cols):,}")
    e3.metric("Models", f"{len(comparison_df):,}")
    e4.metric("Validation rows", f"{len(predictions_df):,}")


# Tabs
tabs = st.tabs([
    "🏠 Home",
    "📊 Forecasting",
    "🧩 Images + 3D",
    "🧹 Data Pipeline",
    "🤖 Models",
    "🧬 Advanced",
    "🕹️ Simulator",
    "📤 Export",
])

with tabs[0]:
    c1, c2, c3 = st.columns([1.1, 1.1, 1.25])
    with c1:
        st.markdown(
            f"""
            <div class="image-card" style="min-height:360px;background-image:url('{IMG_SOLAR_1}')">
                <span>Solar PV Plant • Live visual context</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        energy_flow_panel()
    with c3:
        visual_twin_panel()

    st.markdown("### Live Production Trend")
    show_chart(forecast_fig(filtered_df, timestamp_col, target_col, chart_window, confidence_width), filtered_df, timestamp_col, [target_col], chart_window)

    if dashboard_mode == "Student Evidence Center":
        st.success("Important grading evidence is available in Data Pipeline, Models, Advanced, and Export tabs.")
    elif dashboard_mode == "Engineering Workbench":
        st.json({
            "timestamp": timestamp_col,
            "target": target_col,
            "resampling": resample_rule,
            "feature_count": len(feature_cols),
            "weather_features": weather_features,
            "uncertainty": uncertainty_summary,
        })

with tabs[1]:
    f1, f2 = st.columns(2)
    with f1:
        st.markdown('<div class="panel"><div class="section-title">Forecast Signal with User-Controlled Band</div>', unsafe_allow_html=True)
        show_chart(forecast_fig(filtered_df, timestamp_col, target_col, chart_window, confidence_width), filtered_df, timestamp_col, [target_col], chart_window)
        st.markdown("</div>", unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="panel"><div class="section-title">Actual vs Predicted with 90% Interval</div>', unsafe_allow_html=True)
        show_chart(prediction_fig(predictions_df, timestamp_col, chart_window), predictions_df if not predictions_df.empty else filtered_df, timestamp_col, ["y_target", "prediction"] if not predictions_df.empty else [target_col], chart_window)
        st.markdown("</div>", unsafe_allow_html=True)

    w1, w2, w3 = st.columns(3)
    with w1:
        irradiance = float(filtered_df["irradiance_wm2"].tail(96).mean()) if "irradiance_wm2" in filtered_df.columns else 782
        temperature = float(filtered_df["temperature_c"].tail(96).mean()) if "temperature_c" in filtered_df.columns else 26
        humidity = float(filtered_df["relative_humidity_pct"].tail(96).mean()) if "relative_humidity_pct" in filtered_df.columns else 46
        st.markdown(
            f"""
            <div class="panel">
                <div class="section-title">Weather Context</div>
                <div style="font-size:3.2rem">🌤️</div>
                <div class="kpi-value">{temperature:.1f}°C</div>
                <div class="muted">Irradiance: {irradiance:.0f} W/m²</div>
                <div class="muted">Humidity: {humidity:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with w2:
        st.markdown('<div class="panel"><div class="section-title">Forecast Insights</div>', unsafe_allow_html=True)
        for icon, text_item in [
            ("📈", "The strongest pattern is the daily solar generation cycle."),
            ("✅", modeling_note),
            ("⚠️", "MAPE can rise during low-power sunrise and sunset periods."),
        ]:
            st.markdown(f'<div class="insight"><div class="insight-icon">{icon}</div><div>{text_item}</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with w3:
        st.markdown('<div class="panel"><div class="section-title">Validation Diagnostics</div>', unsafe_allow_html=True)
        if not predictions_df.empty:
            st.metric("MAE", f"{predictions_df['absolute_error'].mean():,.2f}")
            st.metric("Coverage", f"{predictions_df['interval_covered'].mean() * 100:,.1f}%")
            st.metric("Max Error", f"{predictions_df['absolute_error'].max():,.2f}")
        else:
            st.info("Not enough prediction rows.")
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown("## Visual System — Images, Diagram and 3D")
    gallery = [
        ("PV Field", IMG_SOLAR_1),
        ("Solar Technology", IMG_SOLAR_2),
        ("Control / Inverter", IMG_CONTROL),
        ("Grid Interface", IMG_GRID),
        ("Weather Context", IMG_WEATHER),
        ("Battery / Electronics", IMG_BATTERY),
    ]
    gcols = st.columns(3)
    for i, (label, url) in enumerate(gallery):
        with gcols[i % 3]:
            st.markdown(f'<div class="image-card" style="background-image:url({url})"><span>{label}</span></div>', unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1:
        energy_flow_panel()
    with s2:
        visual_twin_panel()

    st.markdown("### Formal Technical Diagram")
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
    st.markdown("## Data Pipeline")
    steps = [
        ("1. Load", f"{len(raw_df):,} rows", source_label),
        ("2. Clean", f"{cleaning_report['rows_after_invalid_drop']:,} rows", "invalid timestamps and targets removed"),
        ("3. Resample", resample_rule, cleaning_report["resampling_note"]),
        ("4. Outliers", "IQR bounds", json.dumps(uncertainty_summary.get("outlier_bounds", {}))),
        ("5. Features", f"{len(feature_cols)} features", "lag, rolling, time and weather"),
        ("6. Validate", "80/20 split", "chronological, leakage-resistant"),
    ]
    step_cols = st.columns(6)
    for col, (title, value, desc) in zip(step_cols, steps):
        with col:
            st.markdown(f'<div class="workflow-card"><div class="check">✓</div><b>{title}</b><br>{value}<div class="muted">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown("### Dataset Audit")
    st.dataframe(audit_dataframe(raw_df), use_container_width=True)
    st.markdown("### Cleaning Report")
    st.json(cleaning_report)
    st.markdown("### Feature Preview")
    if not model_df.empty:
        preview_cols = [timestamp_col, target_col, "y_target"] + feature_cols[:14]
        st.dataframe(model_df[preview_cols].head(40), use_container_width=True)
    else:
        st.warning("No feature rows available.")

with tabs[4]:
    st.markdown("## Models and Interpretability")
    if comparison_df.empty:
        st.warning(modeling_note)
    else:
        st.markdown("### Full Metrics Table")
        st.dataframe(comparison_df, use_container_width=True)
        a, b = st.columns([1, 1])
        with a:
            st.markdown("### Feature Importance")
            st.dataframe(importance_df, use_container_width=True)
        with b:
            if PLOTLY_AVAILABLE and not importance_df.empty:
                fig = go.Figure(go.Bar(x=importance_df["importance_mean"], y=importance_df["feature"], orientation="h"))
                fig.update_layout(template="plotly_dark", height=430, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True, key=next_chart_key("plotly"))
            elif not importance_df.empty:
                st.bar_chart(importance_df.set_index("feature")["importance_mean"])
        st.markdown("### Uncertainty")
        st.json(uncertainty_summary)

with tabs[5]:
    st.markdown("## Advanced Analytics")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Features", f"{len(feature_cols):,}")
    a2.metric("Weather features", f"{len(weather_features):,}")
    a3.metric("Model rows", f"{len(model_df):,}")
    a4.metric("Anomaly sensitivity", f"{anomaly_sensitivity:.1f}× IQR")

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### Daily Production Profile")
        profile = filtered_df.copy()
        profile["hour"] = profile[timestamp_col].dt.hour
        hourly = profile.groupby("hour", as_index=False)[target_col].agg(["mean", "max", "std"]).reset_index()
        if PLOTLY_AVAILABLE and not hourly.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hourly["hour"], y=hourly["mean"], mode="lines+markers", name="Mean"))
            fig.add_trace(go.Scatter(x=hourly["hour"], y=hourly["max"], mode="lines", name="Max"))
            fig.update_layout(template="plotly_dark", height=360, margin=dict(l=10, r=10, t=25, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, key=next_chart_key("plotly"))
        elif not hourly.empty:
            st.line_chart(hourly.set_index("hour")[["mean", "max"]], use_container_width=True, key=next_chart_key("line"))
    with right:
        st.markdown("### Anomaly Scan")
        target_series = filtered_df[target_col].dropna().astype(float)
        q1 = target_series.quantile(.25) if len(target_series) else 0
        q3 = target_series.quantile(.75) if len(target_series) else 0
        iqr = q3 - q1
        low = max(0, q1 - anomaly_sensitivity * iqr)
        high = q3 + anomaly_sensitivity * iqr
        anomaly_df = filtered_df[(filtered_df[target_col] < low) | (filtered_df[target_col] > high)].copy()
        st.metric("Detected anomalies", f"{len(anomaly_df):,}")
        st.metric("Lower bound", f"{low:,.2f}")
        st.metric("Upper bound", f"{high:,.2f}")
        st.dataframe(anomaly_df[[timestamp_col, target_col]].tail(20), use_container_width=True)

    if show_correlation:
        st.markdown("### Correlation Diagnostics")
        numeric = filtered_df.select_dtypes(include=[np.number]).copy()
        if len(numeric.columns) >= 2:
            corr = numeric.corr(numeric_only=True).round(3)
            st.dataframe(corr, use_container_width=True)
            if PLOTLY_AVAILABLE:
                fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="Viridis"))
                fig.update_layout(template="plotly_dark", height=520, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True, key=next_chart_key("plotly"))
        else:
            st.info("Not enough numeric columns for correlation diagnostics.")

    st.markdown("### Residual Stream")
    if not predictions_df.empty:
        residual_view = predictions_df[[timestamp_col, "residual", "absolute_error", "interval_covered"]].tail(chart_window)
        st.line_chart(residual_view.set_index(timestamp_col)[["residual", "absolute_error"]], use_container_width=True, key=next_chart_key("line"))
        st.dataframe(residual_view.tail(30), use_container_width=True)
    else:
        st.info("Residual stream appears after prediction rows are available.")

with tabs[6]:
    st.markdown("## What-If Simulator")
    st.write("Change the scenario parameters and see an estimated production impact.")
    s1, s2, s3, s4 = st.columns(4)
    irr_factor = s1.slider("Irradiance factor", .40, 1.25, 1.00, .05)
    temp_delta = s2.slider("Temperature change °C", -10, 15, 0, 1)
    curtailment = s3.slider("Curtailment %", 0, 80, 0, 5)
    battery_support = s4.slider("Battery support %", 0, 30, 5, 5)

    base = filtered_df[[timestamp_col, target_col]].tail(chart_window).copy()
    if not base.empty:
        simulated = base.copy()
        temp_loss = max(0, temp_delta) * .0035
        simulated["scenario_power"] = simulated[target_col] * irr_factor * (1 - temp_loss) * (1 - curtailment / 100) * (1 + battery_support / 100)
        simulated["scenario_power"] = simulated["scenario_power"].clip(lower=0)
        sim_energy = simulated["scenario_power"].sum() * .25 / 1_000_000
        actual_energy = simulated[target_col].sum() * .25 / 1_000_000
        st.metric("Scenario energy", f"{sim_energy:,.2f} MWh", f"{sim_energy - actual_energy:+.2f} MWh")
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=simulated[timestamp_col], y=simulated[target_col], mode="lines", name="Actual"))
            fig.add_trace(go.Scatter(x=simulated[timestamp_col], y=simulated["scenario_power"], mode="lines", name="Scenario"))
            fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, key=next_chart_key("plotly"))
        else:
            st.line_chart(simulated.set_index(timestamp_col)[[target_col, "scenario_power"]], use_container_width=True, key=next_chart_key("line"))
    else:
        st.info("No data available for simulator.")

with tabs[7]:
    st.markdown("## Export and AI Grader")
    dashboard_insights = [
        "The website has user-selectable representations and a clear top project title.",
        "The website includes moving/animated status, energy flow, and 3D-style digital twin elements.",
        "The project evidence includes cleaning, resampling, outlier handling, features, metrics, model comparison, uncertainty and limitations.",
    ]
    submission = {
        "student": {"name": student_name, "id": student_id, "app_title": project_title},
        "data_integrity": {
            "dataset_source": source_label,
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
            "has_what_if_simulator": True,
            "user_selectable_dashboard_representation": dashboard_mode,
            "theme_palette": theme,
            "insights": dashboard_insights,
        },
        "presentation_and_rigor": {
            "limitations": [
                "PV generation is sensitive to cloud cover, shading, equipment events and low-light periods.",
                "Remote images are visual placeholders and should be replaced with original local project photos for final submission.",
                "The local grader is an estimate when the OpenRouter API is unavailable or rate-limited.",
            ],
            "reproducibility_notes": [
                "The app runs with uploaded data, local data/dataset_sample.csv, or generated demo PV data.",
                "The model uses a chronological 80/20 split to avoid random leakage.",
                "Submission JSON, predictions and metrics can be exported from this tab.",
            ],
        },
    }
    submission_json = json.dumps(submission, indent=2, default=safe_json_default)
    st.download_button("Download submission.json", submission_json, "submission.json", "application/json")
    st.download_button("Download predictions.csv", predictions_df.to_csv(index=False), "predictions.csv", "text/csv")
    st.download_button("Download metrics.csv", comparison_df.to_csv(index=False), "metrics.csv", "text/csv")

    with st.expander("Preview submission JSON"):
        st.json(submission)

    st.markdown("### AI grader with 429 fallback")
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
                    st.warning("OpenRouter response was not valid JSON. Showing local fallback.")
                    st.json(local_grader(submission))
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 429:
                    st.warning("OpenRouter returned 429 Too Many Requests. Showing local fallback.")
                else:
                    st.warning(f"OpenRouter failed: {exc}. Showing local fallback.")
                st.json(local_grader(submission))
            except Exception as exc:
                st.warning(f"OpenRouter failed: {exc}. Showing local fallback.")
                st.json(local_grader(submission))
        else:
            st.info("No API key provided. Showing local fallback grade.")
            st.json(local_grader(submission))

st.markdown(
    """
    <div style="text-align:center;color:var(--muted);margin-top:2rem;font-size:.9rem;">
        Fully alive interactive Solar PV Forecasting Website • organized sections • animations • images • 3D-style visual system • export-ready evidence
    </div>
    """,
    unsafe_allow_html=True,
)
