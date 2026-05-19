import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st


STUDENT_NAME_DEFAULT = "MAZEN AL-HIMALI"
STUDENT_ID_DEFAULT = "PG12S2540572"
DEFAULT_DATA_PATH = "data/dataset_sample.csv"
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_TARGET_COL = "total_active_power_w"
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

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

st.set_page_config(
    page_title="Mini Project B — Interactive Time-Series Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #08111f 0%, #10233d 35%, #13385c 100%);
            color: #f5f7fa;
        }
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #06101d 0%, #0d2034 100%);
            border-right: 1px solid rgba(200, 155, 60, 0.15);
        }
        h1, h2, h3, h4 {
            color: #ffffff !important;
        }
        .hero-box {
            background: linear-gradient(135deg, rgba(200,155,60,0.16), rgba(10,122,90,0.14));
            border: 1px solid rgba(200,155,60,0.35);
            border-radius: 20px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 28px rgba(0,0,0,0.20);
        }
        .metric-card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        }
        .section-box {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.06);
            border-radius: 12px 12px 0 0;
            color: white;
            padding: 10px 16px;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(200,155,60,0.20) !important;
            border-bottom: 2px solid #C89B3C;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 12px 14px;
            border-radius: 16px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid rgba(200,155,60,0.35);
            background: linear-gradient(135deg, #0A7A5A, #0B5C45);
            color: white;
            font-weight: 600;
        }
        .stSelectbox label, .stTextInput label, .stNumberInput label, .stTextArea label, .stCheckbox label, .stDateInput label {
            color: #f5f7fa !important;
            font-weight: 600;
        }
        .small-note {
            color: #dce7f5;
            opacity: 0.9;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_json_default(obj):
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return str(obj)


def get_openrouter_key():
    key = None
    try:
        key = st.secrets.get("OPENROUTER_API_KEY", None)
    except Exception:
        key = None

    if not key:
        key = os.environ.get("OPENROUTER_API_KEY")

    if not key:
        key = st.text_input(
            "OpenRouter API key",
            type="password",
            help="Used only when you click the AI grader button. It is not stored in app.py.",
        )
    return key


def audit_dataframe(dataframe):
    audit = pd.DataFrame(
        {
            "column": dataframe.columns,
            "dtype": [str(dataframe[c].dtype) for c in dataframe.columns],
            "non_null_count": [int(dataframe[c].notna().sum()) for c in dataframe.columns],
            "missing_pct": [
                round(float(dataframe[c].isna().mean() * 100), 3)
                for c in dataframe.columns
            ],
            "unique_count": [int(dataframe[c].nunique(dropna=True)) for c in dataframe.columns],
        }
    )
    missing_top = audit.sort_values("missing_pct", ascending=False).head(10)
    return audit, missing_top


def infer_frequency(series):
    times = pd.to_datetime(series, errors="coerce").dropna().sort_values()
    if len(times) < 3:
        return {"median_gap": None, "inferred_freq": None, "large_gap_count": 0}

    diffs = times.diff().dropna()
    median_gap = diffs.median()
    inferred = pd.infer_freq(times.drop_duplicates().head(5000))

    large_gap_count = 0
    if pd.notna(median_gap) and median_gap.total_seconds() > 0:
        large_gap_count = int((diffs > 3 * median_gap).sum())

    return {
        "median_gap": str(median_gap),
        "inferred_freq": inferred,
        "large_gap_count": large_gap_count,
    }


def prepare_timeseries(dataframe, timestamp_column, target_column, resample_rule):
    prepared = dataframe.copy()
    prepared[timestamp_column] = pd.to_datetime(prepared[timestamp_column], errors="coerce")
    prepared[target_column] = pd.to_numeric(prepared[target_column], errors="coerce")

    before_rows = len(prepared)
    prepared = prepared.dropna(subset=[timestamp_column, target_column])
    after_drop_rows = len(prepared)
    prepared = prepared.sort_values(timestamp_column)
    duplicate_count = int(prepared[timestamp_column].duplicated().sum())

    numeric_cols = []
    for col in prepared.columns:
        converted = pd.to_numeric(prepared[col], errors="coerce")
        if converted.notna().sum() > 0 and col != timestamp_column:
            prepared[col] = converted
            numeric_cols.append(col)

    prepared = (
        prepared.groupby(timestamp_column, as_index=False)[numeric_cols]
        .mean()
        .sort_values(timestamp_column)
    )

    resampling_note = "No resampling selected."
    if resample_rule and resample_rule != "None":
        prepared = (
            prepared.set_index(timestamp_column)
            .resample(resample_rule)
            .mean(numeric_only=True)
            .interpolate(limit_direction="both")
            .reset_index()
        )
        resampling_note = f"Resampled to {resample_rule} using mean aggregation and interpolation."

    cleaning_report = {
        "rows_before_cleaning": int(before_rows),
        "rows_after_dropping_invalid_timestamp_or_target": int(after_drop_rows),
        "duplicate_timestamps_before_grouping": int(duplicate_count),
        "rows_after_grouping_and_resampling": int(len(prepared)),
        "resampling_note": resampling_note,
    }
    return prepared, cleaning_report


def build_baseline_features(prepared, timestamp_column, target_column, horizon):
    work = prepared[[timestamp_column, target_column]].copy()
    work[timestamp_column] = pd.to_datetime(work[timestamp_column], errors="coerce")
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    work = work.dropna(subset=[timestamp_column, target_column]).sort_values(timestamp_column)

    work["lag_1"] = work[target_column].shift(1)
    work["lag_24"] = work[target_column].shift(24)
    work["rolling_mean_24"] = work[target_column].shift(1).rolling(24).mean()
    work["hour"] = work[timestamp_column].dt.hour
    work["weekend"] = (work[timestamp_column].dt.dayofweek >= 5).astype(int)
    work["month"] = work[timestamp_column].dt.month
    work["y_target"] = work[target_column].shift(-int(horizon))

    feature_columns = ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"]
    feature_table = work.dropna(subset=feature_columns + ["y_target"]).copy()
    X = feature_table[feature_columns]
    y = feature_table["y_target"]
    return feature_table, X, y, feature_columns


def robust_parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def call_openrouter_grader(api_key, evidence_json):
    prompt = AI_GRADER_PROMPT_TEMPLATE.replace(
        "<insert submission.json contents here>",
        evidence_json,
    )

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit.io",
            "X-Title": "UTAS EDA Mini Project B",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def render_hero(student_name, student_id, app_title):
    st.markdown(
        f"""
        <div class="hero-box">
            <h1 style="margin-bottom:0.3rem;">📈 {app_title}</h1>
            <p style="font-size:1.05rem; margin-bottom:0.2rem;"><b>Interactive Time-Series Forecasting Dashboard</b></p>
            <p class="small-note">UTAS Energy Data Analytics • Student: <b>{student_name}</b> • ID: <b>{student_id}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(kpis):
    cols = st.columns(len(kpis))
    for col, (label, value) in zip(cols, kpis):
        col.metric(label, value)


def load_data(load_mode, data_path, uploaded_file):
    if load_mode == "Upload file" and uploaded_file is not None:
        name = str(uploaded_file.name).lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded_file)
        if name.endswith(".json"):
            return pd.read_json(uploaded_file)
        raise ValueError("Unsupported uploaded file type. Use CSV, XLSX, XLS, or JSON.")
    return pd.read_csv(data_path)


inject_custom_css()

# Defaults to avoid export errors
results_df = None
predictions_df = pd.DataFrame()
student_added_features = []
student_added_dashboard = False
student_dashboard_insights = []
has_time_based_split = False
student_modeling_notes = "Modeling has not run yet."
outlier_summary = {}
model_comparison_df = pd.DataFrame()
feature_importance_df = pd.DataFrame()
uncertainty_summary = {}
best_model_name = None
best_model_details = {}
monthly_error_summary = pd.DataFrame()

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    student_name = st.text_input("Student name", value=STUDENT_NAME_DEFAULT)
    student_id = st.text_input("Student ID", value=STUDENT_ID_DEFAULT)
    app_title = st.text_input("Project title", value="HKUST Rooftop PV Power Forecasting")
    project_goal = st.text_area(
        "Project goal",
        value=(
            "Forecast rooftop PV inverter active power using historical time-series "
            "patterns and weather-related variables."
        ),
        height=100,
    )
    deployed_url = st.text_input("Deployed Streamlit URL", value="")
    github_url = st.text_input("GitHub repository URL", value="")

    st.markdown("---")
    load_mode = st.radio("Dataset source", ["Local path", "Upload file"], index=0)
    data_path = st.text_input("Dataset path", value=DEFAULT_DATA_PATH)
    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls", "json"])

render_hero(student_name, student_id, app_title)
st.caption("Modern interactive layout with filters, tabs, custom dashboard views, and a premium dark renewable-energy theme.")

try:
    df = load_data(load_mode, data_path, uploaded_file)
except FileNotFoundError:
    st.error(f"Could not find {data_path}. Upload a file or make sure the path exists.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

columns = list(df.columns)
audit_table, missing_top10 = audit_dataframe(df)

# Main selectors placed near top for interactivity
selector_box = st.container()
with selector_box:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("### Quick setup")
    sel1, sel2, sel3, sel4 = st.columns([1.2, 1.2, 1, 1])
    default_timestamp_index = columns.index(DEFAULT_TIMESTAMP_COL) if DEFAULT_TIMESTAMP_COL in columns else 0
    timestamp_col = sel1.selectbox("Timestamp column", columns, index=default_timestamp_index)

    numeric_candidate_columns = []
    for col in columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            numeric_candidate_columns.append(col)
    if not numeric_candidate_columns:
        st.error("No numeric target candidates were found.")
        st.stop()
    default_target_index = (
        numeric_candidate_columns.index(DEFAULT_TARGET_COL)
        if DEFAULT_TARGET_COL in numeric_candidate_columns else 0
    )
    target_col = sel2.selectbox("Target column", numeric_candidate_columns, index=default_target_index)
    resample_rule = sel3.selectbox("Resampling rule", ["None", "15min", "30min", "1h", "1D"], index=3)
    horizon = int(sel4.number_input("Forecast horizon", min_value=1, max_value=168, value=1, step=1))
    st.markdown('</div>', unsafe_allow_html=True)

prepared_df, cleaning_report = prepare_timeseries(df, timestamp_col, target_col, resample_rule)
feature_table, X, y, feature_cols = build_baseline_features(prepared_df, timestamp_col, target_col, horizon)

timestamp_preview = pd.to_datetime(df[timestamp_col], errors="coerce")
valid_timestamp_pct = float(timestamp_preview.notna().mean() * 100)
coverage_min = timestamp_preview.min()
coverage_max = timestamp_preview.max()
freq_info = infer_frequency(timestamp_preview)
target_preview = pd.to_numeric(df[target_col], errors="coerce")

render_kpi_row([
    ("Rows loaded", f"{len(df):,}"),
    ("Prepared rows", f"{len(prepared_df):,}"),
    ("Feature rows", f"{len(feature_table):,}"),
    ("Timestamp valid", f"{valid_timestamp_pct:.1f}%"),
])

# Dashboard filters
st.markdown("### Interactive dashboard filters")
filter1, filter2, filter3, filter4 = st.columns([1.2, 1.2, 1, 1])
prepared_min = pd.to_datetime(prepared_df[timestamp_col].min()) if not prepared_df.empty else pd.Timestamp.today()
prepared_max = pd.to_datetime(prepared_df[timestamp_col].max()) if not prepared_df.empty else pd.Timestamp.today()
start_date = filter1.date_input("Start date", value=prepared_min.date(), min_value=prepared_min.date(), max_value=prepared_max.date())
end_date = filter2.date_input("End date", value=prepared_max.date(), min_value=prepared_min.date(), max_value=prepared_max.date())
chart_tail = int(filter3.slider("Chart rows", min_value=100, max_value=max(100, min(5000, len(prepared_df) if len(prepared_df) else 100)), value=min(1000, max(100, len(prepared_df) if len(prepared_df) else 100)), step=100))
rolling_window = int(filter4.slider("Rolling window", min_value=1, max_value=72, value=24, step=1))

filtered_prepared_df = prepared_df.copy()
if not filtered_prepared_df.empty:
    filtered_prepared_df[timestamp_col] = pd.to_datetime(filtered_prepared_df[timestamp_col], errors="coerce")
    filtered_prepared_df = filtered_prepared_df[
        (filtered_prepared_df[timestamp_col].dt.date >= start_date)
        & (filtered_prepared_df[timestamp_col].dt.date <= end_date)
    ].copy()

# Modeling controls
st.markdown("### Modeling controls")
mc1, mc2, mc3, mc4 = st.columns([1, 1.2, 1, 1])
run_model = mc1.checkbox("Run modeling", value=True)
target_strategy = mc2.selectbox(
    "Target strategy",
    ["Winsorized target", "Winsorized + log1p transform"],
    index=1,
)
selection_metric = mc3.selectbox("Best model metric", ["MAPE_pct", "RMSE", "MAE"], index=0)
max_model_rows = mc4.slider(
    "Rows for modeling",
    min_value=1000,
    max_value=max(1000, int(len(feature_table)) if len(feature_table) else 1000),
    value=min(30000, max(1000, int(len(feature_table)) if len(feature_table) else 1000)),
    step=1000,
)

main_tabs = st.tabs([
    "🏠 Overview",
    "🧹 Audit & Cleaning",
    "🧠 Modeling",
    "📊 Interactive Dashboard",
    "📤 Export & AI Grader",
])

with main_tabs[0]:
    st.markdown("## Project Overview")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.write(f"**Project goal:** {project_goal}")
        st.write(f"**Dataset source:** {'Uploaded file' if load_mode == 'Upload file' and uploaded_file is not None else data_path}")
        st.write(f"**Coverage:** {coverage_min} to {coverage_max}")
        st.write(f"**Median time gap:** {freq_info['median_gap']}")
        st.write(f"**Inferred frequency:** {freq_info['inferred_freq']}")
        st.write(f"**Large gap count:** {freq_info['large_gap_count']}")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.write("**Target quick stats**")
        st.json(
            {
                "target_missing_pct": round(float(target_preview.isna().mean() * 100), 3),
                "target_mean": round(float(target_preview.mean()), 3),
                "target_std": round(float(target_preview.std()), 3),
                "target_unique_count": int(target_preview.nunique(dropna=True)),
            }
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### First 10 rows")
    st.dataframe(df.head(10), width='stretch')

    if not filtered_prepared_df.empty:
        st.markdown("### Target trend in selected date window")
        overview_chart_df = filtered_prepared_df.set_index(timestamp_col)[target_col].tail(chart_tail)
        st.line_chart(overview_chart_df, height=320)

        rolling_df = filtered_prepared_df[[timestamp_col, target_col]].copy()
        rolling_df["rolling_mean"] = rolling_df[target_col].rolling(rolling_window).mean()
        st.markdown("### Rolling mean view")
        st.line_chart(rolling_df.set_index(timestamp_col)[[target_col, "rolling_mean"]].tail(chart_tail), height=320)
    else:
        st.info("No rows remain after applying the current date filter.")

with main_tabs[1]:
    st.markdown("## Audit & Cleaning")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Audit table")
        st.dataframe(audit_table, width='stretch')
    with c2:
        st.markdown("### Top missing-value columns")
        st.dataframe(missing_top10, width='stretch')

    st.markdown("### Cleaning and resampling report")
    st.json(cleaning_report)

    st.markdown("### Prepared time-series preview")
    st.dataframe(prepared_df.head(20), width='stretch')

    st.markdown("### Baseline feature preview")
    st.write(f"Feature table rows: {len(feature_table):,}")
    st.write(f"X shape: {X.shape}, y length: {len(y):,}")
    st.dataframe(feature_table.head(20), width='stretch')

with main_tabs[2]:
    st.markdown("## Modeling & Evaluation")
    st.write(
        "This section compares multiple models, applies a strict time-based split, shows feature importance, and builds empirical prediction intervals."
    )

    if run_model:
        try:
            from sklearn.base import clone
            from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
            from sklearn.inspection import permutation_importance
            from sklearn.linear_model import RidgeCV
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            model_df = feature_table.copy()
            model_df[timestamp_col] = pd.to_datetime(model_df[timestamp_col], errors="coerce")

            exog_candidates = [
                "irradiance_wm2", "temperature_c", "relative_humidity_pct", "sea_level_pressure_hpa",
                "visibility_km", "wind_speed_ms", "wind_direction_deg", "rainfall_mm",
            ]
            available_exog = [c for c in exog_candidates if c in prepared_df.columns and c != target_col]

            if available_exog:
                exog_df = prepared_df[[timestamp_col] + available_exog].copy()
                exog_df[timestamp_col] = pd.to_datetime(exog_df[timestamp_col], errors="coerce")
                for col in available_exog:
                    exog_df[col] = pd.to_numeric(exog_df[col], errors="coerce")
                model_df = model_df.merge(exog_df, on=timestamp_col, how="left")

            model_df["dayofyear"] = model_df[timestamp_col].dt.dayofyear
            model_df["quarter"] = model_df[timestamp_col].dt.quarter
            model_df["dayofweek"] = model_df[timestamp_col].dt.dayofweek
            model_df["is_daylight_hour"] = model_df["hour"].between(7, 18).astype(int)
            model_df["lag1_x_hour"] = model_df["lag_1"] * model_df["hour"]
            model_df["hour_sin"] = np.sin(2 * np.pi * model_df["hour"] / 24)
            model_df["hour_cos"] = np.cos(2 * np.pi * model_df["hour"] / 24)
            model_df["dayofyear_sin"] = np.sin(2 * np.pi * model_df["dayofyear"] / 365.25)
            model_df["dayofyear_cos"] = np.cos(2 * np.pi * model_df["dayofyear"] / 365.25)

            engineered_features = [
                "dayofyear", "quarter", "dayofweek", "is_daylight_hour", "lag1_x_hour",
                "hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos",
            ]
            student_added_features = engineered_features + available_exog
            model_features = feature_cols + student_added_features

            for col in model_features:
                model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

            model_df = model_df.dropna(subset=[timestamp_col, "y_target"] + model_features).sort_values(timestamp_col)
            target_series = pd.to_numeric(prepared_df[target_col], errors="coerce").dropna()
            q1 = float(target_series.quantile(0.25))
            q3 = float(target_series.quantile(0.75))
            iqr = q3 - q1

            if iqr > 0:
                iqr_lower = q1 - 1.5 * iqr
                iqr_upper = q3 + 1.5 * iqr
            else:
                iqr_lower = float(target_series.min())
                iqr_upper = float(target_series.max())

            target_min = float(target_series.min()) if len(target_series) else 0.0
            outlier_lower = max(0.0, iqr_lower) if target_min >= -1 else iqr_lower
            outlier_upper = iqr_upper
            outlier_mask = (target_series < outlier_lower) | (target_series > outlier_upper)

            outlier_summary = {
                "method": "IQR rule with PV-aware lower clipping when the observed target is non-negative",
                "q1": round(float(q1), 3),
                "q3": round(float(q3), 3),
                "iqr": round(float(iqr), 3),
                "lower_bound": round(float(outlier_lower), 3),
                "upper_bound": round(float(outlier_upper), 3),
                "outlier_count": int(outlier_mask.sum()),
                "outlier_pct": round(float(outlier_mask.mean() * 100), 3),
                "treatment": (
                    "Model training targets and model predictions are winsorized to the selected IQR bounds. "
                    "A log1p target option is also available to reduce the effect of extreme power spikes on percentage-based errors."
                ),
            }
            st.markdown("### Outlier summary")
            st.json(outlier_summary)

            if len(model_df) < 100:
                st.warning("Not enough rows for a reliable time-based train/validation split.")
            else:
                model_df_used = model_df.tail(int(max_model_rows)).copy()
                split_idx = int(len(model_df_used) * 0.80)
                train_df = model_df_used.iloc[:split_idx].copy()
                valid_df = model_df_used.iloc[split_idx:].copy()

                calibration_idx = max(int(len(train_df) * 0.85), 1)
                train_core_df = train_df.iloc[:calibration_idx].copy()
                calibration_df = train_df.iloc[calibration_idx:].copy()
                if len(calibration_df) < 50:
                    train_core_df = train_df.copy()
                    calibration_df = train_df.tail(min(200, len(train_df))).copy()

                X_train_core = train_core_df[model_features]
                y_train_core_raw = train_core_df["y_target"].clip(outlier_lower, outlier_upper)
                X_train_all = train_df[model_features]
                y_train_all_raw = train_df["y_target"].clip(outlier_lower, outlier_upper)
                X_cal = calibration_df[model_features]
                y_cal = calibration_df["y_target"]
                X_valid = valid_df[model_features]
                y_valid = valid_df["y_target"]

                use_log_target = target_strategy == "Winsorized + log1p transform"

                def transform_target(values):
                    values = pd.Series(values).astype(float).clip(outlier_lower, outlier_upper)
                    if use_log_target:
                        return np.log1p(np.maximum(values.to_numpy(), 0))
                    return values.to_numpy()

                def inverse_target(values):
                    values = np.asarray(values, dtype=float)
                    if use_log_target:
                        values = np.expm1(values)
                    return np.clip(values, outlier_lower, outlier_upper)

                def metric_row(model_name, y_true, y_pred, train_rows, validation_rows, notes=""):
                    y_true = pd.Series(y_true).astype(float)
                    y_pred = np.asarray(y_pred, dtype=float)
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    mape = float(np.mean(np.abs((y_true.to_numpy() - y_pred) / np.maximum(np.abs(y_true.to_numpy()), 1))) * 100)
                    r2 = r2_score(y_true, y_pred)
                    return {
                        "model": model_name,
                        "split_type": "time_based_80_20",
                        "train_start": str(train_df[timestamp_col].min()),
                        "train_end": str(train_df[timestamp_col].max()),
                        "validation_start": str(valid_df[timestamp_col].min()),
                        "validation_end": str(valid_df[timestamp_col].max()),
                        "horizon_rows": int(horizon),
                        "target_strategy": target_strategy,
                        "MAE": round(float(mae), 3),
                        "RMSE": round(float(rmse), 3),
                        "MAPE_pct": round(float(mape), 3),
                        "R2": round(float(r2), 4),
                        "train_rows": int(train_rows),
                        "validation_rows": int(validation_rows),
                        "notes": notes,
                    }

                candidate_specs = [
                    {"name": "Naive seasonal lag_24 baseline", "estimator": None, "params": {"prediction": "lag_24_fallback_lag_1"}},
                    {"name": "RidgeCV scaled linear model", "estimator": make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])), "params": {"alphas": [0.1, 1.0, 10.0, 100.0]}},
                    {"name": "RandomForestRegressor compact", "estimator": RandomForestRegressor(n_estimators=60, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1), "params": {"n_estimators": 60, "max_depth": 14, "min_samples_leaf": 3}},
                ]
                hgb_param_grid = [
                    {"max_iter": 150, "learning_rate": 0.08, "max_leaf_nodes": 15, "l2_regularization": 0.0},
                    {"max_iter": 220, "learning_rate": 0.06, "max_leaf_nodes": 31, "l2_regularization": 0.0},
                    {"max_iter": 260, "learning_rate": 0.04, "max_leaf_nodes": 31, "l2_regularization": 0.1},
                ]
                for idx, params in enumerate(hgb_param_grid, start=1):
                    candidate_specs.append({"name": f"HistGradientBoostingRegressor tuned #{idx}", "estimator": HistGradientBoostingRegressor(**params, random_state=42), "params": params})

                try:
                    from xgboost import XGBRegressor
                    candidate_specs.append(
                        {"name": "XGBoostRegressor optional", "estimator": XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=4, subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror", random_state=42, n_jobs=2), "params": {"n_estimators": 250, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.85, "colsample_bytree": 0.85}}
                    )
                except Exception:
                    st.caption("Optional XGBoost is not installed. Comparison uses scikit-learn models only.")

                comparison_rows = []
                fitted_candidates = []
                for spec in candidate_specs:
                    name = spec["name"]
                    if spec["estimator"] is None:
                        y_pred_valid = valid_df["lag_24"].fillna(valid_df["lag_1"]).fillna(train_df["y_target"].median()).to_numpy()
                        y_pred_valid = np.clip(y_pred_valid, outlier_lower, outlier_upper)
                        comparison_rows.append(metric_row(name, y_valid, y_pred_valid, len(train_df), len(valid_df), notes="Transparent baseline for comparison.") | {"params": json.dumps(spec["params"])})
                        fitted_candidates.append({"name": name, "estimator": None, "params": spec["params"], "valid_pred": y_pred_valid, "calibration_residuals": None})
                        continue
                    estimator = clone(spec["estimator"])
                    estimator.fit(X_train_all, transform_target(y_train_all_raw))
                    y_pred_valid = inverse_target(estimator.predict(X_valid))
                    calibration_model = clone(spec["estimator"])
                    calibration_model.fit(X_train_core, transform_target(y_train_core_raw))
                    y_pred_cal = inverse_target(calibration_model.predict(X_cal))
                    calibration_residuals = y_cal.to_numpy(dtype=float) - y_pred_cal
                    comparison_rows.append(metric_row(name, y_valid, y_pred_valid, len(train_df), len(valid_df), notes="Candidate model in explicit comparison table.") | {"params": json.dumps(spec["params"])})
                    fitted_candidates.append({"name": name, "estimator": estimator, "params": spec["params"], "valid_pred": y_pred_valid, "calibration_residuals": calibration_residuals})

                model_comparison_df = pd.DataFrame(comparison_rows).sort_values([selection_metric, "RMSE"], ascending=[True, True]).reset_index(drop=True)
                best_row = model_comparison_df.iloc[0].to_dict()
                best_model_name = str(best_row["model"])
                best_candidate = next(item for item in fitted_candidates if item["name"] == best_model_name)
                best_model_details = {
                    "selected_by": selection_metric,
                    "best_model": best_model_name,
                    "best_params": best_candidate["params"],
                    "target_strategy": target_strategy,
                    "candidate_count": int(len(model_comparison_df)),
                }

                calibration_residuals = best_candidate.get("calibration_residuals")
                if calibration_residuals is None or len(calibration_residuals) < 20:
                    calibration_residuals = y_valid.to_numpy(dtype=float) - best_candidate["valid_pred"]
                    interval_source = "validation residual fallback"
                else:
                    interval_source = "time-ordered calibration slice inside training period"

                lower_resid = float(np.nanquantile(calibration_residuals, 0.05))
                upper_resid = float(np.nanquantile(calibration_residuals, 0.95))
                y_pred = best_candidate["valid_pred"]
                pred_lower = np.clip(y_pred + lower_resid, outlier_lower, outlier_upper)
                pred_upper = np.clip(y_pred + upper_resid, outlier_lower, outlier_upper)

                predictions_df = valid_df[[timestamp_col, target_col, "y_target"]].copy()
                predictions_df["prediction"] = y_pred
                predictions_df["prediction_lower_90"] = np.minimum(pred_lower, pred_upper)
                predictions_df["prediction_upper_90"] = np.maximum(pred_lower, pred_upper)
                predictions_df["residual"] = predictions_df["y_target"] - predictions_df["prediction"]
                predictions_df["absolute_error"] = predictions_df["residual"].abs()
                predictions_df["interval_covered"] = (
                    (predictions_df["y_target"] >= predictions_df["prediction_lower_90"])
                    & (predictions_df["y_target"] <= predictions_df["prediction_upper_90"])
                )

                interval_coverage = float(predictions_df["interval_covered"].mean() * 100)
                avg_interval_width = float((predictions_df["prediction_upper_90"] - predictions_df["prediction_lower_90"]).mean())
                uncertainty_summary = {
                    "method": "Empirical 90% prediction interval from residual quantiles",
                    "interval_source": interval_source,
                    "lower_residual_quantile_5pct": round(lower_resid, 3),
                    "upper_residual_quantile_95pct": round(upper_resid, 3),
                    "validation_interval_coverage_pct": round(interval_coverage, 3),
                    "average_interval_width": round(avg_interval_width, 3),
                }

                if best_candidate["estimator"] is not None:
                    importance_sample_size = min(1200, len(X_valid))
                    X_importance = X_valid.tail(importance_sample_size)
                    y_importance = y_valid.tail(importance_sample_size)
                    try:
                        perm = permutation_importance(
                            best_candidate["estimator"],
                            X_importance,
                            transform_target(y_importance),
                            n_repeats=5,
                            random_state=42,
                            scoring="neg_mean_absolute_error",
                        )
                        feature_importance_df = pd.DataFrame(
                            {
                                "feature": model_features,
                                "importance_mean": perm.importances_mean,
                                "importance_std": perm.importances_std,
                            }
                        ).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
                    except Exception as importance_exc:
                        feature_importance_df = pd.DataFrame([{"feature": "Feature importance unavailable", "importance_mean": 0.0, "importance_std": 0.0, "reason": str(importance_exc)}])
                else:
                    feature_importance_df = pd.DataFrame([{"feature": "lag_24", "importance_mean": 1.0, "importance_std": 0.0, "reason": "Naive seasonal baseline selected."}])

                results_df = model_comparison_df.copy()
                has_time_based_split = True
                student_modeling_notes = (
                    "A time-based split was used throughout: earliest 80% for training and latest 20% for validation. "
                    "A later calibration slice inside training is used for empirical intervals. Multiple candidate models are compared, "
                    f"and the selected model is {best_model_name}, chosen by {selection_metric}. Target treatment: {target_strategy}."
                )

                st.markdown("### Model comparison")
                st.dataframe(model_comparison_df, width='stretch')
                st.markdown("### Selected model details")
                st.json(best_model_details)
                st.markdown("### Feature importance")
                st.dataframe(feature_importance_df, width='stretch')
                if "importance_mean" in feature_importance_df.columns and not feature_importance_df.empty:
                    st.bar_chart(feature_importance_df.set_index("feature")["importance_mean"], height=300)
                st.markdown("### Uncertainty summary")
                st.json(uncertainty_summary)
                st.markdown("### Prediction sample")
                st.dataframe(predictions_df.head(20), width='stretch')
                plot_df = predictions_df.set_index(timestamp_col)[["y_target", "prediction", "prediction_lower_90", "prediction_upper_90"]].tail(500)
                st.markdown("### Actual vs predicted with 90% interval")
                st.line_chart(plot_df, height=320)

        except Exception as exc:
            st.error(f"Student modeling section failed: {exc}")
            results_df = None
            model_comparison_df = pd.DataFrame()
            feature_importance_df = pd.DataFrame()
            predictions_df = pd.DataFrame()
            uncertainty_summary = {}
            best_model_name = None
            best_model_details = {}
            has_time_based_split = False
            student_modeling_notes = f"Modeling failed with error: {exc}"
    else:
        st.info("Modeling is turned off.")

with main_tabs[3]:
    st.markdown("## Interactive Dashboard")
    dash_df = filtered_prepared_df[[timestamp_col, target_col]].dropna().copy() if not filtered_prepared_df.empty else pd.DataFrame()
    dash_df = dash_df.sort_values(timestamp_col) if not dash_df.empty else dash_df

    if dash_df.empty:
        st.warning("Dashboard cannot be created because the filtered time-series data is empty.")
        student_added_dashboard = False
        student_dashboard_insights = []
    else:
        latest_value = float(dash_df[target_col].iloc[-1])
        mean_value = float(dash_df[target_col].mean())
        max_value = float(dash_df[target_col].max())
        zero_pct = float((dash_df[target_col] <= 0).mean() * 100)
        render_kpi_row([
            ("Latest power", f"{latest_value:,.1f} W"),
            ("Average power", f"{mean_value:,.1f} W"),
            ("Maximum power", f"{max_value:,.1f} W"),
            ("Zero / negative power", f"{zero_pct:.1f}%"),
        ])

        view1, view2 = st.columns([1.3, 1])
        with view1:
            st.markdown("### Recent target trend")
            st.line_chart(dash_df.set_index(timestamp_col)[target_col].tail(chart_tail), height=300)
            dash_roll = dash_df[[timestamp_col, target_col]].copy()
            dash_roll["rolling_mean"] = dash_roll[target_col].rolling(rolling_window).mean()
            st.markdown("### Trend + rolling mean")
            st.line_chart(dash_roll.set_index(timestamp_col)[[target_col, "rolling_mean"]].tail(chart_tail), height=300)
        with view2:
            st.markdown("### Average power by hour of day")
            hourly_profile = dash_df.assign(hour=dash_df[timestamp_col].dt.hour).groupby("hour")[target_col].mean().reset_index()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(hourly_profile["hour"], hourly_profile[target_col], marker="o")
            ax.set_xlabel("Hour of day")
            ax.set_ylabel(target_col)
            ax.set_title("Average PV power by hour")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        extra_tabs = st.tabs(["Daily Summary", "Distribution", "Diagnostics"])
        with extra_tabs[0]:
            daily_summary = (
                dash_df.set_index(timestamp_col)[target_col]
                .resample("D")
                .agg(["mean", "max", "count"])
                .dropna()
                .tail(30)
                .reset_index()
            )
            st.dataframe(daily_summary, width='stretch')
        with extra_tabs[1]:
            hist_fig, hist_ax = plt.subplots(figsize=(8, 4))
            hist_ax.hist(dash_df[target_col].dropna(), bins=30)
            hist_ax.set_title("Target distribution")
            hist_ax.set_xlabel(target_col)
            hist_ax.set_ylabel("Count")
            st.pyplot(hist_fig)
        with extra_tabs[2]:
            time_diffs = dash_df[timestamp_col].diff().dropna()
            median_gap = time_diffs.median() if len(time_diffs) else pd.NaT
            large_gap_count = int((time_diffs > 3 * median_gap).sum()) if pd.notna(median_gap) and median_gap.total_seconds() > 0 else 0
            student_dashboard_insights = [
                f"The target power ranges from {dash_df[target_col].min():,.1f} W to {dash_df[target_col].max():,.1f} W.",
                f"The average target power is {mean_value:,.1f} W after cleaning and optional resampling.",
                "The hourly profile shows the expected solar production pattern, with generation concentrated during daylight hours.",
                f"The dashboard detected {large_gap_count} unusually large time gaps after preparation.",
                "A limitation is that descriptive plots alone do not prove forecasting quality, so model metrics are also included.",
            ]
            for insight in student_dashboard_insights:
                st.write(f"- {insight}")
            student_added_dashboard = True

        if isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty:
            st.markdown("### Forecast diagnostics")
            diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)
            diag_col1.metric("Validation rows", f"{len(predictions_df):,}")
            diag_col2.metric("Mean absolute error", f"{predictions_df['absolute_error'].mean():,.2f}")
            diag_col3.metric("Max absolute error", f"{predictions_df['absolute_error'].max():,.2f}")
            diag_col4.metric("90% interval coverage", f"{predictions_df['interval_covered'].mean() * 100:,.1f}%")

            interval_plot_df = predictions_df.set_index(timestamp_col)[["y_target", "prediction", "prediction_lower_90", "prediction_upper_90"]].tail(500)
            st.line_chart(interval_plot_df, height=320)

            residual_plot_df = predictions_df.set_index(timestamp_col)["residual"].tail(500)
            st.markdown("### Residual plot")
            st.line_chart(residual_plot_df, height=260)

            monthly_base_df = predictions_df.copy()
            monthly_base_df[timestamp_col] = pd.to_datetime(monthly_base_df[timestamp_col], errors="coerce")
            monthly_base_df = monthly_base_df.dropna(subset=[timestamp_col]).set_index(timestamp_col)
            monthly_group_df = monthly_base_df.copy()
            monthly_group_df["month"] = monthly_group_df.index.to_period("M").astype(str)
            monthly_error_summary = (
                monthly_group_df.groupby("month", as_index=False)
                .agg(
                    actual_mean=("y_target", "mean"),
                    prediction_mean=("prediction", "mean"),
                    mae=("absolute_error", "mean"),
                    max_error=("absolute_error", "max"),
                    n=("absolute_error", "count"),
                )
                .dropna()
            )
            st.markdown("### Monthly validation error summary")
            st.dataframe(monthly_error_summary, width='stretch')

            student_dashboard_insights.extend([
                "A forecasting model was evaluated using a time-based validation period.",
                "The dashboard compares actual and predicted power values for the validation period.",
                "Residual diagnostics show when the model under-predicts or over-predicts PV power.",
                "Monthly error summaries help identify whether model performance changes over time.",
                "Prediction intervals communicate forecast confidence instead of showing only a single point forecast.",
            ])
            student_added_dashboard = True
        else:
            st.info("Forecast diagnostics will appear here after modeling runs.")

with main_tabs[4]:
    st.markdown("## Export & AI Grader")
    has_metrics_table = isinstance(results_df, pd.DataFrame) and not results_df.empty
    results_table = results_df.to_dict(orient="records") if has_metrics_table else []

    submission = {
        "student": {
            "name": student_name,
            "id": student_id,
            "app_title": app_title,
            "project_goal": project_goal,
            "deployed_url": deployed_url,
            "github_url": github_url,
        },
        "data_integrity": {
            "dataset_path": data_path,
            "rows_loaded": int(len(df)),
            "columns_loaded": int(len(df.columns)),
            "timestamp_column": timestamp_col,
            "target_column": target_col,
            "timestamp_coverage_start": str(coverage_min),
            "timestamp_coverage_end": str(coverage_max),
            "timestamp_valid_pct": round(float(valid_timestamp_pct), 3),
            "median_time_gap": freq_info["median_gap"],
            "inferred_frequency": freq_info["inferred_freq"],
            "large_gap_count": freq_info["large_gap_count"],
            "cleaning_report": cleaning_report,
            "missing_table_top10": missing_top10.to_dict(orient="records"),
            "outliers_discussed": True,
            "outlier_summary": outlier_summary,
            "resampling_discussed": True,
            "resampling_note": f"Selected resampling rule: {resample_rule}. The app allows regular interval resampling before feature creation.",
        },
        "feature_engineering": {
            "baseline_features": feature_cols,
            "student_added_features": student_added_features,
            "horizon_rows": int(horizon),
            "feature_table_rows": int(len(feature_table)),
        },
        "modeling_and_evaluation": {
            "has_time_based_split": bool(has_time_based_split),
            "has_metrics_table": bool(has_metrics_table),
            "results_table": results_table,
            "model_comparison_table": model_comparison_df.to_dict(orient="records") if isinstance(model_comparison_df, pd.DataFrame) and not model_comparison_df.empty else [],
            "best_model_name": best_model_name,
            "best_model_details": best_model_details,
            "feature_importance_table": feature_importance_df.to_dict(orient="records") if isinstance(feature_importance_df, pd.DataFrame) and not feature_importance_df.empty else [],
            "uncertainty_summary": uncertainty_summary,
            "predictions_created": isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty,
            "prediction_interval_columns_present": isinstance(predictions_df, pd.DataFrame) and {"prediction_lower_90", "prediction_upper_90"}.issubset(predictions_df.columns),
            "student_notes": student_modeling_notes,
        },
        "dashboard": {
            "has_baseline_plot": True,
            "has_student_added_dashboard": bool(student_added_dashboard),
            "insights": student_dashboard_insights,
        },
        "presentation_and_rigor": {
            "limitations": [
                "PV output depends on weather, cloud cover, shading, and system conditions.",
                "The selected model is chosen from a compact comparison table, not from an exhaustive production AutoML search.",
                "Empirical prediction intervals depend on residual behavior in the calibration period and may under-cover during unusual weather or equipment events.",
                "Validation metrics should be interpreted for the chosen horizon, resampling level, target treatment, and selected row window.",
            ],
            "reproducibility_notes": [
                "The app loads a local dataset path or an uploaded dataset file.",
                "The model uses a time-based train/validation split and an internal calibration slice for uncertainty estimates.",
                "The exported submission.json captures model comparison, hyper-parameter tuning, feature importance, and uncertainty evidence used by the AI grader.",
            ],
        },
    }

    submission_json = json.dumps(submission, indent=2, default=safe_json_default)
    project_card = f"""# {app_title}

## Student
- Name: {student_name}
- ID: {student_id}

## Goal
{project_goal}

## Dataset
- Source: {'Uploaded file' if load_mode == 'Upload file' and uploaded_file is not None else data_path}
- Timestamp column: {timestamp_col}
- Target column: {target_col}
- Rows loaded: {len(df):,}
- Rows prepared: {len(prepared_df):,}
- Coverage: {coverage_min} to {coverage_max}

## Preparation
{json.dumps(cleaning_report, indent=2, default=safe_json_default)}

## Features
Baseline features:
{", ".join(feature_cols)}

Student-added features:
{", ".join(student_added_features) if student_added_features else 'None yet'}

## Modeling and evaluation
{student_modeling_notes}

Metrics table available: {has_metrics_table}

Selected model:
{best_model_name if best_model_name else 'None yet'}

Uncertainty summary:
{json.dumps(uncertainty_summary, indent=2, default=safe_json_default) if uncertainty_summary else 'None yet'}

## Dashboard insights
{chr(10).join('- ' + item for item in student_dashboard_insights)}

## Limitations
- PV generation is weather-sensitive and can change sharply under cloud cover.
- Results depend on the selected resampling interval and forecast horizon.
- More advanced models and deeper error analysis may improve performance.
"""

    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download submission.json", data=submission_json, file_name="submission.json", mime="application/json")
    with d2:
        st.download_button("Download project_card.md", data=project_card, file_name="project_card.md", mime="text/markdown")

    exp1, exp2 = st.tabs(["Preview submission.json", "Preview project_card.md"])
    with exp1:
        st.json(submission)
    with exp2:
        st.markdown(project_card)

    st.markdown("### AI grader (/80)")
    st.write(f"The AI grader uses OpenRouter with the fixed model string `{OPENROUTER_MODEL}` and the fixed Project B /80 rubric.")
    with st.expander("Show fixed AI grader prompt"):
        st.code(AI_GRADER_PROMPT_TEMPLATE, language="text")

    api_key = get_openrouter_key()
    if st.button("Run AI grader"):
        if not api_key:
            st.error("Please provide OPENROUTER_API_KEY through secrets, environment, or the password field.")
        else:
            with st.spinner("Calling AI grader..."):
                try:
                    raw_output = call_openrouter_grader(api_key, submission_json)
                    parsed = robust_parse_json(raw_output)
                    if parsed is not None:
                        st.success("AI grader returned valid JSON.")
                        st.json(parsed)
                    else:
                        st.warning("Could not parse grader output as JSON. Raw output below.")
                        st.text(raw_output)
                except Exception as exc:
                    st.error(f"AI grader request failed: {exc}")
