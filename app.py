import json
import os
from pathlib import Path
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
    page_title="Mini Project B — Time-Series Forecasting Starter",
    page_icon="📈",
    layout="wide",
)


def safe_json_dumps(obj):
    """Convert Python objects to readable JSON."""
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


@st.cache_data(show_spinner=False)
def load_dataset(csv_path):
    """Load the local CSV dataset used by the deployed app."""
    return pd.read_csv(csv_path)


def dataframe_audit(dataframe):
    """Return column-level audit information."""
    return pd.DataFrame(
        {
            "column": list(dataframe.columns),
            "dtype": [str(dataframe[col].dtype) for col in dataframe.columns],
            "non_null_count": [int(dataframe[col].notna().sum()) for col in dataframe.columns],
            "missing_pct": [round(float(dataframe[col].isna().mean() * 100), 3) for col in dataframe.columns],
            "unique_count": [int(dataframe[col].nunique(dropna=True)) for col in dataframe.columns],
        }
    )


def likely_datetime_columns(dataframe):
    """Suggest likely timestamp columns without treating ordinary numeric columns as dates."""
    rows = []
    tokens = ["time", "date", "timestamp", "datetime"]
    sample = dataframe.head(min(len(dataframe), 5000))

    for col in dataframe.columns:
        col_lower = str(col).lower()
        name_score = int(any(token in col_lower for token in tokens))
        should_try_parse = name_score or pd.api.types.is_object_dtype(sample[col]) or pd.api.types.is_string_dtype(sample[col])

        if should_try_parse:
            parsed = pd.to_datetime(sample[col], errors="coerce")
            valid_ratio = float(parsed.notna().mean()) if len(sample) else 0.0
        else:
            valid_ratio = 0.0

        score = name_score + valid_ratio
        if name_score or valid_ratio >= 0.70:
            rows.append(
                {
                    "column": col,
                    "valid_datetime_pct": round(valid_ratio * 100, 2),
                    "name_hint": bool(name_score),
                    "score": round(score, 3),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["column", "valid_datetime_pct", "name_hint", "score"])

    return pd.DataFrame(rows).sort_values(["score", "valid_datetime_pct"], ascending=False)


def numeric_target_candidates(dataframe):
    """Suggest numeric target columns that are not mostly missing, constant, or ID-like."""
    rows = []
    n_rows = max(len(dataframe), 1)

    for col in dataframe.columns:
        converted = pd.to_numeric(dataframe[col], errors="coerce")
        valid_ratio = float(converted.notna().mean())
        missing_pct = 100 - valid_ratio * 100
        unique_count = int(converted.nunique(dropna=True))
        std_value = converted.std(skipna=True)

        is_id_like_name = "id" in str(col).lower()
        is_id_like_unique = unique_count >= max(50, int(0.95 * n_rows))
        is_constant = unique_count <= 2 or pd.isna(std_value) or float(std_value) == 0.0

        if valid_ratio >= 0.50 and not is_constant and not is_id_like_name and not is_id_like_unique:
            rows.append(
                {
                    "column": col,
                    "missing_pct": round(missing_pct, 3),
                    "mean": round(float(converted.mean(skipna=True)), 4),
                    "std": round(float(std_value), 4),
                    "unique_count": unique_count,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["column", "missing_pct", "mean", "std", "unique_count"])

    return pd.DataFrame(rows).sort_values(["missing_pct", "std"], ascending=[True, False])


def clean_time_series(dataframe, timestamp_col, target_col):
    """Parse timestamp, convert target to numeric, drop invalid rows, sort by time."""
    if timestamp_col == target_col:
        raise ValueError("Timestamp column and target column must be different.")

    cleaned = dataframe.copy()
    rows_before = len(cleaned)

    parsed_timestamp = pd.to_datetime(cleaned[timestamp_col], errors="coerce")
    numeric_target = pd.to_numeric(cleaned[target_col], errors="coerce")

    invalid_timestamps = int(parsed_timestamp.isna().sum())
    missing_targets = int(numeric_target.isna().sum())

    cleaned[timestamp_col] = parsed_timestamp
    cleaned[target_col] = numeric_target
    cleaned = cleaned.dropna(subset=[timestamp_col, target_col]).sort_values(timestamp_col)
    duplicate_count_before_drop = int(cleaned[timestamp_col].duplicated().sum())
    cleaned = cleaned.drop_duplicates(subset=[timestamp_col], keep="last")

    report = {
        "rows_before_cleaning": int(rows_before),
        "rows_after_cleaning": int(len(cleaned)),
        "invalid_timestamp_rows_removed": invalid_timestamps,
        "missing_target_rows_removed": missing_targets,
        "duplicate_timestamps_removed": duplicate_count_before_drop,
    }
    return cleaned.reset_index(drop=True), report


def resample_time_series(dataframe, timestamp_col, target_col, rule):
    """Optionally resample numeric columns by mean."""
    if dataframe.empty or rule == "No resampling":
        return dataframe.copy()

    numeric_df = dataframe.copy()
    for col in numeric_df.columns:
        if col != timestamp_col:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

    numeric_cols = numeric_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        numeric_cols.append(target_col)

    resampled = (
        numeric_df.set_index(timestamp_col)[numeric_cols]
        .resample(rule)
        .mean()
        .reset_index()
    )
    return resampled.dropna(subset=[target_col]).reset_index(drop=True)


def infer_time_coverage(dataframe, timestamp_col):
    """Return basic coverage and frequency diagnostics."""
    if dataframe.empty or timestamp_col not in dataframe.columns:
        return {
            "start": None,
            "end": None,
            "rows": int(len(dataframe)),
            "median_time_gap": None,
            "large_gap_count": 0,
        }

    ts = pd.to_datetime(dataframe[timestamp_col], errors="coerce").dropna().sort_values()
    if ts.empty:
        return {
            "start": None,
            "end": None,
            "rows": int(len(dataframe)),
            "median_time_gap": None,
            "large_gap_count": 0,
        }

    diffs = ts.diff().dropna()
    median_gap = diffs.median() if len(diffs) else pd.NaT
    large_gap_count = 0
    if pd.notna(median_gap) and median_gap.total_seconds() > 0:
        large_gap_count = int((diffs > 3 * median_gap).sum())

    return {
        "start": str(ts.min()),
        "end": str(ts.max()),
        "rows": int(len(dataframe)),
        "median_time_gap": str(median_gap) if pd.notna(median_gap) else None,
        "large_gap_count": large_gap_count,
    }


def make_baseline_features(dataframe, timestamp_col, target_col, horizon):
    """Create baseline forecasting features only. No model training is included."""
    feature_cols = ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"]

    if dataframe.empty:
        empty = pd.DataFrame(columns=[timestamp_col, target_col] + feature_cols + ["y_target"])
        return empty, pd.DataFrame(columns=feature_cols), pd.Series(dtype=float, name="y_target"), feature_cols

    work = dataframe[[timestamp_col, target_col]].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[timestamp_col, target_col]).sort_values(timestamp_col).reset_index(drop=True)

    work["lag_1"] = work[target_col].shift(1)
    work["lag_24"] = work[target_col].shift(24)
    work["rolling_mean_24"] = work[target_col].shift(1).rolling(window=24, min_periods=3).mean()
    work["hour"] = work[timestamp_col].dt.hour
    work["weekend"] = work[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)
    work["month"] = work[timestamp_col].dt.month
    work["y_target"] = work[target_col].shift(-int(horizon))

    feature_table = work[[timestamp_col, target_col] + feature_cols + ["y_target"]].dropna().reset_index(drop=True)
    X = feature_table[feature_cols].copy()
    y = feature_table["y_target"].copy()
    return feature_table, X, y, feature_cols


def get_openrouter_api_key():
    """Read API key from Streamlit Secrets, environment variable, or password field."""
    key = ""

    try:
        key = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.getenv("OPENROUTER_API_KEY", "")

    if not key:
        key = st.text_input(
            "OpenRouter API key",
            type="password",
            help="Used only for the AI grader. It is not stored in this app.",
        )

    return str(key).strip() if key else ""


def extract_first_json_object(text):
    """Extract the first balanced JSON object from model output."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        char = text[idx]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]

    return None


def parse_grader_response(raw_text):
    """Parse grader output as JSON, with fallback extraction."""
    try:
        return json.loads(raw_text), None
    except Exception as first_error:
        json_candidate = extract_first_json_object(raw_text)
        if json_candidate:
            try:
                return json.loads(json_candidate), None
            except Exception as second_error:
                return None, f"{first_error}; fallback parse failed: {second_error}"
        return None, str(first_error)


def call_openrouter_grader(api_key, prompt):
    """Call OpenRouter using the fixed model and prompt."""
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "UTAS EDA Mini Project B Grader",
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def to_records_or_empty(value):
    """Safely convert a DataFrame to records for submission.json."""
    if isinstance(value, pd.DataFrame):
        return value.replace({np.nan: None}).to_dict(orient="records")
    return []


st.title("Mini Project B — Time-Series Forecasting Starter")
st.caption("This starter prepares the data and baseline feature table. Students add the forecasting model, metrics, and extra evidence.")

st.header("1) Student and project information")

info_col1, info_col2 = st.columns(2)
with info_col1:
    student_name = st.text_input("Student name", value=STUDENT_NAME_DEFAULT)
    student_id = st.text_input("Student ID", value=STUDENT_ID_DEFAULT)
    project_title = st.text_input(
        "Project title",
        value="HKUST Rooftop PV Power Forecasting",
    )
with info_col2:
    deployed_url = st.text_input("Deployed Streamlit URL", value="")
    repo_url = st.text_input("GitHub repo URL", value="")
    data_path = st.text_input("Dataset path", value=DEFAULT_DATA_PATH)

project_goal = st.text_area(
    "Project goal",
    value=(
        "Forecast rooftop PV active power using timestamp-based features and weather/inverter data. "
        "This starter prepares clean time-series data and baseline features; the student must add the model, metrics, and final insights."
    ),
    height=90,
)

st.header("2) Load dataset and audit")

try:
    df = load_dataset(data_path)
except FileNotFoundError:
    st.error(f"Dataset not found at `{data_path}`. Keep `data/dataset_sample.csv` in the repository.")
    st.stop()
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

st.subheader("First 10 rows")
st.dataframe(df.head(10), use_container_width=True)

audit_df = dataframe_audit(df)
st.subheader("Columns, dtypes, missing %, and unique counts")
st.dataframe(audit_df, use_container_width=True)

st.subheader("Top 10 missing-value columns")
st.dataframe(audit_df.sort_values("missing_pct", ascending=False).head(10), use_container_width=True)

audit_col1, audit_col2 = st.columns(2)
with audit_col1:
    st.subheader("Likely timestamp columns")
    st.dataframe(likely_datetime_columns(df).head(3), use_container_width=True)
with audit_col2:
    st.subheader("Likely numeric target columns")
    st.dataframe(numeric_target_candidates(df).head(3), use_container_width=True)

st.header("3) Select timestamp and target")

if DEFAULT_TIMESTAMP_COL in df.columns:
    default_ts_index = list(df.columns).index(DEFAULT_TIMESTAMP_COL)
else:
    default_ts_index = 0

timestamp_col = st.selectbox("Timestamp column", options=list(df.columns), index=int(default_ts_index))

numeric_like_cols = [
    col
    for col in df.columns
    if pd.to_numeric(df[col], errors="coerce").notna().mean() >= 0.50 and col != timestamp_col
]

if DEFAULT_TARGET_COL not in numeric_like_cols and DEFAULT_TARGET_COL in df.columns and DEFAULT_TARGET_COL != timestamp_col:
    numeric_like_cols.insert(0, DEFAULT_TARGET_COL)

if not numeric_like_cols:
    st.error("No viable numeric target column was found. Choose a different dataset or check the uploaded file.")
    st.stop()

default_target_index = numeric_like_cols.index(DEFAULT_TARGET_COL) if DEFAULT_TARGET_COL in numeric_like_cols else 0
target_col = st.selectbox("Target column", options=numeric_like_cols, index=int(default_target_index))

try:
    cleaned_df, cleaning_report = clean_time_series(df, timestamp_col, target_col)
except Exception as exc:
    st.error(f"Could not prepare time-series data: {exc}")
    st.stop()

coverage = infer_time_coverage(cleaned_df, timestamp_col)

st.subheader("Cleaned time-series summary")
st.json({**cleaning_report, **coverage})

if cleaned_df.empty:
    st.error("No valid rows remain after parsing the timestamp and target columns.")
    st.stop()

st.header("4) Optional resampling and forecast horizon")

resample_rule = st.selectbox(
    "Resampling rule",
    options=["No resampling", "5min", "15min", "30min", "1h", "1D"],
    index=0,
)
horizon = st.number_input(
    "Forecast horizon in rows after optional resampling",
    min_value=1,
    max_value=1000,
    value=1,
    step=1,
)

prepared_df = resample_time_series(cleaned_df, timestamp_col, target_col, resample_rule)
prepared_coverage = infer_time_coverage(prepared_df, timestamp_col)

st.subheader("Prepared time-series coverage")
st.json(prepared_coverage)

if prepared_df.empty:
    st.error("No rows remain after optional resampling. Try a different resampling rule.")
    st.stop()

st.header("5) Baseline feature table")

feature_table, X, y, feature_cols = make_baseline_features(prepared_df, timestamp_col, target_col, horizon)

st.write(f"Feature table rows: {len(feature_table):,}")
st.write(f"X shape: {X.shape}; y length: {len(y):,}")
st.dataframe(feature_table.head(20), use_container_width=True)

if feature_table.empty:
    st.warning("The feature table is empty. Reduce the horizon or avoid coarse resampling.")
else:
    st.line_chart(
        prepared_df.set_index(timestamp_col)[target_col].head(1000),
        height=260,
    )

st.header("6) STUDENT ADDITIONS — MODELING")
st.info("Add your own time-based split, forecasting model, prediction table, and metrics table here. Keep results in a pandas DataFrame named results_df.")

# Default values keep the starter runnable before the student adds modeling.
results_df = None
has_time_based_split = False
student_modeling_notes = "Modeling and metrics have not been added yet."

st.code(
    """
# Paste modeling work below this marker in app.py.
# Required student output:
# - a time-based train/test split
# - at least one forecasting model
# - a metrics table assigned to results_df
# - evidence that predictions were made without leaking future values

results_df = None
has_time_based_split = False
student_modeling_notes = "Replace these defaults after adding model and evaluation code."
""",
    language="python",
)

st.header("7) STUDENT ADDITIONS — DASHBOARD")
st.info("Add extra plots, KPIs, diagnostics, or insight text here.")

# Safe defaults for export. Student dashboard code can overwrite these.
student_added_dashboard = False
student_dashboard_insights = []

st.code(
    """
# Paste dashboard additions below this marker in app.py.
# Examples:
# - actual vs predicted plot
# - residual plot
# - daily or monthly error summary
# - written insights and limitations

student_added_dashboard = False
student_dashboard_insights = []
""",
    language="python",
)

# Dashboard addition requested by the student: descriptive PV diagnostics only.
st.subheader("Student Dashboard: PV Power Diagnostics")

dash_df = prepared_df[[timestamp_col, target_col]].dropna().copy()
dash_df = dash_df.sort_values(timestamp_col)

if dash_df.empty:
    st.warning("Dashboard cannot be created because the prepared time-series data is empty.")
else:
    latest_value = float(dash_df[target_col].iloc[-1])
    mean_value = float(dash_df[target_col].mean())
    max_value = float(dash_df[target_col].max())
    zero_pct = float((dash_df[target_col] <= 0).mean() * 100)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Latest power", f"{latest_value:,.1f} W")
    kpi2.metric("Average power", f"{mean_value:,.1f} W")
    kpi3.metric("Maximum power", f"{max_value:,.1f} W")
    kpi4.metric("Zero/negative power", f"{zero_pct:.1f}%")

    st.markdown("### Recent target trend")
    recent_n = min(1000, len(dash_df))
    st.line_chart(
        dash_df.set_index(timestamp_col)[target_col].tail(recent_n),
        height=280,
    )

    st.markdown("### Average power by hour of day")
    hourly_profile = (
        dash_df.assign(hour=dash_df[timestamp_col].dt.hour)
        .groupby("hour", as_index=False)[target_col]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hourly_profile["hour"], hourly_profile[target_col], marker="o")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(target_col)
    ax.set_title("Average PV power by hour")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Daily power summary")
    daily_summary = (
        dash_df.set_index(timestamp_col)[target_col]
        .resample("D")
        .agg(["mean", "max", "count"])
        .dropna()
        .tail(30)
        .reset_index()
    )
    st.dataframe(daily_summary, use_container_width=True)

    time_diffs = dash_df[timestamp_col].diff().dropna()
    median_gap = time_diffs.median() if len(time_diffs) else pd.NaT
    large_gap_count = 0
    if pd.notna(median_gap) and median_gap.total_seconds() > 0:
        large_gap_count = int((time_diffs > 3 * median_gap).sum())

    student_dashboard_insights = [
        f"The target power ranges from {dash_df[target_col].min():,.1f} W to {dash_df[target_col].max():,.1f} W.",
        f"The average target power is {mean_value:,.1f} W after cleaning and optional resampling.",
        "The hourly profile shows the expected solar pattern, with generation concentrated during daylight hours.",
        f"The dashboard detected {large_gap_count} unusually large time gaps after preparation.",
        "A limitation is that this dashboard is descriptive; prediction quality still requires a forecasting model and metrics table.",
    ]

    st.markdown("### Insights and limitations")
    for insight in student_dashboard_insights:
        st.write(f"- {insight}")

    student_added_dashboard = True

st.header("8) Export submission files")

has_metrics_table = isinstance(results_df, pd.DataFrame) and not results_df.empty
results_table = to_records_or_empty(results_df)

submission = {
    "student": {
        "name": student_name,
        "id": student_id,
    },
    "project": {
        "title": project_title,
        "goal": project_goal,
        "streamlit_url": deployed_url,
        "github_repo_url": repo_url,
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
        "outliers_discussed": False,
        "resampling_discussed": bool(resample_rule),
    },
    "features": {
        "baseline_features": feature_cols,
        "feature_table_rows": int(len(feature_table)),
        "student_added_features": [],
    },
    "modeling_and_evaluation": {
        "has_time_based_split": bool(has_time_based_split),
        "has_metrics_table": bool(has_metrics_table),
        "results_table": results_table,
        "student_notes": student_modeling_notes,
    },
    "dashboard": {
        "has_baseline_plot": True,
        "has_student_added_dashboard": bool(student_added_dashboard),
        "insights": list(student_dashboard_insights),
    },
}

submission_json = safe_json_dumps(submission)

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
- Rows after cleaning: {len(cleaned_df):,}
- Resampling: {resample_rule}
- Forecast horizon rows: {int(horizon)}

## Baseline features prepared
{", ".join(feature_cols)}

## Dashboard additions
- Student-added dashboard present: {bool(student_added_dashboard)}
- Insights count: {len(student_dashboard_insights)}

## Student additions still required
- Add a time-based train/test split.
- Add at least one forecasting model.
- Add a metrics table assigned to `results_df`.
- Add final interpretation of forecasting performance and limitations.

## Links
- Streamlit app: {deployed_url}
- GitHub repo: {repo_url}
"""

download_col1, download_col2 = st.columns(2)
with download_col1:
    st.download_button(
        "Download submission.json",
        data=submission_json,
        file_name="submission.json",
        mime="application/json",
    )
with download_col2:
    st.download_button(
        "Download project_card.md",
        data=project_card,
        file_name="project_card.md",
        mime="text/markdown",
    )

with st.expander("Preview submission.json", expanded=False):
    st.code(submission_json, language="json")

with st.expander("Preview project_card.md", expanded=False):
    st.markdown(project_card)

st.header("9) AI grader /80")
st.caption(f"Model: {OPENROUTER_MODEL}")

api_key = get_openrouter_api_key()
grader_prompt = AI_GRADER_PROMPT_TEMPLATE.replace(
    "<insert submission.json contents here>",
    submission_json,
)

with st.expander("Preview AI grader prompt", expanded=False):
    st.code(grader_prompt)

if st.button("Run AI grader"):
    if not api_key:
        st.error("Provide OPENROUTER_API_KEY using Streamlit Secrets, environment variable, or the password field.")
    else:
        try:
            with st.spinner("Calling AI grader..."):
                raw_output = call_openrouter_grader(api_key, grader_prompt)
            parsed_output, parse_error = parse_grader_response(raw_output)
            if parsed_output is not None:
                st.success("Parsed grader JSON")
                st.json(parsed_output)
            else:
                st.warning(f"Could not parse grader response as JSON: {parse_error}")
                st.code(raw_output)
        except Exception as exc:
            st.error(f"AI grader call failed: {exc}")

st.divider()
st.caption("Starter stops before model training and scoring. Students must add forecasting models, metrics, evidence, and final interpretation under the marked sections.")
