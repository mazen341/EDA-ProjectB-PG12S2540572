import json
import os
import re
import time
from pathlib import Path

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

# Multiple free models to try as fallbacks when one is rate-limited
OPENROUTER_MODELS = [
    "openai/gpt-oss-20b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen-2.5-72b-instruct:free",
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


st.set_page_config(
    page_title="Mini Project B — Time-Series Forecasting Starter",
    page_icon="📈",
    layout="wide",
)


def safe_json_dumps(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


@st.cache_data(show_spinner=False)
def load_dataset(csv_path):
    return pd.read_csv(csv_path)


def dataframe_audit(df):
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "non_null_count": [int(df[col].notna().sum()) for col in df.columns],
            "missing_pct": [round(float(df[col].isna().mean() * 100), 3) for col in df.columns],
            "unique_count": [int(df[col].nunique(dropna=True)) for col in df.columns],
        }
    )


def likely_datetime_columns(df):
    rows = []
    for col in df.columns:
        name_score = int(any(token in col.lower() for token in ["time", "date", "timestamp", "datetime"]))
        parsed = pd.to_datetime(df[col], errors="coerce")
        valid_ratio = float(parsed.notna().mean())
        score = name_score + valid_ratio
        if name_score or valid_ratio > 0.70:
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


def numeric_target_candidates(df):
    rows = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        valid_ratio = float(converted.notna().mean())
        missing_pct = 100 - valid_ratio * 100
        unique_count = int(converted.nunique(dropna=True))
        is_id_like = "id" in col.lower() or unique_count >= max(50, int(0.95 * len(df)))
        if valid_ratio >= 0.50 and unique_count > 2 and not is_id_like:
            rows.append(
                {
                    "column": col,
                    "missing_pct": round(missing_pct, 3),
                    "mean": round(float(converted.mean()), 4),
                    "std": round(float(converted.std()), 4),
                    "unique_count": unique_count,
                }
            )
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
    result = (
        numeric_df.set_index(timestamp_col)[numeric_cols]
        .resample(rule)
        .mean()
        .reset_index()
    )
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


def make_baseline_features(df, timestamp_col, target_col, horizon):
    work = df[[timestamp_col, target_col]].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[timestamp_col, target_col]).sort_values(timestamp_col)

    work["lag_1"] = work[target_col].shift(1)
    work["lag_24"] = work[target_col].shift(24)
    work["rolling_mean_24"] = work[target_col].shift(1).rolling(24).mean()
    work["hour"] = work[timestamp_col].dt.hour
    work["weekend"] = work[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)
    work["month"] = work[timestamp_col].dt.month
    work["y_target"] = work[target_col].shift(-int(horizon))

    feature_cols = ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"]
    feature_table = work.dropna(subset=feature_cols + ["y_target"]).reset_index(drop=True)
    X = feature_table[feature_cols].copy()
    y = feature_table["y_target"].copy()
    return feature_table, X, y, feature_cols


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

    return st.text_input(
        "OpenRouter API key",
        type="password",
        help="Used only for the AI grader call. It is not stored in the app code.",
    )


def parse_grader_response(raw_text):
    try:
        return json.loads(raw_text), None
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), None
        except Exception as exc:
            return None, str(exc)
    return None, "No JSON object found in model response."


def call_openrouter_grader(
    api_key,
    prompt,
    model,
    max_retries=4,
    base_backoff=5,
    status_callback=None,
):
    """
    Call OpenRouter with retry logic for 429 (rate limit) and 5xx errors.

    - Respects the `Retry-After` header when present.
    - Uses exponential backoff (5s, 10s, 20s, 40s) otherwise.
    - Surfaces the upstream error body so the cause is visible.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenRouter recommends these for free-tier identification
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "Mini Project B Grader",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=90)
        except requests.RequestException as exc:
            last_error = f"Network error on attempt {attempt}: {exc}"
            if status_callback:
                status_callback(last_error)
            time.sleep(base_backoff * (2 ** (attempt - 1)))
            continue

        # Success
        if response.status_code == 200:
            payload = response.json()
            try:
                return payload["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as exc:
                raise RuntimeError(
                    f"Unexpected response shape from OpenRouter: {payload}"
                ) from exc

        # Rate limited — honor Retry-After if present, otherwise backoff
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            try:
                wait_seconds = float(retry_after) if retry_after else base_backoff * (2 ** (attempt - 1))
            except ValueError:
                wait_seconds = base_backoff * (2 ** (attempt - 1))
            # Cap waits to keep the UI responsive
            wait_seconds = min(wait_seconds, 60)

            try:
                err_body = response.json()
            except Exception:
                err_body = response.text[:300]
            last_error = (
                f"429 Too Many Requests on `{model}` "
                f"(attempt {attempt}/{max_retries}). "
                f"Waiting {wait_seconds:.1f}s before retry. Details: {err_body}"
            )
            if status_callback:
                status_callback(last_error)

            if attempt < max_retries:
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(last_error)

        # Transient server errors — backoff and retry
        if 500 <= response.status_code < 600:
            wait_seconds = base_backoff * (2 ** (attempt - 1))
            try:
                err_body = response.json()
            except Exception:
                err_body = response.text[:300]
            last_error = (
                f"{response.status_code} server error on `{model}` "
                f"(attempt {attempt}/{max_retries}). "
                f"Waiting {wait_seconds:.1f}s. Details: {err_body}"
            )
            if status_callback:
                status_callback(last_error)
            if attempt < max_retries:
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(last_error)

        # Other client errors are not worth retrying
        try:
            err_body = response.json()
        except Exception:
            err_body = response.text[:500]
        raise RuntimeError(
            f"{response.status_code} error from OpenRouter on `{model}`: {err_body}"
        )

    raise RuntimeError(last_error or "AI grader call failed after retries.")


def call_grader_with_model_fallback(api_key, prompt, models, status_callback=None):
    """Try each model in order. If one fails after all its retries, move to the next."""
    errors = []
    for model in models:
        if status_callback:
            status_callback(f"Trying model: `{model}` ...")
        try:
            return call_openrouter_grader(
                api_key, prompt, model, status_callback=status_callback
            ), model
        except Exception as exc:
            errors.append(f"`{model}` → {exc}")
            if status_callback:
                status_callback(f"Falling back from `{model}`: {exc}")
            continue
    raise RuntimeError(
        "All models failed. Errors:\n" + "\n".join(errors)
    )


st.title("Mini Project B — Time-Series Forecasting Starter")
st.caption("This starter prepares the dataset, baseline feature table, exports, and AI grader. Students add forecasting models and dashboard improvements.")

with st.sidebar:
    st.header("1) Student information")
    student_name = st.text_input("Student name", value=STUDENT_NAME_DEFAULT)
    student_id = st.text_input("Student ID", value=STUDENT_ID_DEFAULT)
    deployed_url = st.text_input("Deployed Streamlit app URL", value="")
    repo_url = st.text_input("GitHub repo URL", value="")
    project_title = st.text_input("Project title", value="HKUST SQ1 PV Power Forecasting")
    project_goal = st.text_area(
        "Project goal",
        value="Forecast inverter total active AC power from a cleaned time-series dataset using time-aware baseline features.",
        height=100,
    )

st.header("2) Load local dataset")
data_path = st.text_input("Dataset path", value=DEFAULT_DATA_PATH)

try:
    df = load_dataset(data_path)
except Exception as exc:
    st.error(f"Could not load dataset from {data_path}: {exc}")
    st.stop()

st.success(f"Loaded dataset with {len(df):,} rows and {len(df.columns):,} columns.")
st.subheader("First 10 rows")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Dataset audit")
audit = dataframe_audit(df)
st.dataframe(audit, use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    st.write("Likely timestamp columns")
    st.dataframe(likely_datetime_columns(df).head(3), use_container_width=True)
with col_b:
    st.write("Likely numeric target columns")
    st.dataframe(numeric_target_candidates(df).head(3), use_container_width=True)

st.header("3) Select timestamp and target")
default_ts_index = df.columns.get_loc(DEFAULT_TIMESTAMP_COL) if DEFAULT_TIMESTAMP_COL in df.columns else 0
timestamp_col = st.selectbox("Timestamp column", options=list(df.columns), index=int(default_ts_index))

numeric_like_cols = [
    col for col in df.columns
    if pd.to_numeric(df[col], errors="coerce").notna().mean() >= 0.50
]
if DEFAULT_TARGET_COL not in numeric_like_cols and DEFAULT_TARGET_COL in df.columns:
    numeric_like_cols.insert(0, DEFAULT_TARGET_COL)
default_target_index = numeric_like_cols.index(DEFAULT_TARGET_COL) if DEFAULT_TARGET_COL in numeric_like_cols else 0
target_col = st.selectbox("Target column", options=numeric_like_cols, index=int(default_target_index))

cleaned_df, cleaning_report = clean_time_series(df, timestamp_col, target_col)
coverage = infer_time_coverage(cleaned_df, timestamp_col)

st.subheader("Cleaned time-series summary")
st.json({**cleaning_report, **coverage})

if cleaned_df.empty:
    st.error("No valid rows remain after parsing the timestamp and target columns.")
    st.stop()

st.header("4) Optional resampling and forecast horizon")
resample_rule = st.selectbox(
    "Resampling rule",
    options=["No resampling", "5min", "15min", "30min", "1H", "1D"],
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

st.write("Prepared time-series coverage")
st.json(prepared_coverage)

st.header("5) Baseline feature table")
feature_table, X, y, feature_cols = make_baseline_features(prepared_df, timestamp_col, target_col, horizon)

st.write(f"Feature table rows: {len(feature_table):,}")
st.write(f"X shape: {X.shape}; y length: {len(y):,}")
st.dataframe(feature_table.head(20), use_container_width=True)

st.line_chart(
    prepared_df.set_index(timestamp_col)[target_col].head(1000),
    height=260,
)

st.header("6) STUDENT ADDITIONS — MODELING")
st.info("Add your own time-based split, forecasting model, prediction table, and metrics table here. Keep results in a pandas DataFrame named results_df.")
results_df = None

st.code(
    """
# Paste your modeling work below this marker in app.py.
# Required student output:
# - a time-based train/test split
# - at least one forecasting model
# - a metrics table assigned to results_df
# - evidence that predictions were made without leaking future values

results_df = None
""",
    language="python",
)

st.header("7) STUDENT ADDITIONS — DASHBOARD")
st.info("Add extra plots, KPIs, diagnostics, or insight text here.")

st.code(
    """
# Paste dashboard additions below this marker in app.py.
# Examples:
# - actual vs predicted plot
# - residual plot
# - daily or monthly error summary
# - written insights and limitations
""",
    language="python",
)

st.header("8) Export submission files")

has_metrics_table = isinstance(results_df, pd.DataFrame)
results_table = [] if results_df is None else results_df.to_dict(orient="records")

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
        "missing_values_discussed": bool(cleaning_report["invalid_timestamp_rows_removed"] >= 0),
        "resampling_discussed": bool(resample_rule),
    },
    "features": {
        "baseline_features": feature_cols,
        "feature_table_rows": int(len(feature_table)),
        "student_added_features": [],
    },
    "modeling_and_evaluation": {
        "has_time_based_split": False,
        "has_metrics_table": has_metrics_table,
        "results_table": results_table,
        "student_notes": "Replace these defaults after adding model and evaluation code.",
    },
    "dashboard": {
        "has_baseline_plot": True,
        "has_student_added_dashboard": False,
        "insights": [],
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

## Student additions still required
- Add time-based train/test split.
- Add at least one forecasting model.
- Add metrics table assigned to `results_df`.
- Add extra dashboard plots/KPIs and written insights.

## Links
- Streamlit app: {deployed_url}
- GitHub repo: {repo_url}
"""

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "Download submission.json",
        data=submission_json,
        file_name="submission.json",
        mime="application/json",
    )
with col2:
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

# Model selector with fallback option
col_m1, col_m2 = st.columns([2, 1])
with col_m1:
    selected_model = st.selectbox(
        "Primary model",
        options=OPENROUTER_MODELS,
        index=0,
        help="Free models share rate limits. If one returns 429, try another.",
    )
with col_m2:
    enable_fallback = st.checkbox(
        "Auto-fallback to other free models",
        value=True,
        help="If the selected model is rate-limited after retries, try the others in the list.",
    )

api_key = get_openrouter_api_key()
grader_prompt = AI_GRADER_PROMPT_TEMPLATE.replace(
    "<insert submission.json contents here>",
    submission_json,
)

with st.expander("Preview AI grader prompt", expanded=False):
    st.code(grader_prompt)

st.caption(
    "Note on 429 errors: OpenRouter's `:free` models share a small per-minute and "
    "per-day quota across all users. If you hit the limit, wait ~60 seconds, switch "
    "model, or add credits to your OpenRouter account for a paid model."
)

if st.button("Run AI grader"):
    if not api_key:
        st.error("Provide OPENROUTER_API_KEY using Streamlit Secrets, environment variable, or the password field.")
    else:
        status_box = st.empty()
        log_messages = []

        def log_status(msg):
            log_messages.append(msg)
            status_box.info("\n\n".join(log_messages[-5:]))

        # Build the model list: primary first, then the rest if fallback is on
        if enable_fallback:
            ordered_models = [selected_model] + [m for m in OPENROUTER_MODELS if m != selected_model]
        else:
            ordered_models = [selected_model]

        try:
            with st.spinner("Calling AI grader (with retries and fallbacks)..."):
                raw_output, used_model = call_grader_with_model_fallback(
                    api_key, grader_prompt, ordered_models, status_callback=log_status
                )
            st.success(f"Got response from `{used_model}`")
            parsed_output, parse_error = parse_grader_response(raw_output)
            if parsed_output is not None:
                st.success("Parsed grader JSON")
                st.json(parsed_output)
            else:
                st.warning(f"Could not parse grader response as JSON: {parse_error}")
                st.code(raw_output)
        except Exception as exc:
            st.error(f"AI grader call failed: {exc}")
            st.info(
                "Suggestions:\n"
                "1. Wait 60 seconds and try again — free-tier limits reset quickly.\n"
                "2. Switch to a different free model in the dropdown above.\n"
                "3. Add credits at https://openrouter.ai/credits and use a paid model "
                "(e.g. `openai/gpt-4o-mini`) for reliable access."
            )

st.divider()
st.caption("Starter stops before model training and scoring. Students must add forecasting models, metrics, dashboard evidence, and insights under the marked sections.")
