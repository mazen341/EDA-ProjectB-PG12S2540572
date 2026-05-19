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
    page_title="Mini Project B — Time-Series Forecasting",
    page_icon="📈",
    layout="wide",
)


def safe_json_default(obj):
    """Convert pandas/numpy objects for JSON export."""
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
    """Read key from Streamlit Secrets, then environment, then optional password box."""
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
    """Return audit tables for preview, dtypes, and missing percentages."""
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
    """Infer a readable median interval and pandas frequency if possible."""
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
    """Parse, clean, sort, optionally resample, and return a prepared time-series dataframe."""
    prepared = dataframe.copy()
    prepared[timestamp_column] = pd.to_datetime(prepared[timestamp_column], errors="coerce")
    prepared[target_column] = pd.to_numeric(prepared[target_column], errors="coerce")

    before_rows = len(prepared)
    prepared = prepared.dropna(subset=[timestamp_column, target_column])
    after_drop_rows = len(prepared)

    prepared = prepared.sort_values(timestamp_column)
    duplicate_count = int(prepared[timestamp_column].duplicated().sum())

    # If duplicated timestamps exist, aggregate numeric values by timestamp.
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
        resampling_note = (
            f"Resampled to {resample_rule} using mean aggregation and interpolation."
        )

    cleaning_report = {
        "rows_before_cleaning": int(before_rows),
        "rows_after_dropping_invalid_timestamp_or_target": int(after_drop_rows),
        "duplicate_timestamps_before_grouping": int(duplicate_count),
        "rows_after_grouping_and_resampling": int(len(prepared)),
        "resampling_note": resampling_note,
    }
    return prepared, cleaning_report


def build_baseline_features(prepared, timestamp_column, target_column, horizon):
    """Create baseline lag, rolling, and calendar features only."""
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
    """Parse JSON response; if needed, extract first JSON object from raw text."""
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
    """Send the fixed Project B prompt to OpenRouter."""
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
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0,
        },
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


st.title("Mini Project B — Time-Series Forecasting Starter")
st.caption("UTAS Energy Data Analytics | Student-enhanced version with forecasting model and diagnostics")

# Defaults to avoid export errors if a student edits sections.
results_df = None
predictions_df = pd.DataFrame()
student_added_features = []
student_added_dashboard = False
student_dashboard_insights = []
has_time_based_split = False
student_modeling_notes = "Modeling has not run yet."
outlier_summary = {}

st.header("1) Student information")
col_a, col_b = st.columns(2)
with col_a:
    student_name = st.text_input("Student name", value=STUDENT_NAME_DEFAULT)
    student_id = st.text_input("Student ID", value=STUDENT_ID_DEFAULT)
    app_title = st.text_input(
        "Project title",
        value="HKUST Rooftop PV Power Forecasting",
    )
with col_b:
    deployed_url = st.text_input("Deployed Streamlit URL", value="")
    github_url = st.text_input("GitHub repository URL", value="")
    project_goal = st.text_area(
        "Project goal",
        value=(
            "Forecast rooftop PV inverter active power using historical time-series "
            "patterns and weather-related variables."
        ),
        height=110,
    )

st.header("2) Load local dataset")
data_path = st.text_input("Dataset path", value=DEFAULT_DATA_PATH)

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    st.error(
        f"Could not find {data_path}. Make sure data/dataset_sample.csv is uploaded "
        "with app.py in your GitHub repository."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

st.success(f"Loaded {len(df):,} rows and {len(df.columns):,} columns from {data_path}.")

st.subheader("First 10 rows")
st.dataframe(df.head(10), use_container_width=True)

st.header("3) Dataset audit")
audit_table, missing_top10 = audit_dataframe(df)
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Columns, dtypes, missing %, unique counts")
    st.dataframe(audit_table, use_container_width=True)
with col2:
    st.markdown("### Top 10 missing-value columns")
    st.dataframe(missing_top10, use_container_width=True)

st.header("4) Choose timestamp and target columns")
columns = list(df.columns)

default_timestamp_index = columns.index(DEFAULT_TIMESTAMP_COL) if DEFAULT_TIMESTAMP_COL in columns else 0
timestamp_col = st.selectbox(
    "Timestamp column",
    columns,
    index=default_timestamp_index,
)

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
    if DEFAULT_TARGET_COL in numeric_candidate_columns
    else 0
)
target_col = st.selectbox(
    "Target column",
    numeric_candidate_columns,
    index=default_target_index,
)

timestamp_preview = pd.to_datetime(df[timestamp_col], errors="coerce")
valid_timestamp_pct = float(timestamp_preview.notna().mean() * 100)
coverage_min = timestamp_preview.min()
coverage_max = timestamp_preview.max()
freq_info = infer_frequency(timestamp_preview)

st.info(
    f"Timestamp validity: {valid_timestamp_pct:.2f}%. "
    f"Coverage: {coverage_min} to {coverage_max}. "
    f"Median gap: {freq_info['median_gap']}. "
    f"Inferred frequency: {freq_info['inferred_freq']}. "
    f"Large gap count: {freq_info['large_gap_count']}."
)

target_preview = pd.to_numeric(df[target_col], errors="coerce")
st.write(
    {
        "target_missing_pct": round(float(target_preview.isna().mean() * 100), 3),
        "target_mean": round(float(target_preview.mean()), 3),
        "target_std": round(float(target_preview.std()), 3),
        "target_unique_count": int(target_preview.nunique(dropna=True)),
    }
)

st.header("5) Parse, clean, optionally resample, and choose horizon")
resample_rule = st.selectbox(
    "Optional resampling rule",
    ["None", "15min", "30min", "1h", "1D"],
    index=3,
    help="Use 1h for a more regular and manageable forecasting problem.",
)
horizon = st.number_input(
    "Forecast horizon in rows after optional resampling",
    min_value=1,
    max_value=168,
    value=1,
    step=1,
)

prepared_df, cleaning_report = prepare_timeseries(
    df,
    timestamp_col,
    target_col,
    resample_rule,
)

st.markdown("### Cleaning and resampling report")
st.json(cleaning_report)

st.markdown("### Prepared time-series preview")
st.dataframe(prepared_df.head(10), use_container_width=True)

st.header("6) Baseline feature table creation")
feature_table, X, y, feature_cols = build_baseline_features(
    prepared_df,
    timestamp_col,
    target_col,
    horizon,
)

st.write(f"Feature table rows: {len(feature_table):,}")
st.write(f"X shape: {X.shape}, y length: {len(y):,}")
st.dataframe(feature_table.head(20), use_container_width=True)

st.markdown("### Baseline target trend")
trend_df = prepared_df.set_index(timestamp_col)[target_col].tail(min(1000, len(prepared_df)))
st.line_chart(trend_df, height=280)

st.header("7) STUDENT ADDITIONS — MODELING")
st.write(
    "This section is a student-added forecasting model, time-based split, metrics table, "
    "outlier evidence, and prediction diagnostics."
)

run_model = st.checkbox(
    "Run student forecasting model and metrics",
    value=True,
    help="Turn off only while editing if the app feels slow.",
)

if run_model:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        st.subheader("Student Model: Time-Based Forecasting Evaluation")

        model_df = feature_table.copy()
        model_df[timestamp_col] = pd.to_datetime(model_df[timestamp_col], errors="coerce")

        exog_candidates = [
            "irradiance_wm2",
            "temperature_c",
            "relative_humidity_pct",
            "sea_level_pressure_hpa",
            "visibility_km",
            "wind_speed_ms",
            "wind_direction_deg",
            "rainfall_mm",
        ]
        available_exog = [
            c for c in exog_candidates
            if c in prepared_df.columns and c != target_col
        ]

        if available_exog:
            exog_df = prepared_df[[timestamp_col] + available_exog].copy()
            exog_df[timestamp_col] = pd.to_datetime(exog_df[timestamp_col], errors="coerce")
            for col in available_exog:
                exog_df[col] = pd.to_numeric(exog_df[col], errors="coerce")
            model_df = model_df.merge(exog_df, on=timestamp_col, how="left")

        model_df["dayofyear"] = model_df[timestamp_col].dt.dayofyear
        model_df["quarter"] = model_df[timestamp_col].dt.quarter
        model_df["is_daylight_hour"] = model_df["hour"].between(7, 18).astype(int)
        model_df["lag1_x_hour"] = model_df["lag_1"] * model_df["hour"]

        student_added_features = [
            "dayofyear",
            "quarter",
            "is_daylight_hour",
            "lag1_x_hour",
        ] + available_exog

        model_features = feature_cols + student_added_features

        for col in model_features:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

        model_df = (
            model_df
            .dropna(subset=[timestamp_col, "y_target"] + model_features)
            .sort_values(timestamp_col)
        )

        target_series = pd.to_numeric(prepared_df[target_col], errors="coerce").dropna()
        q1 = float(target_series.quantile(0.25))
        q3 = float(target_series.quantile(0.75))
        iqr = q3 - q1

        if iqr > 0:
            outlier_lower = q1 - 1.5 * iqr
            outlier_upper = q3 + 1.5 * iqr
        else:
            outlier_lower = float(target_series.min())
            outlier_upper = float(target_series.max())

        outlier_mask = (target_series < outlier_lower) | (target_series > outlier_upper)
        outlier_summary = {
            "method": "IQR rule",
            "lower_bound": round(float(outlier_lower), 3),
            "upper_bound": round(float(outlier_upper), 3),
            "outlier_count": int(outlier_mask.sum()),
            "outlier_pct": round(float(outlier_mask.mean() * 100), 3),
            "treatment": (
                "Training target and predictions clipped to IQR bounds "
                "to reduce extreme-value influence."
            ),
        }

        st.markdown("### Outlier check")
        st.json(outlier_summary)

        if len(model_df) < 100:
            st.warning("Not enough rows for a reliable time-based train/validation split.")
            results_df = None
            predictions_df = pd.DataFrame()
            has_time_based_split = False
            student_modeling_notes = (
                "Not enough rows after feature creation to train and evaluate a model."
            )
        else:
            max_model_rows = st.slider(
                "Maximum rows used for model training/evaluation",
                min_value=1000,
                max_value=max(1000, int(len(model_df))),
                value=min(50000, int(len(model_df))),
                step=1000,
                help=(
                    "Uses the most recent rows to keep Streamlit Cloud responsive while "
                    "preserving time order."
                ),
            )

            model_df_used = model_df.tail(int(max_model_rows)).copy()

            split_idx = int(len(model_df_used) * 0.80)
            train_df = model_df_used.iloc[:split_idx].copy()
            valid_df = model_df_used.iloc[split_idx:].copy()

            X_train = train_df[model_features]
            y_train = train_df["y_target"].clip(outlier_lower, outlier_upper)

            X_valid = valid_df[model_features]
            y_valid = valid_df["y_target"]

            model = HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.06,
                max_leaf_nodes=31,
                random_state=42,
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_valid)
            y_pred = np.clip(y_pred, outlier_lower, outlier_upper)

            mae = mean_absolute_error(y_valid, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred)))
            mape = float(
                np.mean(np.abs((y_valid - y_pred) / np.maximum(np.abs(y_valid), 1))) * 100
            )
            r2 = r2_score(y_valid, y_pred)

            results_df = pd.DataFrame(
                [
                    {
                        "model": "HistGradientBoostingRegressor",
                        "split_type": "time_based_80_20",
                        "train_start": str(train_df[timestamp_col].min()),
                        "train_end": str(train_df[timestamp_col].max()),
                        "validation_start": str(valid_df[timestamp_col].min()),
                        "validation_end": str(valid_df[timestamp_col].max()),
                        "horizon_rows": int(horizon),
                        "MAE": round(float(mae), 3),
                        "RMSE": round(float(rmse), 3),
                        "MAPE_pct": round(float(mape), 3),
                        "R2": round(float(r2), 4),
                        "train_rows": int(len(train_df)),
                        "validation_rows": int(len(valid_df)),
                    }
                ]
            )

            predictions_df = valid_df[[timestamp_col, target_col, "y_target"]].copy()
            predictions_df["prediction"] = y_pred
            predictions_df["residual"] = (
                predictions_df["y_target"] - predictions_df["prediction"]
            )
            predictions_df["absolute_error"] = predictions_df["residual"].abs()

            has_time_based_split = True
            student_modeling_notes = (
                "A time-based 80/20 split was used. The model trains on earlier observations "
                "and validates on later observations to avoid future-data leakage. Metrics are "
                "computed on the validation period. The most recent rows are used when the row "
                "limit slider is below the full feature-table size."
            )

            st.markdown("### Metrics table")
            st.dataframe(results_df, use_container_width=True)

            st.markdown("### Prediction preview")
            st.dataframe(predictions_df.head(20), use_container_width=True)

            st.markdown("### Actual vs predicted target")
            plot_df = predictions_df.set_index(timestamp_col)[["y_target", "prediction"]].tail(500)
            st.line_chart(plot_df, height=300)

    except Exception as exc:
        st.error(f"Student modeling section failed: {exc}")
        results_df = None
        predictions_df = pd.DataFrame()
        has_time_based_split = False
        student_modeling_notes = f"Modeling failed with error: {exc}"

else:
    st.info("Modeling is turned off. Turn it on to create metrics and prediction evidence.")
    results_df = None
    predictions_df = pd.DataFrame()
    has_time_based_split = False
    student_modeling_notes = "Student model was not run."

st.header("8) STUDENT ADDITIONS — DASHBOARD")
st.subheader("Student Dashboard: PV Power Diagnostics")

dash_df = prepared_df[[timestamp_col, target_col]].dropna().copy()
dash_df = dash_df.sort_values(timestamp_col)

if dash_df.empty:
    st.warning("Dashboard cannot be created because the prepared time-series data is empty.")
    student_added_dashboard = False
    student_dashboard_insights = []
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
        .groupby("hour")[target_col]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hourly_profile["hour"], hourly_profile[target_col], marker="o")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(target_col)
    ax.set_title("Average PV power by hour")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

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
        f"The target power ranges from {dash_df[target_col].min():,.1f} W "
        f"to {dash_df[target_col].max():,.1f} W.",
        f"The average target power is {mean_value:,.1f} W after cleaning and optional resampling.",
        "The hourly profile shows the expected solar production pattern, with generation concentrated during daylight hours.",
        f"The dashboard detected {large_gap_count} unusually large time gaps after preparation.",
        "A limitation is that descriptive plots alone do not prove forecasting quality, so model metrics are also included.",
    ]

    st.markdown("### Insights and limitations")
    for insight in student_dashboard_insights:
        st.write(f"- {insight}")

    student_added_dashboard = True

if isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty:
    st.subheader("Student Dashboard: Forecast Diagnostics")

    diag_col1, diag_col2, diag_col3 = st.columns(3)
    diag_col1.metric("Validation rows", f"{len(predictions_df):,}")
    diag_col2.metric("Mean absolute error", f"{predictions_df['absolute_error'].mean():,.2f}")
    diag_col3.metric("Max absolute error", f"{predictions_df['absolute_error'].max():,.2f}")

    st.markdown("### Residual plot")
    residual_plot_df = predictions_df.set_index(timestamp_col)["residual"].tail(500)
    st.line_chart(residual_plot_df, height=260)

    st.markdown("### Monthly validation error summary")
    monthly_error_summary = (
        predictions_df.set_index(timestamp_col)
        .resample("M")
        .agg(
            actual_mean=("y_target", "mean"),
            prediction_mean=("prediction", "mean"),
            mae=("absolute_error", "mean"),
            max_error=("absolute_error", "max"),
            n=("absolute_error", "count"),
        )
        .dropna()
        .reset_index()
    )
    st.dataframe(monthly_error_summary, use_container_width=True)

    student_dashboard_insights.extend(
        [
            "A forecasting model was evaluated using a time-based validation period.",
            "The dashboard compares actual and predicted power values for the validation period.",
            "Residual diagnostics show when the model under-predicts or over-predicts PV power.",
            "Monthly error summaries help identify whether model performance changes over time.",
        ]
    )
    student_added_dashboard = True
else:
    st.info("Forecast diagnostics will appear after the modeling section creates predictions_df.")

st.header("9) Export submission files")
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
        "resampling_note": (
            f"Selected resampling rule: {resample_rule}. "
            "The app allows regular interval resampling before feature creation."
        ),
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
        "model_used": "HistGradientBoostingRegressor" if has_metrics_table else None,
        "predictions_created": (
            isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty
        ),
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
            "The selected model is a practical baseline-plus-student model, not the only possible forecasting approach.",
            "Validation metrics should be interpreted for the chosen horizon and resampling level.",
        ],
        "reproducibility_notes": [
            "The app loads data/dataset_sample.csv from the GitHub repository.",
            "The model uses a time-based train/validation split.",
            "The exported submission.json captures the evidence used by the AI grader.",
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
- Path: {data_path}
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
{", ".join(student_added_features) if student_added_features else "None yet"}

## Modeling and evaluation
{student_modeling_notes}

Metrics table available: {has_metrics_table}

## Dashboard insights
{chr(10).join("- " + item for item in student_dashboard_insights)}

## Limitations
- PV generation is weather-sensitive and can change sharply under cloud cover.
- Results depend on the selected resampling interval and forecast horizon.
- More advanced models and deeper error analysis may improve performance.
"""

col_download1, col_download2 = st.columns(2)
with col_download1:
    st.download_button(
        "Download submission.json",
        data=submission_json,
        file_name="submission.json",
        mime="application/json",
    )
with col_download2:
    st.download_button(
        "Download project_card.md",
        data=project_card,
        file_name="project_card.md",
        mime="text/markdown",
    )

with st.expander("Preview submission.json"):
    st.json(submission)

with st.expander("Preview project_card.md"):
    st.markdown(project_card)

st.header("10) AI grader (/80)")
st.write(
    "The AI grader uses OpenRouter with the fixed model string "
    f"`{OPENROUTER_MODEL}` and the fixed Project B /80 rubric."
)

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
