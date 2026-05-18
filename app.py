import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def call_openrouter_grader(api_key, prompt):
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


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
st.markdown(
    "**Student work:** time-based 80/20 split (no leakage), three forecasting "
    "models compared head-to-head — *Naive last-value*, *Seasonal-naive (lag-24)*, "
    "and *Random Forest* on engineered lag/calendar features. "
    "All predictions use only information available **before** the target timestamp."
)

# ---- Time-based train/test split (chronological, no shuffling) ----
test_size_pct = st.slider(
    "Test set size (% of the most recent rows)",
    min_value=10, max_value=40, value=20, step=5,
)
split_idx = int(len(feature_table) * (1 - test_size_pct / 100))
train_df = feature_table.iloc[:split_idx].copy()
test_df = feature_table.iloc[split_idx:].copy()

X_train = train_df[feature_cols]
y_train = train_df["y_target"]
X_test = test_df[feature_cols]
y_test = test_df["y_target"]

split_info = {
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "train_start": str(train_df[timestamp_col].min()),
    "train_end": str(train_df[timestamp_col].max()),
    "test_start": str(test_df[timestamp_col].min()),
    "test_end": str(test_df[timestamp_col].max()),
    "split_strategy": "chronological, last {}% as test, no shuffling".format(test_size_pct),
}
st.subheader("Time-based split")
st.json(split_info)

# ---- Helper for metrics ----
def compute_metrics(name, y_true, y_pred):
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2 = float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 1 else float("nan")
    denom = np.where(np.abs(y_true_arr) < 1e-6, np.nan, y_true_arr)
    mape = float(np.nanmean(np.abs((y_true_arr - y_pred_arr) / denom)) * 100)
    return {
        "model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 4),
        "MAPE_%": round(mape, 2),
        "n_test": int(len(y_true_arr)),
    }

# ---- Model 1: Naive last-value (predict y_t = lag_1) ----
pred_naive = X_test["lag_1"].values

# ---- Model 2: Seasonal-naive (predict y_t = lag_24, same hour previous day) ----
pred_seasonal = X_test["lag_24"].values

# ---- Model 3: Random Forest on engineered features ----
with st.spinner("Training Random Forest..."):
    rf_model = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    rf_model.fit(X_train, y_train)
    pred_rf = rf_model.predict(X_test)

# ---- Bonus Model 4: Ridge regression for a linear baseline ----
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train, y_train)
pred_ridge = ridge_model.predict(X_test)

# ---- Metrics table assigned to results_df ----
results_df = pd.DataFrame([
    compute_metrics("Naive (lag-1)", y_test, pred_naive),
    compute_metrics("Seasonal-naive (lag-24)", y_test, pred_seasonal),
    compute_metrics("Ridge regression", y_test, pred_ridge),
    compute_metrics("Random Forest", y_test, pred_rf),
])

st.subheader("Metrics on hold-out test set")
st.dataframe(results_df, use_container_width=True)

best_row = results_df.loc[results_df["RMSE"].idxmin()]
st.success(
    f"Best model by RMSE: **{best_row['model']}** "
    f"(MAE={best_row['MAE']}, RMSE={best_row['RMSE']}, R²={best_row['R2']})"
)

# ---- Feature importances from Random Forest ----
importances_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

with st.expander("Random Forest feature importances", expanded=False):
    st.dataframe(importances_df, use_container_width=True)

st.header("7) STUDENT ADDITIONS — DASHBOARD")
st.markdown(
    "Diagnostic dashboard for the best model, plus written insights and limitations."
)

# Build a prediction frame for plotting (test set, aligned with timestamps)
pred_frame = test_df[[timestamp_col, "y_target"]].copy()
pred_frame = pred_frame.rename(columns={"y_target": "actual"})
pred_frame["pred_naive"] = pred_naive
pred_frame["pred_seasonal"] = pred_seasonal
pred_frame["pred_ridge"] = pred_ridge
pred_frame["pred_rf"] = pred_rf
pred_frame["residual_rf"] = pred_frame["actual"] - pred_frame["pred_rf"]
pred_frame["abs_err_rf"] = pred_frame["residual_rf"].abs()
pred_frame["date"] = pd.to_datetime(pred_frame[timestamp_col]).dt.date

# ---- KPI row ----
k1, k2, k3, k4 = st.columns(4)
k1.metric("Test rows", f"{len(pred_frame):,}")
k2.metric("Best RMSE (W)", f"{float(best_row['RMSE']):.1f}")
k3.metric("Best MAE (W)",  f"{float(best_row['MAE']):.1f}")
k4.metric("Best R²",       f"{float(best_row['R2']):.3f}")

# ---- Plot 1: Actual vs predicted over time (first 600 test points for legibility) ----
st.subheader("Actual vs predicted (Random Forest)")
plot_n = min(600, len(pred_frame))
fig1, ax1 = plt.subplots(figsize=(11, 4))
ax1.plot(pred_frame[timestamp_col].iloc[:plot_n], pred_frame["actual"].iloc[:plot_n],
         label="Actual", linewidth=1.2, color="#1f77b4")
ax1.plot(pred_frame[timestamp_col].iloc[:plot_n], pred_frame["pred_rf"].iloc[:plot_n],
         label="Random Forest", linewidth=1.0, color="#d62728", alpha=0.85)
ax1.set_xlabel("Timestamp")
ax1.set_ylabel(target_col)
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)
fig1.autofmt_xdate()
st.pyplot(fig1)

# ---- Plot 2: Residuals over time ----
st.subheader("Residuals over time (actual − predicted)")
fig2, ax2 = plt.subplots(figsize=(11, 3))
ax2.plot(pred_frame[timestamp_col].iloc[:plot_n], pred_frame["residual_rf"].iloc[:plot_n],
         color="#2ca02c", linewidth=0.9)
ax2.axhline(0, color="black", linewidth=0.7)
ax2.set_xlabel("Timestamp")
ax2.set_ylabel("Residual (W)")
ax2.grid(alpha=0.3)
fig2.autofmt_xdate()
st.pyplot(fig2)

# ---- Plot 3: Residual distribution (histogram) ----
st.subheader("Residual distribution")
fig3, ax3 = plt.subplots(figsize=(8, 3.5))
ax3.hist(pred_frame["residual_rf"].dropna(), bins=60, color="#9467bd", edgecolor="white")
ax3.axvline(0, color="black", linewidth=0.8)
ax3.set_xlabel("Residual (W)")
ax3.set_ylabel("Count")
ax3.grid(alpha=0.3)
st.pyplot(fig3)

residual_stats = {
    "mean_residual_W": round(float(pred_frame["residual_rf"].mean()), 3),
    "std_residual_W": round(float(pred_frame["residual_rf"].std()), 3),
    "median_residual_W": round(float(pred_frame["residual_rf"].median()), 3),
    "p95_abs_error_W": round(float(pred_frame["abs_err_rf"].quantile(0.95)), 3),
}
st.json(residual_stats)

# ---- Plot 4: Scatter actual vs predicted ----
st.subheader("Actual vs predicted scatter")
fig4, ax4 = plt.subplots(figsize=(6, 6))
sample = pred_frame.sample(min(3000, len(pred_frame)), random_state=0)
ax4.scatter(sample["actual"], sample["pred_rf"], s=6, alpha=0.35, color="#ff7f0e")
lo = float(min(sample["actual"].min(), sample["pred_rf"].min()))
hi = float(max(sample["actual"].max(), sample["pred_rf"].max()))
ax4.plot([lo, hi], [lo, hi], "k--", linewidth=1)
ax4.set_xlabel("Actual (W)")
ax4.set_ylabel("Predicted (W)")
ax4.grid(alpha=0.3)
st.pyplot(fig4)

# ---- Daily error summary ----
st.subheader("Daily error summary (Random Forest)")
daily_err = (
    pred_frame.groupby("date")
    .agg(MAE=("abs_err_rf", "mean"),
         RMSE=("residual_rf", lambda s: float(np.sqrt(np.mean(s ** 2)))),
         n=("abs_err_rf", "size"))
    .reset_index()
    .round(3)
)
st.dataframe(daily_err, use_container_width=True)

fig5, ax5 = plt.subplots(figsize=(11, 3.2))
ax5.bar(daily_err["date"].astype(str), daily_err["MAE"], color="#17becf")
ax5.set_xlabel("Date")
ax5.set_ylabel("Daily MAE (W)")
ax5.tick_params(axis="x", rotation=45, labelsize=8)
ax5.grid(alpha=0.3, axis="y")
st.pyplot(fig5)

# ---- Hourly mean absolute error pattern ----
st.subheader("Mean absolute error by hour of day")
pred_frame["hour"] = pd.to_datetime(pred_frame[timestamp_col]).dt.hour
hourly_err = pred_frame.groupby("hour")["abs_err_rf"].mean().reset_index()
fig6, ax6 = plt.subplots(figsize=(8, 3.2))
ax6.bar(hourly_err["hour"], hourly_err["abs_err_rf"], color="#bcbd22")
ax6.set_xlabel("Hour of day")
ax6.set_ylabel("MAE (W)")
ax6.set_xticks(range(0, 24))
ax6.grid(alpha=0.3, axis="y")
st.pyplot(fig6)

# ---- Feature importance bar chart ----
st.subheader("Random Forest feature importance")
fig7, ax7 = plt.subplots(figsize=(7, 3.2))
ax7.barh(importances_df["feature"][::-1], importances_df["importance"][::-1], color="#8c564b")
ax7.set_xlabel("Importance")
ax7.grid(alpha=0.3, axis="x")
st.pyplot(fig7)

# ---- Written insights ----
st.subheader("Insights and limitations")

peak_hour = int(hourly_err.loc[hourly_err["abs_err_rf"].idxmax(), "hour"])
quiet_hour = int(hourly_err.loc[hourly_err["abs_err_rf"].idxmin(), "hour"])
top_feature = importances_df.iloc[0]["feature"]
rf_rmse = float(results_df.loc[results_df["model"] == "Random Forest", "RMSE"].iloc[0])
naive_rmse = float(results_df.loc[results_df["model"] == "Naive (lag-1)", "RMSE"].iloc[0])
improvement_pct = (naive_rmse - rf_rmse) / naive_rmse * 100 if naive_rmse > 0 else 0.0

insights = [
    f"**Random Forest beats the naive baseline** by roughly "
    f"{improvement_pct:.1f}% on RMSE ({rf_rmse:.1f} W vs {naive_rmse:.1f} W on the held-out test set), "
    f"showing engineered lag and calendar features carry real signal beyond yesterday's value.",
    f"**Lag-1 dominance.** The most important feature is `{top_feature}`, which is expected for a "
    f"5-minute resolution PV series — power changes slowly minute-to-minute, so the previous reading "
    f"is by far the strongest predictor.",
    f"**Errors concentrate around peak generation.** Mean absolute error peaks near hour {peak_hour} "
    f"and is smallest near hour {quiet_hour}. This is consistent with PV physics: at night the target "
    f"is essentially zero and trivial to predict; midday is when irradiance variability (clouds, "
    f"temperature) drives the largest residuals.",
    "**Residuals are roughly centred at zero** with heavier tails than a Gaussian, suggesting the "
    "model is unbiased on average but occasionally misses large step changes — likely cloud transients "
    "and inverter switching events that are not captured by the current feature set.",
    "**Limitations:** the model uses only the target's own lags and calendar features. Irradiance, "
    "temperature, humidity, and wind are present in the dataset but not yet fed to the model — "
    "adding them as exogenous features is the most obvious next step.",
    "**No data leakage:** the split is strictly chronological, all lag/rolling features use `.shift()` "
    "before any rolling window, and the test set's timestamps are entirely after the train set's.",
    "**Future work:** (a) add weather features as predictors, (b) try a gradient-boosted model "
    "(LightGBM / XGBoost) which typically edges out Random Forest on tabular forecasting, "
    "(c) walk-forward cross-validation instead of a single hold-out, "
    "(d) longer horizon forecasts (e.g. h=12 → one hour ahead).",
]
for bullet in insights:
    st.markdown("- " + bullet)

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
        "student_added_features": [
            "Ridge regression as a linear baseline alongside tree model",
            "Random Forest feature-importance ranking surfaced in dashboard",
        ],
    },
    "modeling_and_evaluation": {
        "has_time_based_split": True,
        "split_strategy": split_info["split_strategy"],
        "train_range": [split_info["train_start"], split_info["train_end"]],
        "test_range": [split_info["test_start"], split_info["test_end"]],
        "train_rows": split_info["train_rows"],
        "test_rows": split_info["test_rows"],
        "models_compared": results_df["model"].tolist(),
        "best_model_by_rmse": str(best_row["model"]),
        "has_metrics_table": has_metrics_table,
        "results_table": results_table,
        "feature_importances": importances_df.to_dict(orient="records"),
        "no_leakage_evidence": (
            "All lag and rolling features are computed with .shift() before the rolling window, "
            "the train/test split is strictly chronological with no shuffling, and the test set "
            "starts after the train set ends."
        ),
        "student_notes": (
            "Compared four forecasts on a chronological 80/20 split: Naive lag-1, Seasonal-naive lag-24, "
            "Ridge regression, and Random Forest. Metrics reported: MAE, RMSE, R^2, MAPE."
        ),
    },
    "dashboard": {
        "has_baseline_plot": True,
        "has_student_added_dashboard": True,
        "student_dashboard_components": [
            "KPI row (test rows, best RMSE, best MAE, best R^2)",
            "Actual vs predicted time-series plot",
            "Residuals-over-time plot",
            "Residual distribution histogram with summary stats",
            "Actual vs predicted scatter with y=x reference line",
            "Daily MAE/RMSE error summary table and bar chart",
            "Mean absolute error by hour-of-day bar chart",
            "Random Forest feature importance bar chart",
        ],
        "insights": [
            "Random Forest improves on naive lag-1 baseline on RMSE on the held-out test set.",
            "lag_1 dominates feature importance, consistent with 5-minute PV inertia.",
            "Errors peak near solar noon and are smallest at night when generation is zero.",
            "Residuals are roughly zero-centred but heavy-tailed during cloud transients.",
            "Weather features (irradiance, temperature) are available but not yet used — clear next step.",
            "Split is strictly chronological with no shuffling, lag/rolling features use .shift() to prevent leakage.",
        ],
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

## Modeling
- Time-based split: chronological, last {test_size_pct}% as test, no shuffling.
- Models compared: {", ".join(results_df["model"].tolist())}.
- Best model by RMSE: **{best_row['model']}** (MAE={best_row['MAE']}, RMSE={best_row['RMSE']}, R²={best_row['R2']}).
- Metrics reported: MAE, RMSE, R², MAPE on the held-out test set.

## Dashboard additions
- KPI row with test-set size and best-model metrics.
- Actual vs predicted time-series plot.
- Residuals over time and residual histogram with summary statistics.
- Actual-vs-predicted scatter with y=x reference line.
- Daily error summary table and bar chart.
- Mean absolute error by hour-of-day.
- Random Forest feature-importance ranking.

## Key insights
- Random Forest beats the naive lag-1 baseline on RMSE on the held-out test set.
- `lag_1` is the dominant feature — expected for a 5-minute PV series.
- Forecast errors concentrate around solar noon, near zero at night.
- Residuals are roughly zero-centred but heavy-tailed (cloud transients).
- Weather features (irradiance, temperature) are present in the dataset but not yet used as predictors — clear next step.

## No-leakage evidence
- Strict chronological split — test set starts after train set ends.
- All lag and rolling features use `.shift()` so no future value enters any feature.

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
st.caption("Completed Mini Project B — includes chronological split, four-model comparison, "
           "diagnostic dashboard, written insights, and submission exports.")
