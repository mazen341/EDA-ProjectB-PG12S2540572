# EDA Project B — Time-Series Forecasting Starter

Student: MAZEN AL-HIMALI  
Student ID: PG12S2540572

This repository contains a starter Streamlit app for Mini Project B. The app loads a cleaned sample of the HKUST SQ1 PV dataset, audits the data, prepares baseline time-series features, exports project evidence files, and includes the fixed AI grader prompt.

## Files

- `app.py` — one-file Streamlit app
- `requirements.txt` — Python dependencies
- `data/dataset_sample.csv` — cleaned dataset slice, limited to at most 250,000 rows

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud deployment

1. Create a public GitHub repository named `EDA-ProjectB-PG12S2540572`.
2. Upload exactly these files:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `data/dataset_sample.csv`
3. Go to Streamlit Community Cloud.
4. Choose **New app**.
5. Connect your GitHub repository.
6. Select branch `main`.
7. Set main file path to `app.py`.
8. Deploy.

## OpenRouter API key for AI grader

Do not hardcode your API key. Use one of these options:

1. Streamlit Secrets: add `OPENROUTER_API_KEY = "your-key"` in app secrets.
2. Environment variable: set `OPENROUTER_API_KEY`.
3. Paste the key into the password field inside the app.

## What to submit

Submit these items to your instructor:

- Streamlit deployed app URL
- GitHub repository URL
- Exported `submission.json` from the app
- Exported `project_card.md` from the app
- Required screenshots:
  - first 10 rows preview
  - metrics table after you add modeling
  - at least one dashboard plot

## Project status — workshop complete

The starter has been extended into a complete forecasting workshop. The
following are implemented in `app.py` and verified to run end-to-end:

- **Time-based train/test split** — strict chronological 80/20 split (no
  shuffling) in `run_models`, avoiding look-ahead leakage.
- **Multiple forecasting models** — a naive seasonal baseline plus
  scikit-learn linear (RidgeCV/LassoCV/ElasticNetCV), tree-ensemble
  (RandomForest/ExtraTrees/GradientBoosting/HistGradientBoosting), and
  distance (KNN/SVR) models, selectable by comparison group.
- **Predictions and metrics** — held-out MAE, RMSE, MAPE and R² in
  `comparison_df`, with per-row predictions and 90% prediction intervals in
  `predictions_df` (exported as `metrics.csv` and `predictions.csv`).
- **Dashboard visuals and written insights** — line/bar/radar/scatter/
  histogram/heatmap charts, an animated digital-twin and energy-flow view,
  a what-if simulator, an all-in-one comparison lab, and structured
  insights/conclusions/limitations.

### Generated submission deliverables

All four export artifacts are produced from the **📤 Export** tab:
`submission.json`, `project_card.md`, `predictions.csv`, and `metrics.csv`.
For the fullest evidence, run a comparison group from the sidebar (any option
except "Do not train yet") before exporting.
