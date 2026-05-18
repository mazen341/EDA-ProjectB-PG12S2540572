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

## Important student work still required

This starter intentionally stops before model training and scoring. You must add:

- time-based train/test split
- at least one forecasting model
- predictions and metrics table assigned to `results_df`
- extra dashboard visuals and written insights
