# MMA Second Opinion: prototype (synthetic data)

## Run it (about 2 minutes)
1. Install Python 3.10 or newer.
2. Open a terminal in this folder and run:
   - `pip install -r requirements.txt`
   - `streamlit run app.py`
3. The dashboard opens in your browser.

Keep all four files together: `app.py`, `core.py`, `requirements.txt`, and the CSV.

## Files
- `core.py`: model (gradient boosting), three-zone triage, what-if explanation
- `app.py`: Streamlit dashboard (5 tabs)
- `mma_synthetic_newborn_screening.csv`: synthetic dataset (see DATA_DICTIONARY.md)

## Honest limits (put these in the pitch)
- All data is synthetic; ranges are illustrative and need checking against a published reference.
- Results show system behaviour, not clinical performance.
- Explanation is a "what-if" method, not SHAP (SHAP was not installable offline; you can add it later).
- Action wording is generic. Align it with the DHA Newborn Screening Guideline after reading it.
