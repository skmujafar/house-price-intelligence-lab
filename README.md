
# House Price Intelligence Lab — Advanced Portfolio Project

Dataset will be downloaded on first run (network needed).

## What is included
A Streamlit real-estate analytics application, ML pipeline, SHAP explainability (optional), data quality checks, model training and evaluation, drift monitoring, map visualization, forecasting demo, investment/rental-yield calculator, PDF report export, CSV/Excel export, SQL persistence (SQLite by default; MySQL configurable), FastAPI endpoint, tests, and deployment templates.

## Dataset: read this before presenting results
The project uses the California Housing dataset from scikit-learn. It has 20,640 census block-group observations derived from 1990 U.S. Census data. The target is historical median house value per block group, not current individual property listing or transaction prices. Values are in units of $100,000 in `MedHouseVal`; the extra `MedHouseVal_USD` column converts that target to USD. It is real-world observational data, but old and aggregated. Do not present it as current Indian property data or current sale prices.

Official documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

If `data/california_housing.csv` is not bundled, run:
```bash
python download_dataset.py
```
First download needs internet.

## Run locally (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_dataset.py
python -m streamlit run streamlit_app.py
```
Open the local URL printed by Streamlit.

## Run the API separately
```powershell
uvicorn api:app --reload
```
API docs: http://127.0.0.1:8000/docs

## Security and limitations
- This is a learning/portfolio starter, not a production real-estate valuation service.
- Authentication is a simple demo login pattern; do not deploy without hardening password storage, sessions, secrets, CSRF protections, rate limits, and database access.
- For production use, set `DATABASE_URL` in the environment, e.g. `mysql+pymysql://user:password@host:3306/house_prices`. Never commit secrets.
- The map shows dataset census-area coordinates, not precise property addresses.
- Forecasting requires time-stamped observations; the app labels its trend simulation accordingly and does not claim a validated forecast from a single cross-sectional dataset.
- Rental yield/ROI values are user-entered scenarios, not ground-truth market data.
- Drift monitoring compares current uploaded data with the training data distribution; it needs new data to be meaningful.
- SHAP is optional and requires installation; if unavailable, the app still works without SHAP.
=======
# house-price-intelligence-lab
House Price Intelligence Lab using Machine Learning and Streamlit
>>>>>>> 8dbae3967b0c233320e7710d2db722dd8a414826
