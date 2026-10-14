# Customer Purchase Propensity Prediction

Logistic regression model plus a Streamlit dashboard (Dashboard, Dataset,
Analysis, Model & Predict) built on `data/customer_purchase_data.xlsx`.

## Run in VS Code

1. Open this folder in VS Code (File > Open Folder) and open a terminal
   (Ctrl + `).
2. Create and activate a virtual environment:
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. Install packages: `pip install -r requirements.txt`
4. Train the model: `python train.py` (prints accuracy, precision, recall, F1, ROC-AUC)
5. Start the dashboard: `streamlit run app.py` (opens http://localhost:8501)

If VS Code asks for a Python interpreter, pick the one inside `.venv`.

## Files

| File | Purpose |
|---|---|
| `data_utils.py` | Load the Excel file, drop duplicates, fill gaps with the median |
| `train.py` | Split 80/20 (stratified), scale and one-hot encode, fit, evaluate, save to `models/` |
| `charts.py` | Matplotlib figures used by the dashboard |
| `app.py` | Streamlit dashboard, with a live "predict a customer" form |
