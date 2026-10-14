"""Customer Purchase Propensity dashboard.

Run:  streamlit run app.py
"""
import json

import joblib
import pandas as pd
import streamlit as st

import charts
import train
from data_utils import FEATURES, TARGET, load_clean, load_raw

st.set_page_config(page_title="Purchase Propensity", page_icon="📊",
                   layout="wide")


@st.cache_data
def get_data():
    return load_raw(), load_clean()


@st.cache_resource
def get_model():
    """Load the saved model, training it first if it does not exist yet."""
    if not (train.MODEL_PATH.exists() and train.METRICS_PATH.exists()):
        train.run()
    model = joblib.load(train.MODEL_PATH)
    metrics = json.loads(train.METRICS_PATH.read_text())
    return model, metrics


raw, df = get_data()
model, M = get_model()

st.sidebar.title("📊 PurchaseIQ")
st.sidebar.caption("Customer purchase propensity prediction")
page = st.sidebar.radio("Go to", ["Dashboard", "Dataset", "Analysis",
                                  "Model & Predict"])
st.sidebar.divider()
st.sidebar.caption("Machine Learning 25SC2107E")

# ------------------------------------------------------------------ Dashboard
if page == "Dashboard":
    st.title("Purchase propensity dashboard")
    st.caption("Which customers buy, and what sets them apart.")

    buy_rate = df[TARGET].mean() * 100
    med = df.groupby(TARGET)["TimeSpentOnWebsite"].median()
    k = st.columns(4)
    k[0].metric("Customers (after cleaning)", f"{len(df):,}",
                f"{M['duplicates_removed']} duplicates removed",
                delta_color="off")
    k[1].metric("Purchase rate", f"{buy_rate:.1f}%")
    k[2].metric("Extra minutes on site (buyers)",
                f"+{med[1] - med[0]:.0f}", "median", delta_color="off")
    k[3].metric("Model accuracy", f"{M['accuracy'] * 100:.1f}%")

    a, b = st.columns(2)
    a.pyplot(charts.class_distribution(df))
    b.pyplot(charts.time_box(df))
    c, d = st.columns(2)
    c.pyplot(charts.rate_by(df, "LoyaltyProgram"))
    d.pyplot(charts.rate_by(df, "ProductCategory"))

    st.subheader("Model performance")
    r = st.columns(5)
    for col, (label, key) in zip(r, [("Accuracy", "accuracy"),
                                     ("Precision", "precision"),
                                     ("Recall", "recall"), ("F1", "f1"),
                                     ("ROC-AUC", "roc_auc")]):
        col.metric(label, f"{M[key]:.3f}")

# -------------------------------------------------------------------- Dataset
elif page == "Dataset":
    st.title("Dataset")
    st.caption(f"{len(raw):,} raw rows, {raw.shape[1]} columns. "
               f"{len(df):,} rows after removing duplicates.")
    f1, f2 = st.columns([1, 2])
    outcome = f1.selectbox("Outcome", ["All", "Purchased", "Did not purchase"])
    search = f2.text_input("Search any value")
    view = df
    if outcome != "All":
        view = view[view[TARGET] == (1 if outcome == "Purchased" else 0)]
    if search:
        mask = view.astype(str).apply(
            lambda col: col.str.contains(search, case=False)).any(axis=1)
        view = view[mask]
    st.write(f"Showing {len(view):,} rows")
    st.dataframe(view)
    st.download_button("Download cleaned data (CSV)",
                       df.to_csv(index=False), "cleaned_data.csv")
    st.subheader("Summary statistics")
    st.dataframe(df.describe().T)

# ------------------------------------------------------------------- Analysis
elif page == "Analysis":
    st.title("Analysis")
    a, b, c = st.columns(3)
    a.metric("Missing values", int(raw.isnull().sum().sum()))
    b.metric("Duplicate rows removed", M["duplicates_removed"])
    c.metric("Rows used", f"{len(df):,}")
    col = st.selectbox("Feature", [f for f in FEATURES])
    left, right = st.columns(2)
    left.pyplot(charts.distribution(df, col))
    right.pyplot(charts.correlation(df))

# ------------------------------------------------------------ Model & Predict
else:
    st.title("Model & Predict")
    st.caption("Logistic regression with scaled numeric inputs and "
               "one-hot encoded product category.")
    r = st.columns(5)
    for col, (label, key) in zip(r, [("Accuracy", "accuracy"),
                                     ("Precision", "precision"),
                                     ("Recall", "recall"), ("F1", "f1"),
                                     ("ROC-AUC", "roc_auc")]):
        col.metric(label, f"{M[key]:.3f}")
    st.caption(f"Evaluated on {M['test_rows']} held-out customers "
               f"(trained on {M['train_rows']}).")

    a, b, c = st.columns(3)
    a.pyplot(charts.confusion(M["confusion_matrix"]))
    b.pyplot(charts.roc(M["roc_curve"], M["roc_auc"]))
    c.pyplot(charts.coefficients(M["coefficients"]))

    st.subheader("Predict a customer")
    with st.form("predict"):
        x = st.columns(4)
        age = x[0].slider("Age", int(df.Age.min()), int(df.Age.max()), 40)
        income = x[1].slider("Annual income", 0, int(df.AnnualIncome.max()),
                             80000, step=1000)
        buys = x[2].slider("Number of purchases", 0,
                           int(df.NumberOfPurchases.max()), 10)
        minutes = x[3].slider("Time on website", 0.0,
                              float(round(df.TimeSpentOnWebsite.max())), 30.0)
        y = st.columns(4)
        gender = y[0].selectbox("Gender (coded)", [0, 1])
        category = y[1].selectbox("Product category",
                                  sorted(df.ProductCategory.unique()))
        loyalty = y[2].selectbox("Loyalty program", [0, 1])
        discounts = y[3].slider("Discounts availed", 0,
                                int(df.DiscountsAvailed.max()), 2)
        go = st.form_submit_button("Predict")
    if go:
        row = pd.DataFrame([{
            "Age": age, "Gender": gender, "AnnualIncome": income,
            "NumberOfPurchases": buys, "TimeSpentOnWebsite": minutes,
            "LoyaltyProgram": loyalty, "DiscountsAvailed": discounts,
            "ProductCategory": category}])[FEATURES]
        p = float(model.predict_proba(row)[0, 1])
        st.metric("Probability of purchase", f"{p * 100:.1f}%")
        st.progress(p)
        st.success("Likely to purchase") if p >= 0.5 else st.warning(
            "Unlikely to purchase")
