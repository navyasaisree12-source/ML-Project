"""Loading and cleaning helpers shared by train.py and app.py."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "customer_purchase_data.xlsx"
TARGET = "PurchaseStatus"
CATEGORICAL = ["ProductCategory"]
NUMERIC = ["Age", "Gender", "AnnualIncome", "NumberOfPurchases",
           "TimeSpentOnWebsite", "LoyaltyProgram", "DiscountsAvailed"]
FEATURES = NUMERIC + CATEGORICAL


def load_raw() -> pd.DataFrame:
    return pd.read_excel(DATA_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and fill any numeric gaps with the median."""
    df = df.drop_duplicates().reset_index(drop=True)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


def load_clean() -> pd.DataFrame:
    return clean(load_raw())
