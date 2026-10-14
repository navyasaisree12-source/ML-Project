"""Train the logistic regression model and save the model plus its metrics.

Run:  python train.py
"""
import json
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_utils import (CATEGORICAL, FEATURES, NUMERIC, ROOT, TARGET,
                        load_clean, load_raw)

MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"


def run() -> dict:
    raw = load_raw()
    df = load_clean()
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = Pipeline([("prep", pre),
                      ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]

    metrics = {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(df)),
        "duplicates_removed": int(len(raw) - len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "coefficients": dict(zip(names.tolist(), coefs.tolist())),
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    m = run()
    print(f"Rows: {m['raw_rows']} raw -> {m['clean_rows']} after cleaning")
    print(f"Train/Test: {m['train_rows']} / {m['test_rows']}")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{k:>10}: {m[k]:.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:", m["confusion_matrix"])
    print("Saved model to", MODEL_PATH)
