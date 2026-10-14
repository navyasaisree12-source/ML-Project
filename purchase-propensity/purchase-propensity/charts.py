"""Matplotlib figures used by the dashboard."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from data_utils import TARGET

BUY, NO = "#0e8f8a", "#e0833a"
LABELS = {0: "Did not purchase", 1: "Purchased"}


def _fig(w=5.2, h=3.4):
    fig, ax = plt.subplots(figsize=(w, h))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


def class_distribution(df):
    fig, ax = _fig()
    counts = df[TARGET].value_counts().sort_index()
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values,
                  color=[NO, BUY])
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Customers")
    ax.set_title("Purchase status")
    return fig


def time_box(df):
    fig, ax = _fig()
    data = [df[df[TARGET] == k]["TimeSpentOnWebsite"] for k in (0, 1)]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, c in zip(bp["boxes"], (NO, BUY)):
        patch.set_facecolor(c)
    ax.set_xticklabels([LABELS[0], LABELS[1]])
    ax.set_ylabel("Time on website")
    ax.set_title("Time on website vs purchase")
    return fig


def rate_by(df, col, title=None):
    fig, ax = _fig()
    rate = df.groupby(col)[TARGET].mean() * 100
    bars = ax.bar(rate.index.astype(str), rate.values, color=BUY)
    ax.bar_label(bars, fmt="%.0f%%", padding=3)
    ax.set_ylabel("Purchase rate (%)")
    ax.set_xlabel(col)
    ax.set_title(title or f"Purchase rate by {col}")
    ax.set_ylim(0, max(100, rate.max() + 10))
    return fig


def distribution(df, col):
    fig, ax = _fig(6, 3.4)
    bins = min(25, max(df[col].nunique(), 5))
    for k, c in ((0, NO), (1, BUY)):
        ax.hist(df[df[TARGET] == k][col], bins=bins, alpha=0.65,
                color=c, label=LABELS[k])
    ax.set_xlabel(col)
    ax.set_ylabel("Customers")
    ax.legend(frameon=False)
    ax.set_title(f"{col} by purchase status")
    return fig


def correlation(df):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    corr = df.corr(numeric_only=True)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center",
                    va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    return fig


def confusion(cm):
    fig, ax = _fig(4, 3.4)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["No purchase", "Purchase"])
    ax.set_yticks([0, 1], ["No purchase", "Purchase"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    ax.set_title("Confusion matrix (test set)")
    return fig


def roc(curve, auc):
    fig, ax = _fig(4.2, 3.4)
    ax.plot(curve["fpr"], curve["tpr"], color=BUY, lw=2,
            label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("ROC curve")
    return fig


def coefficients(coefs):
    fig, ax = _fig(6, 3.6)
    items = sorted(coefs.items(), key=lambda kv: kv[1])
    names = [k.split("__", 1)[-1] for k, _ in items]
    vals = [v for _, v in items]
    ax.barh(names, vals, color=[BUY if v > 0 else NO for v in vals])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Coefficient (standardised inputs)")
    ax.set_title("What pushes the prediction up or down")
    fig.tight_layout()
    return fig
