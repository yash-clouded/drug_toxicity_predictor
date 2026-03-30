"""
evaluate.py
-----------
Comprehensive evaluation of trained Tox21 toxicity models.

Produces:
  - ROC curves (per-target and averaged)
  - Precision-Recall curves
  - Confusion matrices
  - Calibration plots
  - Summary metrics table (CSV)

Usage:
    python src/evaluate.py
    python src/evaluate.py --target SR-MMP
"""

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    f1_score,
)
from sklearn.calibration import calibration_curve

from data_processing import load_tox21, clean_dataset, get_binary_target, split_data, TOX21_TARGETS
from feature_engineering import compute_all_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "tox21.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#264653", "#A8DADC", "#6A4C93", "#52B788", "#FFB703",
    "#FB8500", "#023047",
]
sns.set_theme(style="whitegrid", font_scale=1.1)


def load_model_and_meta(target: str, model_type: str = "xgb"):
    safe_target = target.replace("-", "_")
    model_path = os.path.join(MODEL_DIR, f"{safe_target}_{model_type}.pkl")
    imputer_path = os.path.join(MODEL_DIR, f"{safe_target}_imputer.pkl")
    meta_path = os.path.join(MODEL_DIR, f"{safe_target}_meta.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\nRun src/train.py first."
        )

    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, imputer, meta


def get_test_predictions(df, target, model, imputer, meta):
    smiles, labels = get_binary_target(df, target)
    splits = split_data(smiles, labels)
    smi_test, y_test = splits["test"]

    X_test, _ = compute_all_features(
        smi_test,
        use_fingerprints=meta["use_fingerprints"],
        use_descriptors=meta["use_descriptors"],
        fp_bits=meta["fp_bits"],
        show_progress=False,
    )
    X_test = imputer.transform(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return y_test.values, proba, pred


# ---------------------------------------------------------------------------
# Individual target plots
# ---------------------------------------------------------------------------

def plot_roc_pr_cm(y_true, proba, pred, target, save_dir):
    """Generate ROC, PR, and Confusion Matrix plots for one target."""
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f"Evaluation: {target}", fontsize=14, fontweight="bold", y=1.02)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    roc_auc = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)

    # --- ROC Curve ---
    ax1 = fig.add_subplot(gs[0])
    fpr, tpr, _ = roc_curve(y_true, proba)
    ax1.plot(fpr, tpr, color=PALETTE[0], lw=2, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])

    # --- PR Curve ---
    ax2 = fig.add_subplot(gs[1])
    prec, rec, _ = precision_recall_curve(y_true, proba)
    ax2.plot(rec, prec, color=PALETTE[1], lw=2, label=f"AP = {pr_auc:.3f}")
    baseline = y_true.mean()
    ax2.axhline(y=baseline, color="k", linestyle="--", lw=1, label=f"Baseline = {baseline:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="upper right")

    # --- Confusion Matrix ---
    ax3 = fig.add_subplot(gs[2])
    cm = confusion_matrix(y_true, pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
        xticklabels=["Non-toxic", "Toxic"],
        yticklabels=["Non-toxic", "Toxic"],
    )
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    ax3.set_title("Confusion Matrix")

    safe_target = target.replace("-", "_")
    save_path = os.path.join(save_dir, f"{safe_target}_eval.png")
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Saved: {save_path}")
    return roc_auc, pr_auc


# ---------------------------------------------------------------------------
# Multi-target summary plot
# ---------------------------------------------------------------------------

def plot_all_roc_curves(results: list, save_dir: str):
    """Overlay ROC curves for all targets on a single plot."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="_nolegend_")

    for i, r in enumerate(results):
        fpr, tpr, _ = roc_curve(r["y_true"], r["proba"])
        ax.plot(
            fpr, tpr,
            color=PALETTE[i % len(PALETTE)],
            lw=1.8,
            alpha=0.85,
            label=f"{r['target']} (AUC={r['roc_auc']:.3f})",
        )

    mean_auc = np.mean([r["roc_auc"] for r in results])
    ax.set_title(f"ROC Curves — All Tox21 Targets  (Mean AUC = {mean_auc:.3f})", fontsize=13)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    save_path = os.path.join(save_dir, "all_roc_curves.png")
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Saved: {save_path}")


def plot_auc_bar(results: list, save_dir: str):
    """Horizontal bar chart of test ROC-AUC per target."""
    df = pd.DataFrame(results).sort_values("roc_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(
        df["target"], df["roc_auc"],
        color=[PALETTE[i % len(PALETTE)] for i in range(len(df))],
        edgecolor="white",
    )
    ax.axvline(x=0.5, color="red", linestyle="--", lw=1.2, label="Random (0.50)")
    ax.axvline(x=df["roc_auc"].mean(), color="black", linestyle=":", lw=1.5,
               label=f"Mean ({df['roc_auc'].mean():.3f})")
    ax.set_xlabel("Test ROC-AUC")
    ax.set_title("Model Performance per Tox21 Target")
    ax.set_xlim([0.4, 1.0])
    for bar, val in zip(bars, df["roc_auc"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.legend()
    save_path = os.path.join(save_dir, "auc_barplot.png")
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_all(df, targets, model_type="xgb"):
    results = []
    metrics_rows = []

    for target in targets:
        if target not in df.columns:
            continue
        try:
            model, imputer, meta = load_model_and_meta(target, model_type)
        except FileNotFoundError as e:
            print(f"[evaluate] Skipping {target}: {e}")
            continue

        y_true, proba, pred = get_test_predictions(df, target, model, imputer, meta)

        roc_auc, pr_auc = plot_roc_pr_cm(y_true, proba, pred, target, RESULTS_DIR)
        f1 = f1_score(y_true, pred, zero_division=0)

        results.append({
            "target": target,
            "y_true": y_true,
            "proba": proba,
            "roc_auc": roc_auc,
        })
        metrics_rows.append({
            "target": target,
            "test_roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "f1": round(f1, 4),
            "n_test": len(y_true),
            "pos_rate": round(y_true.mean(), 4),
        })
        print(f"  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  F1={f1:.4f}")

    if results:
        plot_all_roc_curves(results, RESULTS_DIR)
        plot_auc_bar(results, RESULTS_DIR)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[evaluate] Metrics saved to {metrics_path}")
    print(metrics_df.to_string(index=False))
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--target", default=None)
    parser.add_argument("--model", default="xgb", choices=["xgb", "rf"])
    args = parser.parse_args()

    df = load_tox21(args.data)
    df = clean_dataset(df)

    targets = [args.target] if args.target else TOX21_TARGETS
    evaluate_all(df, targets, model_type=args.model)
