"""
explain.py
----------
SHAP-based model interpretability for Tox21 toxicity predictors.

Identifies which molecular features (descriptors / fingerprint bits) most
strongly drive toxicity predictions.

Outputs (saved to results/):
  - SHAP summary bar plot (global feature importance)
  - SHAP beeswarm plot (direction of feature effects)
  - SHAP dependency plot for top features
  - CSV of top feature importances

Usage:
    python src/explain.py
    python src/explain.py --target SR-MMP --n-samples 500
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

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARNING] SHAP not installed. Run: pip install shap")

from data_processing import load_tox21, clean_dataset, get_binary_target, split_data, TOX21_TARGETS
from feature_engineering import compute_all_features, SELECTED_DESCRIPTORS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "tox21.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_model_and_meta(target: str, model_type: str = "xgb"):
    safe_target = target.replace("-", "_")
    model = joblib.load(os.path.join(MODEL_DIR, f"{safe_target}_{model_type}.pkl"))
    imputer = joblib.load(os.path.join(MODEL_DIR, f"{safe_target}_imputer.pkl"))
    with open(os.path.join(MODEL_DIR, f"{safe_target}_meta.json")) as f:
        meta = json.load(f)
    return model, imputer, meta


def get_descriptor_subset(X: np.ndarray, feature_names: list) -> tuple[np.ndarray, list]:
    """
    Extract only the molecular descriptor columns (skip FP_ columns).
    This makes SHAP plots more interpretable.
    """
    desc_indices = [i for i, n in enumerate(feature_names) if not n.startswith("FP_")]
    desc_names = [feature_names[i] for i in desc_indices]
    return X[:, desc_indices], desc_names


def explain_target(
    df: pd.DataFrame,
    target: str,
    model_type: str = "xgb",
    n_samples: int = 500,
    top_k: int = 20,
):
    """Run SHAP analysis for a single target."""
    if not SHAP_AVAILABLE:
        raise RuntimeError("Install SHAP: pip install shap")

    print(f"\n[explain] Target: {target}")
    model, imputer, meta = load_model_and_meta(target, model_type)

    # If it's an ensemble, take the XGBoost part for faster/better SHAP
    if hasattr(model, 'estimators_'):
        base_model = next((est for name, est in model.estimators_ if name == 'xgb'), model.estimators_[0])
    else:
        base_model = model

    smiles, labels = get_binary_target(df, target)
    splits = split_data(smiles, labels)
    smi_test, y_test = splits["test"]

    X_test, feat_names = compute_all_features(
        smi_test,
        use_fingerprints=meta["use_fingerprints"],
        use_descriptors=meta["use_descriptors"],
        fp_bits=meta["fp_bits"],
        show_progress=False,
    )
    X_test = imputer.transform(X_test)

    n_samples = min(n_samples, len(X_test))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[idx]

    X_desc, desc_names = get_descriptor_subset(X_sample, feat_names)
    desc_indices = [i for i, n in enumerate(feat_names) if not n.startswith("FP_")]

    try:
        explainer = shap.TreeExplainer(base_model)
        shap_values_full = explainer.shap_values(X_sample)
        
        # XGBoost 2.0+ returns a 2D array [samples, features] for binary 
        # but sometimes a 3D [samples, features, 2]. 
        # Older versions return a list.
        if isinstance(shap_values_full, list):
            shap_vals = shap_values_full[1]
        elif len(shap_values_full.shape) == 3:
            shap_vals = shap_values_full[:, :, 1]
        else:
            shap_vals = shap_values_full
            
        shap_desc = shap_vals[:, desc_indices]
        print(f"[explain] SHAP values computed via TreeExplainer. Shape: {shap_desc.shape}")
    except Exception as e:
        print(f"[explain] TreeExplainer failed ({e}), using KernelExplainer (slower)...")
        background = shap.kmeans(X_desc, 20)
        def predict_fn(X):
            full = np.zeros((X.shape[0], X_sample.shape[1]), dtype=np.float32)
            full[:, desc_indices] = X
            return model.predict_proba(full)[:, 1]
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_desc = explainer.shap_values(X_desc, nsamples=100)

    # 1. Global Bar Plot
    mean_abs_shap = np.abs(shap_desc).mean(axis=0)
    feat_imp_df = pd.DataFrame({"feature": desc_names, "mean_abs_shap": mean_abs_shap}).sort_values("mean_abs_shap", ascending=False)
    safe_target = target.replace("-", "_")

    imp_path = os.path.join(RESULTS_DIR, f"{safe_target}_feature_importance.csv")
    feat_imp_df.head(top_k).to_csv(imp_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    top = feat_imp_df.head(top_k)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#457b9d")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Toxicity Drivers: {target}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{safe_target}_shap_bar.png"), dpi=120)
    plt.close(fig)

    # 2. Beeswarm
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_desc, X_desc, feature_names=desc_names, max_display=top_k, show=False)
    plt.title(f"Molecular Toxicity Profile: {target}", fontsize=14)
    plt.savefig(os.path.join(RESULTS_DIR, f"{safe_target}_shap_beeswarm.png"), dpi=120, bbox_inches="tight")
    plt.close()

    return feat_imp_df

    # ----------------------------------------------------------------
    # 3. Dependency plots for top 3 features
    # ----------------------------------------------------------------
    top3 = feat_imp_df.head(3)["feature"].tolist()
    for feat in top3:
        try:
            fidx = desc_names.index(feat)
            fig, ax = plt.subplots(figsize=(7, 5))
            shap.dependence_plot(
                fidx,
                shap_desc,
                X_desc,
                feature_names=desc_names,
                ax=ax,
                show=False,
            )
            ax.set_title(f"SHAP Dependence: {feat} — {target}")
            dep_path = os.path.join(
                RESULTS_DIR, f"{safe_target}_dep_{feat.replace('/', '_')}.png"
            )
            fig.savefig(dep_path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(f"[explain] Dependence plot saved: {dep_path}")
        except Exception as e:
            print(f"[explain] Could not plot dependence for {feat}: {e}")

    return feat_imp_df


# ---------------------------------------------------------------------------
# Global descriptor importance across all targets
# ---------------------------------------------------------------------------

def global_descriptor_importance(results: dict, save_dir: str, top_k: int = 20):
    """
    Aggregate per-target SHAP importances and plot a heatmap.

    Args:
        results: dict of {target: feat_imp_df}
    """
    if not results:
        return

    # Align on shared descriptors
    all_features = SELECTED_DESCRIPTORS
    combined = pd.DataFrame(index=all_features)

    for target, df_imp in results.items():
        s = df_imp.set_index("feature")["mean_abs_shap"]
        combined[target] = s.reindex(all_features, fill_value=0)

    # Compute overall importance (mean across targets)
    combined["overall"] = combined.mean(axis=1)
    combined = combined.sort_values("overall", ascending=False)

    # Heatmap of top-K
    top_feats = combined.head(top_k).drop(columns="overall")
    fig, ax = plt.subplots(figsize=(max(10, len(top_feats.columns)), 8))
    sns_data = top_feats.T
    import seaborn as sns
    sns.heatmap(
        sns_data,
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.3,
        linecolor="white",
        annot=False,
    )
    ax.set_title(f"SHAP Importance Heatmap — Top {top_k} Descriptors vs All Targets", fontsize=12)
    ax.set_ylabel("Toxicity Target")
    ax.set_xlabel("Molecular Descriptor")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, "shap_heatmap_all_targets.png")
    fig.savefig(heatmap_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[explain] Heatmap saved: {heatmap_path}")

    overall_path = os.path.join(save_dir, "global_feature_importance.csv")
    combined.to_csv(overall_path)
    print(f"[explain] Global importance CSV saved: {overall_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--target", default=None)
    parser.add_argument("--model", default="xgb", choices=["xgb", "rf"])
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    from data_processing import load_tox21, clean_dataset
    df = load_tox21(args.data)
    df = clean_dataset(df)

    targets = [args.target] if args.target else TOX21_TARGETS
    results = {}
    for target in targets:
        if target not in df.columns:
            continue
        try:
            feat_imp_df = explain_target(
                df, target,
                model_type=args.model,
                n_samples=args.n_samples,
                top_k=args.top_k,
            )
            results[target] = feat_imp_df
        except Exception as e:
            print(f"[explain] ERROR on {target}: {e}")

    if len(results) > 1:
        global_descriptor_importance(results, RESULTS_DIR, top_k=args.top_k)
