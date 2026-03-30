"""
train.py
--------
Full training pipeline for the Tox21 drug toxicity predictor.

Strategy:
  - For each of the 12 Tox21 assay targets, train a dedicated XGBoost classifier.
  - Handles class imbalance via scale_pos_weight.
  - Saves models and feature matrices to disk for evaluation and SHAP analysis.

Usage:
    python src/train.py
    python src/train.py --target SR-MMP          # single target
    python src/train.py --fp-bits 1024 --no-desc # fingerprints only
"""

import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

try:
    import xgboost as xgb
    X_AVAILABLE = True
except ImportError:
    X_AVAILABLE = False
    print("[WARNING] XGBoost not installed. Falling back to GradientBoostingClassifier.")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from data_processing import load_tox21, clean_dataset, get_binary_target, split_data, TOX21_TARGETS
from feature_engineering import compute_all_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "tox21.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Model Builder & Tuner
# ---------------------------------------------------------------------------

def get_optimized_params(X_train, y_train, X_val, y_val, n_trials=20):
    """Use Optuna to find best XGBoost hyperparameters."""
    if not OPTUNA_AVAILABLE or not X_AVAILABLE:
        return {}

    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        }
        
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / max(pos_count, 1)
        
        model = xgb.XGBClassifier(
            **param,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def build_model(scale_pos_weight: float = 1.0, params: dict = None) -> object:
    """Build a single XGBoost classifier."""
    xgb_params = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'use_label_encoder': False,
        'eval_metric': "auc",
        'random_state': 42,
        'n_jobs': -1,
    }
    if params:
        xgb_params.update(params)

    if X_AVAILABLE:
        return xgb.XGBClassifier(**xgb_params, scale_pos_weight=scale_pos_weight)
    else:
        return GradientBoostingClassifier(random_state=42)


# ---------------------------------------------------------------------------
# Training for a single target
# ---------------------------------------------------------------------------

def train_single_target(
    df: pd.DataFrame,
    target: str,
    fp_bits: int = 2048,
    use_descriptors: bool = True,
    use_fingerprints: bool = True,
    tune: bool = False,
    n_trials: int = 20,
) -> dict:
    """Train and evaluate a model for one toxicity target."""
    print(f"\n{'='*60}\n  Target: {target}\n{'='*60}")

    smiles, labels = get_binary_target(df, target)
    splits = split_data(smiles, labels)
    smi_train, y_train = splits["train"]
    smi_val, y_val = splits["val"]
    smi_test, y_test = splits["test"]

    print("[train] Computing features...")
    X_train, feat_names = compute_all_features(smi_train, use_fingerprints, use_descriptors, fp_bits=fp_bits)
    X_val, _ = compute_all_features(smi_val, use_fingerprints, use_descriptors, fp_bits=fp_bits)
    X_test, _ = compute_all_features(smi_test, use_fingerprints, use_descriptors, fp_bits=fp_bits)

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    best_params = {}
    if tune:
        print(f"[train] Optimizing hyperparameters with {n_trials} trials...")
        best_params = get_optimized_params(X_train, y_train, X_val, y_val, n_trials=n_trials)
        print(f"[train] Best Params: {best_params}")

    neg_count = (y_train == 0).sum()
    pos_count = y_train.sum()
    scale_pos_weight = neg_count / max(pos_count, 1)

    model = build_model(scale_pos_weight=scale_pos_weight, params=best_params)
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    test_auc = roc_auc_score(y_test, test_proba)

    print(f"[train] Val ROC-AUC:  {val_auc:.4f} | Test ROC-AUC: {test_auc:.4f}")

    safe_target = target.replace("-", "_")
    joblib.dump(model, os.path.join(MODEL_DIR, f"{safe_target}_xgb.pkl"))
    joblib.dump(imputer, os.path.join(MODEL_DIR, f"{safe_target}_imputer.pkl"))

    meta = {
        "target": target, "model_type": "xgboost", "fp_bits": fp_bits,
        "n_features": X_train.shape[1], "feature_names": feat_names,
        "val_roc_auc": round(val_auc, 4), "test_roc_auc": round(test_auc, 4),
        "train_size": len(y_train), "pos_rate_train": round(float(y_train.mean()), 4),
    }
    with open(os.path.join(MODEL_DIR, f"{safe_target}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def train_all_targets(df: pd.DataFrame, targets: list, **kwargs) -> pd.DataFrame:
    all_metrics = []
    for target in targets:
        if target not in df.columns: continue
        try:
            metrics = train_single_target(df, target, **kwargs)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"[train_all] ERROR on '{target}': {e}")
    
    summary = pd.DataFrame(all_metrics)[["target", "val_roc_auc", "test_roc_auc", "train_size", "pos_rate_train"]]
    summary.to_csv(os.path.join(RESULTS_DIR, "training_summary.csv"), index=False)
    print(f"\n[train_all] Summary:\n{summary.to_string(index=False)}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--target", default=None)
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter optimization")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--quick", action="store_true", help="Run a quick smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = load_tox21(args.data)
    if args.quick:
        df = df.sample(min(1000, len(df)), random_state=42)
    df = clean_dataset(df)
    targets = [args.target] if args.target else TOX21_TARGETS
    train_all_targets(df, targets=targets, tune=args.tune, n_trials=args.trials)
