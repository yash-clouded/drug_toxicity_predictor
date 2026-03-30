import os
import sys
import joblib
import json
import pandas as pd
import numpy as np
import traceback

# FORCE the current project's src to the front of sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_SRC = os.path.join(BASE_DIR, "src")
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from atom_shap import quick_heatmap_from_model
from feature_engineering import compute_all_features

def test_heatmap():
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    target = "NR-AR"
    safe = target.replace("-", "_")
    
    # Use absolute paths for assets too
    model_path = os.path.join(BASE_DIR, "models", f"{safe}_xgb.pkl")
    imputer_path = os.path.join(BASE_DIR, "models", f"{safe}_imputer.pkl")
    meta_path = os.path.join(BASE_DIR, "models", f"{safe}_meta.json")
    
    try:
        model = joblib.load(model_path)
        imputer = joblib.load(imputer_path)
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as e:
        print(f"Failed to load assets from {BASE_DIR}/models: {e}")
        return

    print(f"Testing heatmap for {target} using SRC at {PROJECT_SRC}...")
    try:
        svg = quick_heatmap_from_model(smiles, model, imputer, meta, compute_all_features)
        if svg:
            print("SUCCESS: SVG generated.")
        else:
            print("FAILED: No SVG generated (None returned).")
    except Exception as e:
        print(f"CRASH: {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    test_heatmap()
