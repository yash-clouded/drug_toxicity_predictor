"""
feature_engineering.py
-----------------------
Converts raw SMILES strings into numerical feature vectors suitable for ML.

Two feature types are computed:
  1. Morgan Fingerprints (circular fingerprints, radius=2, 2048 bits)
     – Encode local structural environments of atoms.
  2. RDKit Molecular Descriptors (200 physicochemical properties)
     – Encodes global molecular properties: MW, LogP, TPSA, rings, etc.

Both feature sets can be used independently or concatenated.
"""

import numpy as np
import pandas as pd
from typing import Optional
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    from rdkit.ML.Descriptors import MoleculeDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print(
        "[WARNING] RDKit not installed. Feature engineering will not work.\n"
        "Install with: pip install rdkit-pypi  OR  conda install -c conda-forge rdkit"
    )


# ---------------------------------------------------------------------------
# Fingerprint Features
# ---------------------------------------------------------------------------

def smiles_to_morgan_fp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """
    Convert a SMILES string to a Morgan (ECFP) fingerprint bit vector.

    Args:
        smiles: SMILES string.
        radius: Morgan radius (2 ≈ ECFP4).
        n_bits: Length of the bit vector.

    Returns:
        numpy array of shape (n_bits,) or None if the molecule is invalid.
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required for fingerprint generation.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.uint8)


def compute_morgan_fingerprints(
    smiles_series: pd.Series,
    radius: int = 2,
    n_bits: int = 2048,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batch-compute Morgan fingerprints for a Series of SMILES.

    Returns:
        features: ndarray of shape (n_valid, n_bits)
        valid_mask: boolean array indicating which SMILES were parseable
    """
    results = []
    valid_mask = []

    iterator = tqdm(smiles_series, desc="Morgan FP", disable=not show_progress)
    for smi in iterator:
        fp = smiles_to_morgan_fp(smi, radius=radius, n_bits=n_bits)
        if fp is not None:
            results.append(fp)
            valid_mask.append(True)
        else:
            results.append(np.zeros(n_bits, dtype=np.uint8))
            valid_mask.append(False)

    features = np.vstack(results)
    valid_mask = np.array(valid_mask, dtype=bool)
    n_invalid = (~valid_mask).sum()
    if n_invalid:
        print(f"[compute_morgan_fingerprints] {n_invalid} invalid SMILES zeroed out.")
    return features, valid_mask


# ---------------------------------------------------------------------------
# Molecular Descriptor Features
# ---------------------------------------------------------------------------

# Automatically collect all available RDKit descriptors
def _get_all_descriptors():
    if not RDKIT_AVAILABLE:
        return []
    return [d[0] for d in Descriptors.descList]

SELECTED_DESCRIPTORS = _get_all_descriptors()


def smiles_to_descriptors(smiles: str) -> Optional[dict]:
    """
    Compute a dictionary of RDKit molecular descriptors for a single SMILES.

    Returns None if SMILES is invalid.
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required for descriptor computation.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    desc_dict = {}
    for name in SELECTED_DESCRIPTORS:
        try:
            fn = getattr(Descriptors, name, None)
            if fn is None:
                # Try rdMolDescriptors
                fn = getattr(rdMolDescriptors, name, None)
            value = fn(mol) if fn else np.nan
            # Replace RDKit infinity / None with NaN
            if value is None or (isinstance(value, float) and np.isinf(value)):
                value = np.nan
            desc_dict[name] = float(value)
        except Exception:
            desc_dict[name] = np.nan

    return desc_dict


def compute_molecular_descriptors(
    smiles_series: pd.Series,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Batch-compute molecular descriptors for a Series of SMILES.

    Returns:
        desc_df: DataFrame of shape (n_compounds, n_descriptors) with NaN for invalid
        valid_mask: boolean array
    """
    rows = []
    valid_mask = []

    iterator = tqdm(smiles_series, desc="Descriptors", disable=not show_progress)
    for smi in iterator:
        desc = smiles_to_descriptors(smi)
        if desc is not None:
            rows.append(desc)
            valid_mask.append(True)
        else:
            rows.append({k: np.nan for k in SELECTED_DESCRIPTORS})
            valid_mask.append(False)

    desc_df = pd.DataFrame(rows, columns=SELECTED_DESCRIPTORS)
    valid_mask = np.array(valid_mask, dtype=bool)
    return desc_df, valid_mask


# ---------------------------------------------------------------------------
# Combined Features
# ---------------------------------------------------------------------------

def compute_all_features(
    smiles_series: pd.Series,
    use_fingerprints: bool = True,
    use_descriptors: bool = True,
    fp_radius: int = 2,
    fp_bits: int = 2048,
    show_progress: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute and concatenate Morgan fingerprints and/or molecular descriptors.

    Returns:
        X: feature matrix of shape (n_compounds, n_features)
        feature_names: list of feature name strings
    """
    parts = []
    names = []

    if use_fingerprints:
        fp_matrix, _ = compute_morgan_fingerprints(
            smiles_series, radius=fp_radius, n_bits=fp_bits, show_progress=show_progress
        )
        parts.append(fp_matrix.astype(np.float32))
        names += [f"FP_{i}" for i in range(fp_bits)]

    if use_descriptors:
        desc_df, _ = compute_molecular_descriptors(smiles_series, show_progress=show_progress)
        # Impute NaN with column median
        desc_df = desc_df.fillna(desc_df.median())
        parts.append(desc_df.values.astype(np.float32))
        names += list(desc_df.columns)

    if not parts:
        raise ValueError("At least one of use_fingerprints or use_descriptors must be True.")

    X = np.hstack(parts)
    # CLIP to safe float32 range to avoid 'value too large' errors in XGBoost/RF
    f32_max = np.finfo(np.float32).max
    X = np.clip(X, -f32_max, f32_max)
    # Replace any remaining NaNs with 0 (though imputer should handle them)
    X = np.nan_to_num(X, nan=0.0, posinf=f32_max, neginf=-f32_max)

    print(f"[compute_all_features] Feature matrix shape: {X.shape} (clipping applied)")
    return X.astype(np.float32), names



# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_smiles = pd.Series([
        "CC(=O)Oc1ccccc1C(=O)O",   # Aspirin
        "c1ccc2cc3ccccc3cc2c1",     # Pyrene (PAH, known toxicant)
        "CCO",                       # Ethanol
        "INVALID_SMILES",            # Should be handled gracefully
    ])

    X, feat_names = compute_all_features(
        test_smiles,
        use_fingerprints=True,
        use_descriptors=True,
        fp_bits=512,
    )
    print(f"Feature matrix: {X.shape}")
    print(f"First descriptor features: {feat_names[512:520]}")
    print(f"Sample row (descriptors only): {X[0, 512:520]}")
