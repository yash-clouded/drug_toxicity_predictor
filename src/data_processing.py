"""
data_processing.py
------------------
Handles loading, cleaning, and splitting the Tox21 dataset.

Tox21 contains ~12,000 compounds with SMILES strings and binary labels
for 12 toxicological assay endpoints.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# The 12 toxicity targets in Tox21
TOX21_TARGETS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

# Biological descriptions for each target
TOX21_TARGET_INFO = {
    "NR-AR": {
        "full_name": "Androgen Receptor",
        "system": "Endocrine",
        "description": "The 'male' hormone receptor. Disruption can lead to reproductive toxicity and developmental issues.",
    },
    "NR-AR-LBD": {
        "full_name": "Androgen Receptor (Ligand Binding Domain)",
        "system": "Endocrine",
        "description": "Specifically measures how well a molecule binds into the androgen receptor's pocket.",
    },
    "NR-AhR": {
        "full_name": "Aryl Hydrocarbon Receptor",
        "system": "Metabolic Stress",
        "description": "Activated by environmental toxins (like dioxins). Involved in metabolism, immunity, and carcinogenicity.",
    },
    "NR-Aromatase": {
        "full_name": "Aromatase Enzyme",
        "system": "Endocrine",
        "description": "An enzyme that converts androgens to estrogens. Inhibition can disrupt pregnancy and hormonal balance.",
    },
    "NR-ER": {
        "full_name": "Estrogen Receptor",
        "system": "Endocrine",
        "description": "The 'female' hormone receptor. Agonists can mimic estrogen, potentially leading to breast cancer or reproductive dysfunction.",
    },
    "NR-ER-LBD": {
        "full_name": "Estrogen Receptor (Ligand Binding Domain)",
        "system": "Endocrine",
        "description": "Measures direct binding to the estrogen receptor-alpha protein.",
    },
    "NR-PPAR-gamma": {
        "full_name": "Peroxisome Proliferator-Activated Receptor gamma",
        "system": "Metabolism",
        "description": "Regulates fat storage and glucose metabolism. Important in obesity and diabetes research.",
    },
    "SR-ARE": {
        "full_name": "Antioxidant Response Element",
        "system": "Cellular Stress",
        "description": "A signal that the cell is experiencing oxidative stress. High activity indicates potential liver or lung toxicity.",
    },
    "SR-ATAD5": {
        "full_name": "ATAD5 (Genotoxicity)",
        "system": "DNA Damage",
        "description": "A sensitive marker for DNA damage and stalled replication. Correlates with mutagenicity (Ames test).",
    },
    "SR-HSE": {
        "full_name": "Heat Shock Element",
        "system": "Cellular Stress",
        "description": "Triggered when protein folding goes wrong. Indicates general cell stress or protein damage.",
    },
    "SR-MMP": {
        "full_name": "Mitochondrial Membrane Potential",
        "system": "Energy / Mitochondrial",
        "description": "Measures if the cell's 'batteries' are leaking or broken. Common indicator of organ toxicity.",
    },
    "SR-p53": {
        "full_name": "p53 Tumor Suppressor",
        "system": "DNA Damage / Cancer",
        "description": "The 'guardian of the genome'. Activation indicates serious DNA damage that may lead to cancer/mutation.",
    },
}



def load_tox21(filepath: str) -> pd.DataFrame:
    """
    Load the Tox21 dataset from CSV.

    Expected columns:
        - smiles: SMILES string for each compound
        - mol_id (optional): compound identifier
        - <target columns>: binary (0/1) toxicity labels, one per assay

    Args:
        filepath: Path to tox21.csv

    Returns:
        DataFrame with SMILES and toxicity label columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Tox21 dataset not found at '{filepath}'.\n"
            "Download from: https://www.kaggle.com/datasets/epicskills/tox21-dataset\n"
            "and place it at data/tox21.csv"
        )

    df = pd.read_csv(filepath)
    print(f"[load_tox21] Loaded {len(df)} rows, {df.shape[1]} columns.")

    # Normalise column names: lowercase + strip whitespace
    df.columns = df.columns.str.strip().str.lower()

    # Try to identify SMILES column (common names)
    smiles_candidates = ["smiles", "smi", "canonical_smiles", "structure"]
    smiles_col = next((c for c in smiles_candidates if c in df.columns), None)
    if smiles_col is None:
        raise ValueError(
            f"Cannot find SMILES column. Available columns: {list(df.columns)}"
        )
    if smiles_col != "smiles":
        df = df.rename(columns={smiles_col: "smiles"})

    # Keep only SMILES + recognised target columns that are present
    available_targets = [t for t in TOX21_TARGETS if t in df.columns]
    if not available_targets:
        # Some releases use lowercase target names
        lower_targets = {t.lower(): t for t in TOX21_TARGETS}
        for col in df.columns:
            if col in lower_targets:
                df = df.rename(columns={col: lower_targets[col]})
        available_targets = [t for t in TOX21_TARGETS if t in df.columns]

    if not available_targets:
        raise ValueError(
            "No Tox21 target columns found. "
            f"Expected one of {TOX21_TARGETS}. Got: {list(df.columns)}"
        )

    keep_cols = ["smiles"] + available_targets
    # Also keep mol_id if present
    if "mol_id" in df.columns:
        keep_cols = ["mol_id"] + keep_cols
    df = df[keep_cols].copy()

    print(f"[load_tox21] Targets found: {available_targets}")
    return df


def load_zinc(filepath: str) -> pd.DataFrame:
    """
    Load the ZINC250k dataset from CSV.
    Expected columns: smiles, logP, qed, SAS

    Args:
        filepath: Path to zinc250k.csv

    Returns:
        DataFrame with SMILES and property columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ZINC dataset not found at '{filepath}'.")

    df = pd.read_csv(filepath)
    print(f"[load_zinc] Loaded {len(df)} rows, {df.shape[1]} columns.")

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()

    # The ZINC CSV typically has smiles as the first column, sometimes in quotes
    # and properties as [logP, qed, SAS].
    # Let's ensure 'smiles' exists.
    if "smiles" not in df.columns:
        # If not, assume first column is SMILES
        df.rename(columns={df.columns[0]: "smiles"}, inplace=True)

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Tox21 DataFrame:
    - Drop rows with missing SMILES
    - Drop exact duplicate SMILES
    - Convert target columns to numeric (coerce errors → NaN)

    Args:
        df: Raw Tox21 DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    original_len = len(df)

    # Drop missing / empty SMILES
    df = df.dropna(subset=["smiles"])
    df = df[df["smiles"].str.strip() != ""]

    # Drop duplicate SMILES (keep first occurrence)
    df = df.drop_duplicates(subset=["smiles"], keep="first")

    # Convert target columns to numeric
    target_cols = [c for c in df.columns if c in TOX21_TARGETS]
    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(
        f"[clean_dataset] {original_len} → {len(df)} rows after cleaning "
        f"({original_len - len(df)} removed)."
    )
    return df.reset_index(drop=True)


def get_binary_target(df: pd.DataFrame, target: str) -> tuple[pd.Series, pd.Series]:
    """
    Return (smiles, labels) for a single target column, dropping NaN labels.

    Args:
        df: Cleaned Tox21 DataFrame.
        target: One of TOX21_TARGETS.

    Returns:
        (smiles Series, binary label Series)
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in DataFrame columns.")

    sub = df[["smiles", target]].dropna(subset=[target])
    labels = sub[target].astype(int)
    smiles = sub["smiles"]

    pos_rate = labels.mean() * 100
    print(
        f"[get_binary_target] '{target}': {len(labels)} compounds, "
        f"{pos_rate:.1f}% positive (toxic)."
    )
    return smiles.reset_index(drop=True), labels.reset_index(drop=True)


def split_data(
    smiles: pd.Series,
    labels: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """
    Stratified train / validation / test split.

    Returns a dict with keys: train, val, test — each a (smiles, labels) tuple.
    """
    # First split off test set
    smi_trainval, smi_test, y_trainval, y_test = train_test_split(
        smiles,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    # Then split train / val from remaining
    val_fraction = val_size / (1.0 - test_size)
    smi_train, smi_val, y_train, y_val = train_test_split(
        smi_trainval,
        y_trainval,
        test_size=val_fraction,
        stratify=y_trainval,
        random_state=random_state,
    )

    print(
        f"[split_data] Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}"
    )
    return {
        "train": (smi_train.reset_index(drop=True), y_train.reset_index(drop=True)),
        "val": (smi_val.reset_index(drop=True), y_val.reset_index(drop=True)),
        "test": (smi_test.reset_index(drop=True), y_test.reset_index(drop=True)),
    }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tox21.csv")
    df = load_tox21(DATA_PATH)
    df = clean_dataset(df)
    print(df.head())
    smiles, labels = get_binary_target(df, "SR-MMP")
    splits = split_data(smiles, labels)
    print("Train size:", len(splits["train"][0]))
