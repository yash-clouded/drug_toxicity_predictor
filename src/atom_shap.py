"""
atom_shap.py
------------
Maps fingerprint-bit SHAP values back to individual atoms in a molecule
to produce a per-atom importance score and a colour heatmap SVG.

Method:
  1. Compute Morgan fingerprints for the molecule (record bit → atom mapping)
  2. Run a SHAP-explained model on the molecule
  3. For each active fingerprint bit with a SHAP value:
       → identify the central atom that 'owns' that bit
       → accumulate |SHAP| onto that atom
  4. Render the molecule SVG with atoms coloured by importance (blue→red)
"""

from typing import Optional
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Bit → Atom mapping
# ---------------------------------------------------------------------------

def get_bit_to_atom_map(smiles: str, fp_bits: int = 2048, radius: int = 2) -> dict:
    """
    Returns a dict mapping fingerprint bit index → list of atom indices
    (the central atoms that generated that bit).
    """
    if not RDKIT_AVAILABLE:
        return {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    bi = {}   # bit_info: maps bit → list of (atom_idx, radius)
    AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_bits, bitInfo=bi)
    return {bit: [entry[0] for entry in entries] for bit, entries in bi.items()}


# ---------------------------------------------------------------------------
# Atom importance from SHAP (fp bits)
# ---------------------------------------------------------------------------

def compute_atom_importance(
    smiles: str,
    shap_values: np.ndarray,
    feature_names: list[str],
    fp_bits: int = 2048,
    radius: int = 2,
) -> np.ndarray:
    """
    Given SHAP values for all features (FP bits + descriptors),
    aggregate |SHAP| of FP bits onto the atoms that generated those bits.

    Returns an array of shape (n_atoms,) with per-atom importance scores.
    """
    if not RDKIT_AVAILABLE:
        return np.array([])

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.array([])

    n_atoms = mol.GetNumAtoms()
    atom_scores = np.zeros(n_atoms)

    bit_to_atoms = get_bit_to_atom_map(smiles, fp_bits=fp_bits, radius=radius)

    for i, name in enumerate(feature_names):
        if not name.startswith("FP_"):
            continue
        try:
            bit_idx = int(name.split("_")[1])
        except (ValueError, IndexError):
            continue

        shap_val = abs(shap_values[i])
        atoms = bit_to_atoms.get(bit_idx, [])
        if atoms:
            contrib = shap_val / len(atoms)
            for atom in atoms:
                if atom < n_atoms:
                    atom_scores[atom] += contrib

    return atom_scores


# ---------------------------------------------------------------------------
# Colour mapping
# ---------------------------------------------------------------------------

def score_to_color(score: float, max_score: float, scheme: str = "RdBu"):
    """
    Map a normalised score to an RGB tuple.
    scheme 'RdBu': low = blue, high = red (good for toxicity: red = toxic)
    """
    if max_score == 0:
        t = 0.5
    else:
        t = np.clip(score / max_score, 0.0, 1.0)

    # Blue (0,0,0.9) → white (1,1,1) → Red (0.9,0,0)
    if t < 0.5:
        s = t * 2          # 0→1
        r = s
        g = s
        b = 0.9 + (1.0 - 0.9) * s   # 0.9→1.0 but inverted
        r = s * 1.0
        g = s * 1.0
        b = 0.9 + (0.1) * s          # blue fading to white
        return (r, g, b)
    else:
        s = (t - 0.5) * 2  # 0→1
        r = 0.9 + (1.0 - 0.9) * (1 - s)  # towards red
        g = 1.0 - s
        b = 1.0 - s
        r = 0.9 * s + 1.0 * (1 - s)
        g = 1.0 - s
        b = 1.0 - s
        return (r, g, b)


# ---------------------------------------------------------------------------
# SVG renderer with atom heatmap
# ---------------------------------------------------------------------------

def render_atom_heatmap(
    smiles: str,
    atom_scores: np.ndarray,
    width: int = 450,
    height: int = 350,
) -> Optional[str]:
    """
    Render a molecule SVG where each atom is coloured by its importance score.
    Hot = red (high SHAP contribution), cold = white/blue (low).

    Returns SVG string or None.
    """
    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(atom_scores) == 0:
        return None

    n_atoms = mol.GetNumAtoms()
    if len(atom_scores) != n_atoms:
        # Pad/trim to match
        padded = np.zeros(n_atoms)
        padded[:min(len(atom_scores), n_atoms)] = atom_scores[:n_atoms]
        atom_scores = padded

    max_score = atom_scores.max()

    atom_colors = {}
    for i in range(n_atoms):
        atom_colors[i] = score_to_color(atom_scores[i], max_score)

    all_atoms = list(range(n_atoms))
    all_bonds = []
    bond_colors = {}
    for bond in mol.GetBonds():
        ai, aj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        avg = (atom_scores[ai] + atom_scores[aj]) / 2
        bond_colors[bond.GetIdx()] = score_to_color(avg, max_score)
        all_bonds.append(bond.GetIdx())

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=all_atoms,
        highlightBonds=all_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# ---------------------------------------------------------------------------
# Convenience: quick heatmap from model + smiles (no SHAP pre-computed)
# ---------------------------------------------------------------------------

def quick_heatmap_from_model(
    smiles: str,
    model,
    imputer,
    meta: dict,
    compute_features_fn,
    fp_bits: int = 2048,
) -> Optional[str]:
    """
    End-to-end: given a SMILES and a trained model, produce a SHAP atom heatmap SVG.
    Returns SVG string or None on failure.
    """
    try:
        import shap
    except ImportError:
        return None

    try:
        # Prepare features exactly as the model saw them during training
        # We prefer metadata if available, else default to both
        use_fp = meta.get("use_fingerprints", True)
        use_desc = meta.get("use_descriptors", True)
        
        X, feat_names = compute_features_fn(
            __import__("pandas").Series([smiles]),
            use_fingerprints=use_fp,
            use_descriptors=use_desc,
            fp_bits=fp_bits,
            show_progress=False,
        )
        
        # Ensure X matches the expected feature count
        if X.shape[1] != len(meta.get("feature_names", feat_names)):
            # Force match if possible
            if "feature_names" in meta:
                # Re-compute if possible or fail gracefully
                pass
        
        X = imputer.transform(X)

        # Get base model for SHAP
        base_model = model
        if hasattr(model, "named_estimators_"):
            # Best way for scikit-learn VotingClassifier/Pipeline
            ne = model.named_estimators_
            base_model = ne.get("xgb", ne.get("xgboost", next(iter(ne.values()))))
        elif hasattr(model, "estimators_"): 
            try:
                # Some versions/wrappers have a list of objects
                first = model.estimators_[0]
                if isinstance(first, tuple):
                    base_model = next((est for nm, est in model.estimators_ if "xgb" in nm.lower()), model.estimators_[0][1])
                else:
                    base_model = first
            except Exception as e:
                print(f"[atom_shap] Ensemble extraction failed: {e}")
        elif hasattr(model, "best_estimator_"):
            base_model = model.best_estimator_

        # Use the underlying booster for XGBoost to be safe across SHAP versions
        # Try to get the raw booster to avoid 'XGBClassifier' subscripting errors in some SHAP versions
        try:
            booster = base_model.get_booster()
            explainer = shap.TreeExplainer(booster)
        except:
            explainer = shap.TreeExplainer(base_model)
            
        sv = explainer.shap_values(X)
        
        # Handle various SHAP output shapes reliably
        if isinstance(sv, list):
            # Binary classification list [neg_scores, pos_scores]
            sv_vec = sv[1][0] if len(sv) > 1 else sv[0][0]
        elif len(sv.shape) == 3:
            # (n_samples, n_features, n_classes)
            sv_vec = sv[0, :, 1]
        else:
            # (n_samples, n_features)
            sv_vec = sv[0]

        atom_scores = compute_atom_importance(smiles, sv_vec, feat_names, fp_bits=fp_bits)
        return render_atom_heatmap(smiles, atom_scores)
    except Exception as e:
        import traceback
        err_msg = f"[atom_shap] Heatmap failed: {e}\n{traceback.format_exc()}"
        print(err_msg)
        return None
