"""
toxicophores.py
---------------
Chemistry-first toxicity explanations using SMARTS structural alerts.

Provides a curated library of toxic substructures (toxicophores) with:
  - SMARTS pattern for substructure matching
  - Human-readable name
  - Mechanism of toxicity (plain English)
  - Risk level (high / medium / low)
  - References to known assay relevance

Usage:
    from toxicophores import screen_molecule, match_alerts
"""

from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    import io, base64
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Toxicophore Library
# ---------------------------------------------------------------------------

TOXICOPHORE_LIBRARY = [
    # --- Genotoxicity / DNA damage ---
    {
        "id": "nitro_aromatic",
        "name": "Nitro Aromatic",
        "smarts": "[c][N+](=O)[O-]",
        "risk": "High",
        "mechanism": "Bacterial nitroreductases metabolise nitroaromatics to reactive hydroxylamines and nitroso groups that form covalent DNA adducts → mutagenic (Ames test positive).",
        "tox21_targets": ["NR-AR", "SR-p53"],
    },
    {
        "id": "aromatic_amine",
        "name": "Aromatic Primary Amine",
        "smarts": "[c][NH2]",
        "risk": "High",
        "mechanism": "N-hydroxylation by CYP1A2 produces reactive nitrenium ions that alkylate DNA. Classic bladder carcinogen (e.g. 2-naphthylamine).",
        "tox21_targets": ["NR-AhR", "SR-p53"],
    },
    {
        "id": "epoxide",
        "name": "Epoxide",
        "smarts": "C1OC1",
        "risk": "High",
        "mechanism": "Highly electrophilic three-membered ring opens to covalently modify nucleophilic sites on DNA, proteins, and GSH — genotoxic, nephrotoxic.",
        "tox21_targets": ["SR-ARE", "SR-p53"],
    },
    {
        "id": "michael_acceptor",
        "name": "Michael Acceptor (α,β-unsaturated carbonyl)",
        "smarts": "[$([CX3]=[CX3][CX3]=[O]),$([CX3](=[O])[CX2]=[CX3])]",
        "risk": "High",
        "mechanism": "Reacts via 1,4-addition with cysteine thiols in glutathione and proteins, depleting antioxidant defence → oxidative stress, hepatotoxicity.",
        "tox21_targets": ["SR-ARE", "SR-MMP"],
    },
    {
        "id": "quinone",
        "name": "Quinone",
        "smarts": "[#6]1(=O)[#6]=[#6][#6](=O)[#6]=[#6]1",
        "risk": "High",
        "mechanism": "Undergoes redox cycling generating reactive oxygen species (ROS). Also a Michael acceptor. Major driver of mitochondrial toxicity.",
        "tox21_targets": ["SR-MMP", "SR-ARE"],
    },
    {
        "id": "alkyl_halide",
        "name": "Activated Alkyl Halide",
        "smarts": "[CH2X4][F,Cl,Br,I]",
        "risk": "High",
        "mechanism": "Strong SN2 electrophile that alkylates DNA (N7-guanine adducts). Mechanism of action of nitrogen mustard chemotherapy but also carcinogenic.",
        "tox21_targets": ["SR-p53"],
    },
    # --- Reactive / Non-specific ---
    {
        "id": "acyl_halide",
        "name": "Acyl Halide",
        "smarts": "[CX3](=[OX1])[F,Cl,Br,I]",
        "risk": "High",
        "mechanism": "Extremely reactive acylating agent — rapidly reacts with water, amines, and thiols non-selectively. Major skin/respiratory sensitiser.",
        "tox21_targets": ["SR-ARE"],
    },
    {
        "id": "isocyanate",
        "name": "Isocyanate",
        "smarts": "[NX2]=[CX2]=[OX1]",
        "risk": "High",
        "mechanism": "Highly reactive towards nucleophiles; forms stable carbamate or urea adducts with proteins and DNA. Renowned respiratory sensitiser (e.g. TDI).",
        "tox21_targets": ["SR-ARE"],
    },
    {
        "id": "aldehyde",
        "name": "Aldehyde",
        "smarts": "[CX3H1](=O)[#6]",
        "risk": "Medium",
        "mechanism": "Reacts with protein lysine residues (Schiff base) and thiol groups. Formaldehyde-like mechanism at high exposures. Reactive metabolites also possible.",
        "tox21_targets": ["SR-ARE"],
    },
    # --- Endocrine disruption ---
    {
        "id": "estrogen_mimic",
        "name": "Phenolic Estrogen Mimic",
        "smarts": "c1ccc(O)cc1",
        "risk": "Medium",
        "mechanism": "Phenols may bind the estrogen receptor (ER-α / ER-β). Bisphenol A class EDC mechanism. Assessed by NR-ER assay.",
        "tox21_targets": ["NR-ER", "NR-ER-LBD"],
    },
    {
        "id": "androgen_disruptor",
        "name": "Androgen Receptor Ligand (scaffold)",
        "smarts": "[#6]1~[#6]~[#6]~[#6]2~[#6]~[#6]~[#6]~[#6]~[#6]2~[#6]1",
        "risk": "Low",
        "mechanism": "Steroidal or steroid-like scaffolds may bind the androgen receptor (AR), acting as agonists or antagonists. Assessed by NR-AR assay.",
        "tox21_targets": ["NR-AR", "NR-AR-LBD"],
    },
    # --- Mitochondrial toxicity ---
    {
        "id": "poly_halogen",
        "name": "Polychloro/Bromo Aromatic",
        "smarts": "c-[Cl,Br]",
        "risk": "Medium",
        "mechanism": "Halogenated aromatics (PCBs, dioxins) are persistent, bioaccumulate, and activate AhR → CYP1A induction, immunotoxicity, endocrine disruption.",
        "tox21_targets": ["NR-AhR"],
    },
    {
        "id": "thiol_reactive",
        "name": "Thiol-Reactive Group (disulfide / thioester)",
        "smarts": "[#16X2][#16X2]",
        "risk": "Medium",
        "mechanism": "Reacts with glutathione and cysteine residues causing GSH depletion and oxidative stress. Common pathway for hepatotoxicity.",
        "tox21_targets": ["SR-ARE", "SR-MMP"],
    },
    # --- Genotoxicity via AhR ---
    {
        "id": "PAH",
        "name": "Polycyclic Aromatic Hydrocarbon (PAH)",
        "smarts": "c1ccc2ccccc2c1",
        "risk": "High",
        "mechanism": "Metabolised by CYP1A1/1B1 to diol-epoxides that intercalate DNA. Classic carcinogens (benzo[a]pyrene, pyrene). Activate AhR receptor. Fused ring count and planarity are key genotoxicity drivers.",
        "tox21_targets": ["NR-AhR", "SR-p53"],
    },
    {
        "id": "nitroso",
        "name": "Nitroso Group",
        "smarts": "[N;!$(N-N)](=O)",
        "risk": "High",
        "mechanism": "N-nitroso compounds are potent carcinogens. Activated by CYP2E1 to diazonium ions that methylate/ethylate DNA at O6-guanine.",
        "tox21_targets": ["SR-p53", "NR-AhR"],
    },
    {
        "id": "hydrazine",
        "name": "Hydrazine / Hydrazide",
        "smarts": "[NX3][NX3]",
        "risk": "High",
        "mechanism": "Metabolised to reactive radical or carbonium species. Potent liver toxin and carcinogen (e.g. isoniazid metabolites).",
        "tox21_targets": ["SR-ARE", "SR-p53"],
    },
    {
        "id": "sulfonyl_halide",
        "name": "Sulfonyl Halide",
        "smarts": "[SX4](=[OX1])(=[OX1])[F,Cl,Br,I]",
        "risk": "High",
        "mechanism": "Highly reactive electrophile. Covalently modifies protein residues and DNA non-specifically. Potential skin and respiratory sensitiser.",
        "tox21_targets": ["SR-ARE"],
    },
    {
        "id": "thioester",
        "name": "Thioester",
        "smarts": "C(=O)S",
        "risk": "Medium",
        "mechanism": "Activated carbonyl that can acylate biological nucleophiles. Intermediate reactivity between esters and acyl halides.",
        "tox21_targets": ["SR-ARE", "SR-MMP"],
    },
    {
        "id": "azide",
        "name": "Azide",
        "smarts": "[N-]=[N+]=[N-]",
        "risk": "High",
        "mechanism": "Highly reactive functional group. Can release nitrogen gas (explosive hazard) and act as a strong nucleophile or generate nitrenes.",
        "tox21_targets": ["SR-ARE", "SR-p53"],
    },
    {
        "id": "isothiazolone",
        "name": "Isothiazolone / Isothiazole",
        "smarts": "c1nsc(C)c1",
        "risk": "Medium",
        "mechanism": "Common biocide and skin sensitizer. Reacts with thiol groups on proteins through ring-opening or sulfur-thiol exchange.",
        "tox21_targets": ["SR-ARE"],
    },
    # --- Sulfonamide allergenic ---
    {
        "id": "sulfonamide",
        "name": "Sulfonamide",
        "smarts": "[SX4](=[OX1])(=[OX1])[NX3]",
        "risk": "Low",
        "mechanism": "Para-amino sulfonamides are associated with hypersensitivity reactions. Hydroxylamine metabolites are haptens that trigger immune responses.",
        "tox21_targets": ["SR-ARE"],
    },
]



# ---------------------------------------------------------------------------
# Screening Functions
# ---------------------------------------------------------------------------

def match_alerts(smiles: str) -> list[dict]:
    """
    Screen a SMILES against all toxicophores.
    Returns a list of matched alert dicts (with added 'matched_atoms' key).
    """
    if not RDKIT_AVAILABLE:
        return []

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    hits = []
    for alert in TOXICOPHORE_LIBRARY:
        try:
            patt = Chem.MolFromSmarts(alert["smarts"])
            if patt is None:
                continue
            matches = mol.GetSubstructMatches(patt)
            if matches:
                entry = dict(alert)
                entry["matched_atoms"] = [atom for match in matches for atom in match]
                entry["n_matches"] = len(matches)
                hits.append(entry)
        except Exception:
            continue

    # Sort by risk level
    risk_order = {"High": 0, "Medium": 1, "Low": 2}
    hits.sort(key=lambda x: risk_order.get(x["risk"], 3))
    return hits


def render_with_highlights(smiles: str, atom_indices: list[int], color=(0.9, 0.2, 0.2)) -> Optional[str]:
    """
    Render a molecule SVG with specified atoms highlighted in a given colour.
    Returns SVG string or None.
    """
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    highlight_atoms = list(set(atom_indices))
    highlight_bonds = []
    for bond in mol.GetBonds():
        ai, aj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if ai in highlight_atoms and aj in highlight_atoms:
            highlight_bonds.append(bond.GetIdx())

    atom_colors = {a: color for a in highlight_atoms}
    bond_colors = {b: color for b in highlight_bonds}

    drawer = rdMolDraw2D.MolDraw2DSVG(450, 350)
    drawer.drawOptions().addStereoAnnotation = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def render_plain(smiles: str, width: int = 400, height: int = 300) -> Optional[str]:
    """Render a plain molecule SVG without highlights."""
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    d = rdMolDraw2D.MolDraw2DSVG(width, height)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return d.GetDrawingText()


def screen_summary(smiles: str) -> dict:
    """
    Return a summary dict for quick display:
        {
            'alerts': [...],
            'max_risk': 'High' / 'Medium' / 'Low' / 'None',
            'all_highlighted_atoms': [...],
        }
    """
    alerts = match_alerts(smiles)
    all_atoms = []
    for a in alerts:
        all_atoms.extend(a.get("matched_atoms", []))

    max_risk = "None"
    if alerts:
        risk_order = {"High": 0, "Medium": 1, "Low": 2}
        max_risk = min(alerts, key=lambda x: risk_order.get(x["risk"], 3))["risk"]

    return {
        "alerts": alerts,
        "max_risk": max_risk,
        "all_highlighted_atoms": list(set(all_atoms)),
    }


def suggest_optimizations(smiles: str) -> list[str]:
    """
    Rule-based drug optimization suggestions based on detected toxicophores.
    Returns a list of plain-English suggestions with specific bioisostere pairs.
    """
    alerts = match_alerts(smiles)
    suggestions = []

    for alert in alerts:
        aid = alert["id"]
        if aid == "nitro_aromatic":
            suggestions.append("🔄 **Nitro group (–NO₂)**: Replace with bioisosteres like **–CF₃** (trifluoromethyl), **–CN** (cyano), or **–SO₂CH₃** (methylsulfone). These maintain electron-withdrawing properties without the genotoxicity.")
        elif aid == "aromatic_amine":
            suggestions.append("🔄 **Aromatic amine (–NH₂)**: High genotoxicity risk. Consider **N-acylation** (amide formation), **N-sulfonylation**, or replacing with **–OH**, **–F**, or **–OCH₃** to block reactive nitrenium ion formation.")
        elif aid == "michael_acceptor":
            suggestions.append("🔄 **Michael acceptor**: High reactivity towards thiols. Try **saturating the double bond**, adding a **β-methyl group** to provide steric hindrance, or replacing with a **saturated heterocycle**.")
        elif aid == "quinone":
            suggestions.append("🔄 **Quinone**: High redox risk. Replace with a **hydroquinone** (if reversible), or change the aromatic scaffold to a **non-quinoid system** like a pyridine or pyrimidine.")
        elif aid == "epoxide":
            suggestions.append("🔄 **Epoxide**: Extremely reactive. Replace with a **fluoro-alkyl** group or a **1,2-diol** if the polarity is acceptable. Alternatively, use a **oxetane** or **cyclopropane** as a less reactive structural mimic.")
        elif aid == "alkyl_halide":
            suggestions.append("🔄 **Activated alkyl halide**: Covalent hazard. Replace with **–CH₂OH**, **–CH₂NH₂**, or an **ether** linkage. If a halogen is needed for binding, try a **vinyl fluoride** or **aromatic fluoride** which are much less reactive.")
        elif aid == "aldehyde":
            suggestions.append("🔄 **Aldehyde (–CHO)**: Try bioisosteres like **tetrazole**, **hydroxamic acid**, or **primary sulfonamide**. If needed for binding, consider a **ketone** or a **nitrile** as less reactive alternatives.")
        elif aid == "acyl_halide":
            suggestions.append("🔄 **Acyl halide**: Too reactive for drugs. Use a stable **amide**, **ester**, or **carbamate** as a prodrug or stable equivalent.")
        elif aid == "isocyanate":
            suggestions.append("🔄 **Isocyanate**: Replace with a stable **urea** or **carbamate** derivative.")
        elif aid == "PAH":
            suggestions.append("🔄 **PAH (Polycyclic Aromatic)**: Reduce the number of fused rings (target **≤ 3**). Introduce **nitrogen or oxygen atoms** into the rings to reduce planarity and intercalation potential.")
        elif aid == "nitroso":
            suggestions.append("🔄 **Nitroso**: Very high cancer risk. Replace with a **hydroxylamine** or convert to a stable **amide** or **sulfonamide**.")
        elif aid == "poly_halogen":
            suggestions.append("🔄 **Poly-halogenated aromatic**: Bioaccumulation hazard. Replace **Cl/Br** with **F** (fluorine). Fluorine is more metabolically stable and less likely to bioaccumulate.")
        elif aid == "estrogen_mimic":
            suggestions.append("🔄 **Phenol (potential hormone mimic)**: Often binds to ER. Try **etherifying** the –OH to **–OCH₃** or **–OCHF₂** (difluoromethoxy) to block hydrogen bonding to the receptor.")
        elif aid == "hydrazine":
            suggestions.append("🔄 **Hydrazine (–NH–NH₂)**: Liver toxicity hazard. Replace with an **amide**, **urea**, or a **saturated N-heterocycle** like piperidine.")
        elif aid == "sulfonyl_halide":
            suggestions.append("🔄 **Sulfonyl halide**: Covalent hazard. Replace with a **sulfonamide (–SO₂NH₂)** or a **sulfone (–SO₂R)**.")
        elif aid == "thioester":
            suggestions.append("🔄 **Thioester**: Metabolic instability. Replace with a standard **ester** or an **amide** (which is more enzymatically stable).")
        elif aid == "azide":
            suggestions.append("🔄 **Azide (–N₃)**: Explosive/Reactive hazard. Replace with a **triazole** (common bioisostere via 'click' chemistry) or an **amide**.")

    if not suggestions:
        suggestions.append("✅ No high-priority structural modifications suggested. The molecule appears relatively clean from a toxicophore perspective.")

    return suggestions

