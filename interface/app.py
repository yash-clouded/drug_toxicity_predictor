import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add src/ to path - FORCE LOCAL
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
else:
    # Move to front if already there but behind others
    sys.path.remove(SRC_DIR)
    sys.path.insert(0, SRC_DIR)

try:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from feature_engineering import compute_all_features, smiles_to_descriptors
from data_processing import TOX21_TARGETS, TOX21_TARGET_INFO, load_zinc
from toxicophores import match_alerts, render_with_highlights, render_plain, suggest_optimizations, screen_summary
from atom_shap import quick_heatmap_from_model
try:
    from ai_advisor import get_ai_explanation
    AI_READY = True
except ImportError:
    AI_READY = False


def get_global_shap(smiles, target):
    """Compute feature-level SHAP values for a single molecule and target."""
    try:
        import shap
        model, imputer, meta = load_assets(target)
        if not model:
            return None

        # Use descriptors only for a cleaner 'Why' explanation
        X, feat_names = compute_all_features(
            pd.Series([smiles]),
            use_fingerprints=False,
            use_descriptors=True,
            show_progress=False
        )
        X = imputer.transform(X)

        # Get base model for SHAP (handle both single XGB and legacy models)
        base_model = model
        if hasattr(model, "estimators_"):
            # If it's a VotingClassifier, grab the 'xgb' component
            base_model = next((est for nm, est in model.estimators_ if nm == "xgb"), model.estimators_[0][1])
        elif hasattr(model, "best_estimator_"):
            base_model = model.best_estimator_

        explainer = shap.TreeExplainer(base_model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list): sv = sv[1]
        elif len(sv.shape) == 3: sv = sv[:, :, 1]

        df_shap = pd.DataFrame({
            "Feature": feat_names,
            "SHAP Value": sv[0],
            "Effect": ["Increases Toxicity" if x > 0 else "Decreases Toxicity" for x in sv[0]]
        })
        # Filter for top magnitude
        df_shap["abs_shap"] = df_shap["SHAP Value"].abs()
        return df_shap.sort_values("abs_shap", ascending=False).head(10)
    except Exception:
        return None


try:
    from gnn_model import predict_gnn, PYG_AVAILABLE
except Exception:
    PYG_AVAILABLE = False

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ToxAI — Drug Toxicity Predictor",
    page_icon="🧪",
    layout="wide",
)

st.markdown("""
<style>
body { font-family: sans-serif; }
.risk-high { background:#ffd6d6; border-left:5px solid #d62728; padding:12px; border-radius:8px; color:#c62828; margin-bottom:10px; }
.risk-medium { background:#fff3cd; border-left:5px solid #ff7f0e; padding:12px; border-radius:8px; color:#856404; margin-bottom:10px; }
.risk-low { background:#d4edda; border-left:5px solid #2ca02c; padding:12px; border-radius:8px; color:#155724; margin-bottom:10px; }
.mechanism-text { color: #333 !important; font-size: 0.9em; margin-top: 5px; }
.mol-card    { background:white; border-radius:10px; padding:10px; display:flex; justify-content:center; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
@st.cache_resource
def load_assets(target):
    safe = target.replace("-", "_")
    try:
        model   = joblib.load(os.path.join(MODEL_DIR, f"{safe}_xgb.pkl"))
        imputer = joblib.load(os.path.join(MODEL_DIR, f"{safe}_imputer.pkl"))
        with open(os.path.join(MODEL_DIR, f"{safe}_meta.json")) as f:
            meta = json.load(f)
        return model, imputer, meta
    except Exception:
        return None, None, None


def get_prediction(smiles, target):
    model, imputer, meta = load_assets(target)
    if not model:
        return None
    X, _ = compute_all_features(pd.Series([smiles]), fp_bits=meta.get("fp_bits", 2048), show_progress=False)
    X = imputer.transform(X)
    try:
        return float(model.predict_proba(X)[0, 1])
    except:
        return None


def get_gnn_prediction(smiles, target):
    if not PYG_AVAILABLE:
        return None
    safe = target.replace("-", "_")
    ckpt = os.path.join(MODEL_DIR, f"{safe}_gnn.pt")
    if not os.path.exists(ckpt):
        return None
    try:
        return float(predict_gnn(ckpt, [smiles])[0])
    except Exception:
        return None


def svg_block(svg, caption=""):
    html = f'<div class="mol-card">{svg}</div>'
    if caption:
        html += f'<p style="text-align:center;color:#888;font-size:0.8em;">{caption}</p>'
    st.markdown(html, unsafe_allow_html=True)


def risk_badge(risk):
    colors = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c", "None": "#2ca02c"}
    c = colors.get(risk, "#888")
    return f'<span style="background:{c};color:white;padding:2px 10px;border-radius:10px;font-size:0.82em;font-weight:bold;">{risk}</span>'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Initialize global state
    if "results" not in st.session_state: st.session_state["results"] = None
    if "last_smiles" not in st.session_state: st.session_state["last_smiles"] = ""
    if "desc" not in st.session_state: st.session_state["desc"] = {}
    if "target_idx" not in st.session_state: st.session_state["target_idx"] = 0

    st.title("🧪 ToxAI: Advanced Drug Toxicity Predictor")
    st.caption("XGBoost Model · Graph Neural Network · Chemistry Alerts · Drug Optimization")
    st.divider()

    EXAMPLE_SMILES = {
        "Aspirin (safe)":              "CC(=O)Oc1ccccc1C(=O)O",
        "Tamoxifen":                   "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
        "Benzo[a]pyrene (carcinogen)": "c1ccc2cc3ccc4cccc5ccc(c1)c2c3c45",
        "Caffeine":                    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "Ibuprofen":                   "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    }

    tabs = st.tabs([
        "🔍 Predict",
        "⚗️ Chemistry Alerts",
        "🌡️ Atom Heatmap",
        "⚖️ Compare Molecules",
        "💊 Drug Optimization",
        "🧠 GNN vs XGBoost",
        "📊 ZINC Dataset",
        "ℹ️ Model Info",
    ])

    with st.sidebar:
        st.header("🔬 Molecule Input")

        def _set_smiles(new_smi):
            st.session_state["global_smiles"] = new_smi
            st.session_state["results"] = None
            st.session_state["target_idx"] = 0

        def _load_example():
            chosen = st.session_state["_example_sel"]
            if chosen != "Custom":
                _set_smiles(EXAMPLE_SMILES[chosen])

        st.selectbox("Load Example", ["Custom"] + list(EXAMPLE_SMILES.keys()), key="_example_sel", on_change=_load_example)
        smiles = st.text_area("SMILES String", height=90, key="global_smiles")
        analyze = st.button("🔬 Analyze Molecule", type="primary", use_container_width=True)

        if smiles:
            svg = render_plain(smiles, 280, 220)
            if svg: svg_block(svg)
            else: st.error("Invalid SMILES")
        st.divider()

        if smiles and analyze:
            st.subheader("📝 Export Results")
            sum_res = screen_summary(smiles)
            report_lines = [f"# Toxicity Audit: {smiles}", f"**Max Chemical Risk:** {sum_res['max_risk']}", f"**Alerts Found:** {len(sum_res['alerts'])}", "\n## Structural Alerts"]
            for a in sum_res['alerts']: report_lines.append(f"- {a['name']} ({a['risk']}): {a['mechanism']}")
            
            if "ai_notes" in st.session_state and st.session_state["ai_notes"]:
                report_lines.append("\n## Gemini AI: Expert Toxicological Review")
                report_lines.append(st.session_state["ai_notes"])

            st.download_button("⬇️ Download Detailed Report", "\n".join(report_lines), f"ToxAI_Report_{smiles[:10]}.md", "text/markdown", use_container_width=True)

    # TAB 1: PREDICT
    with tabs[0]:
        st.subheader("Toxicity Prediction Across All Targets")
        if smiles.strip() != st.session_state["last_smiles"].strip():
            st.session_state["results"] = None
            st.session_state["target_idx"] = 0

        if not smiles:
            st.info("Enter a SMILES in the sidebar, then click **🔬 Analyze Molecule**.")
        else:
            if analyze:
                with st.spinner("Running XGBoost models..."):
                    rows = []
                    for t in TOX21_TARGETS:
                        p = get_prediction(smiles, t)
                        if p is not None:
                            rows.append({"Target": t, "Probability": float(p), "Risk": "High" if p > 0.7 else ("Medium" if p > 0.4 else "Low")})
                    st.session_state["results"] = rows
                    st.session_state["desc"] = smiles_to_descriptors(smiles) or {}
                    st.session_state["last_smiles"] = smiles
                    st.session_state["target_idx"] = 0

            if st.session_state["results"]:
                df_res = pd.DataFrame(st.session_state["results"])
                desc = st.session_state["desc"]
                n_toxic = int(df_res["Risk"].eq("High").sum())

                if n_toxic == 0: st.success(f"✅ No high-risk flags across {len(df_res)} targets.")
                elif n_toxic <= 2: st.warning(f"⚠️ {n_toxic} high-risk target(s) flagged.")
                else: st.error(f"🚨 {n_toxic} high-risk targets.")

                # Metrics
                cols_m = st.columns(6)
                props = ["MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors", "NumAromaticRings"]
                for i, k in enumerate(props):
                    cols_m[i].metric(k, f"{float(desc.get(k, 0)):.2f}")

                # Target Selection
                bt = st.selectbox("Select target for details", df_res["Target"], index=st.session_state["target_idx"])
                st.session_state["target_idx"] = list(df_res["Target"]).index(bt)

                if bt in TOX21_TARGET_INFO:
                    info = TOX21_TARGET_INFO[bt]
                    st.info(f"**{info['full_name']}** — {info['description']}")

                st.divider()
                c1, c2 = st.columns([1.2, 1])
                with c1:
                    fig = px.bar(df_res, x="Probability", y="Target", color="Risk", orientation="h",
                                 color_discrete_map={"High":"#d62728","Medium":"#ff7f0e","Low":"#2ca02c"},
                                 title="Toxicity Probability Matrix")
                    fig.update_layout(yaxis=dict(autorange="reversed"), height=450)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.markdown(f"**Top Drivers for `{bt}`**")
                    df_s = get_global_shap(smiles, bt)
                    if df_s is not None:
                        fig_s = px.bar(df_s, x="SHAP Value", y="Feature", color="Effect",
                                     color_discrete_map={"Increases Toxicity":"#d62728","Decreases Toxicity":"#2ca02c"},
                                     orientation="h", height=400)
                        fig_s.update_layout(yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_s, use_container_width=True)
                    else: st.caption("SHAP breakdown unavailable.")

                st.dataframe(df_res.style.format({"Probability":"{:.1%}"}).background_gradient(cmap="RdYlGn_r", subset=["Probability"]), use_container_width=True)

                # AI SCIENTIST INTEGRATION
                st.divider()
                st.subheader("🧠 Gemini AI: Expert Toxicological Review")
                if AI_READY:
                    if st.button("🧪 Ask AI Scientist for Review"):
                        with st.spinner("AI is analyzing molecular features, SHAP drivers, and structural alerts..."):
                            summary_res = screen_summary(smiles)
                            ai_notes = get_ai_explanation(smiles, bt, float(df_res[df_res["Target"] == bt]["Probability"].iloc[0]), df_s, summary_res["alerts"])
                            st.session_state["ai_notes"] = ai_notes
                            st.rerun()

                if "ai_notes" in st.session_state and st.session_state["ai_notes"]:
                    st.markdown(f'<div style="background:#f0f7ff; border-left:5px solid #0056b3; padding:15px; border-radius:10px;">{st.session_state["ai_notes"]}</div>', unsafe_allow_html=True)
                    st.download_button("⬇️ Save Detailed Results", st.session_state["ai_notes"], f"AI_Review_{smiles[:8]}.md", use_container_width=True)
                else:
                    st.warning("AI Advisor module (`src/ai_advisor.py`) not found or `google-generativeai` missing.")
            else:
                st.info("Click **🔬 Analyze Molecule** to run predictions.")

    # TAB 2: CHEMISTRY ALERTS
    with tabs[1]:
        st.subheader("⚗️ Structural Toxicity Alerts")
        if smiles:
            summary = screen_summary(smiles)
            col_r, col_mol = st.columns([1, 1.2])
            with col_r:
                st.markdown(f"**Overall Risk:** {risk_badge(summary['max_risk'])}", unsafe_allow_html=True)
                if not summary['alerts']: st.success("✅ No toxicophores detected.")
                else:
                    for a in summary['alerts']:
                        st.markdown(f'<div class="risk-{a["risk"].lower()}"><b>{a["name"]}</b><br><div class="mechanism-text">{a["mechanism"]}</div></div>', unsafe_allow_html=True)
            with col_mol:
                all_atoms = summary["all_highlighted_atoms"]
                if all_atoms: svg_block(render_with_highlights(smiles, all_atoms), "Red = toxicophore atoms")
                else: svg_block(render_plain(smiles, 420, 340), "Clean structure")

    # TAB 3: ATOM HEATMAP
    with tabs[2]:
        st.subheader("🌡️ Atom-Level SHAP Heatmap")
        if smiles:
            target_h = st.selectbox("Select target", TOX21_TARGETS, key="hmap_t")
            if st.button("Generate Heatmap"):
                model, imputer, meta = load_assets(target_h)
                if model:
                    with st.spinner("Computing..."):
                        svg = quick_heatmap_from_model(smiles, model, imputer, meta, compute_all_features)
                    if svg: svg_block(svg, f"SHAP Heatmap - {target_h}")
                    else: st.warning("Heatmap error.")
                else: st.warning("Model not found.")

    # TAB 4: COMPARE
    with tabs[3]:
        st.subheader("⚖️ Side-by-Side Comparison")
        c_a, c_b = st.columns(2)
        with c_a:
            smi_a = st.text_input("SMILES A", value="CC(=O)Oc1ccccc1C(=O)O", key="c_a")
            if smi_a: svg_block(render_plain(smi_a, 320, 240))
        with c_b:
            smi_b = st.text_input("SMILES B", value="c1ccc2cc3ccc4cccc5ccc(c1)c2c3c45", key="c_b")
            if smi_b: svg_block(render_plain(smi_b, 320, 240))
        if st.button("⚖️ Compare Both") and smi_a and smi_b:
            rows = []
            for t in TOX21_TARGETS:
                pa, pb = get_prediction(smi_a, t), get_prediction(smi_b, t)
                if pa is not None: rows.append({"Target": t, "A": pa, "B": pb})
            if rows:
                df = pd.DataFrame(rows)
                fig = go.Figure([go.Bar(name="A", x=df["Target"], y=df["A"]), go.Bar(name="B", x=df["Target"], y=df["B"])])
                fig.update_layout(title="Multi-Target Comparison (Probability)")
                st.plotly_chart(fig, use_container_width=True)

    # TAB 5: DRUG OPTIMIZATION
    with tabs[4]:
        st.subheader("💊 Structural Optimization Suggestions")
        if smiles:
            st.info("Generating safety-focused modifications based on detected toxicophores...")
            opts = suggest_optimizations(smiles)
            if not opts or "✅" in opts[0]:
                st.success("✅ No structural optimizations suggested (low toxicity risk).")
            else:
                for opt_str in opts:
                    # opts[0] is just a string in the current src/toxicophores.py
                    st.markdown(opt_str)
                    
                st.divider()
                st.caption("Tip: Use these suggestions to redesign your molecule for lower toxicity risk while maintaining biological activity.")
        else:
            st.info("Analyze a molecule first to see optimization paths.")

    # TAB 6: GNN VS XGBOOST
    with tabs[5]:
        st.subheader("🧠 GNN vs XGBoost Comparison")
        if PYG_AVAILABLE:
            gnn_user_smi = st.text_input("SMILES to compare", value=smiles or "CC(=O)Oc1ccccc1C(=O)O", key="gnn_comp_s")
            if st.button("⚡ Run Comparison"):
                with st.spinner("Computing..."):
                    rows = []
                    for t in TOX21_TARGETS[:6]: # subset for speed
                        p_xgb = get_prediction(gnn_user_smi, t)
                        p_gnn = get_gnn_prediction(gnn_user_smi, t)
                        rows.append({
                            "Target": t, 
                            "XGBoost": p_xgb if p_xgb is not None else np.nan, 
                            "GNN": p_gnn if p_gnn is not None else np.nan
                        })
                    df_comp = pd.DataFrame(rows)
                    st.dataframe(df_comp.style.format("{:.1%}", subset=["XGBoost", "GNN"], na_rep="N/A"), use_container_width=True)
        else:
            st.warning("GNN backend (PyG) not fully initialized or no GNN models found in `models/`.")

    # TAB 7: ZINC DATASET
    with tabs[6]:
        st.subheader("📊 Explore ZINC250k Subset")
        zinc_path = os.path.join(DATA_DIR, "zinc250k.csv")
        df_z = load_zinc(zinc_path)
        if df_z is not None:
            st.write(f"Sampled {len(df_z)} molecules from ZINC250k lead-like set.")
            sel_zinc = st.selectbox("Select from ZINC", df_z["smiles"].head(50))
            st.button("🔍 Analyze Selected ZINC Molecule", on_click=_set_smiles, args=(sel_zinc,))
            st.dataframe(df_z.head(100), use_container_width=True)
        else:
            st.error("ZINC dataset not found at `data/zinc250k.csv`.")

    # TAB 8: INFO
    with tabs[7]:
        st.subheader("ℹ️ System Info")
        st.markdown("| Component | Details |\n|---|---|\n| **Classical** | Single XGBoost per target |\n| **GNN** | GATConv Graph Model |\n| **Features** | Morgan FP + RDKit Descriptors |")

    st.divider()
    st.caption("⚠️ Research use only.")

if __name__ == "__main__":
    main()
