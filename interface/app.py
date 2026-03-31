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
    if "ai_notes" not in st.session_state: st.session_state["ai_notes"] = None
    
    # Default target
    bt = TOX21_TARGETS[0]

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
            st.session_state["ai_notes"] = None
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

        if st.session_state.get("results"):
            st.subheader("📝 Export Results")
            smiles_val = st.session_state.get("last_smiles") or smiles
            sum_res = screen_summary(smiles_val)
            
            # --- START COMPREHENSIVE REPORT ---
            report_lines = [
                f"# ToxAI: Detailed Toxicity Audit",
                f"**Molecule (SMILES):** `{smiles_val}`",
                f"**Maximum Risk Flag:** {sum_res['max_risk']}",
                "\n## 🧬 Molecular Descriptors",
                "| Property | Value |",
                "|---|---|",
            ]
            desc = st.session_state.get("desc", {})
            for k, v in desc.items():
                report_lines.append(f"| {k} | {v:.4f} |")

            report_lines.append("\n## 🔬 Binary Assay Predictions (XGBoost)")
            report_lines.append("| Target | Probability | Risk Level |")
            report_lines.append("|---|---|---|")
            res_list = st.session_state.get("results") or []
            for r in res_list:
                report_lines.append(f"| {r['Target']} | {r['Probability']:.1%} | {r['Risk']} |")

            report_lines.append("\n## ⚠️ Structural Alerts (Toxicophores)")
            if sum_res['alerts']:
                for a in sum_res['alerts']: 
                    report_lines.append(f"### {a['name']} ({a['risk']})")
                    report_lines.append(f"- **Mechanism**: {a['mechanism']}")
            else:
                report_lines.append("No common toxicophores detected.")

            if st.session_state.get("ai_notes"):
                report_lines.append("\n## 🧠 Gemini AI: Expert Scientific Review")
                report_lines.append(st.session_state["ai_notes"])
            # --- END COMPREHENSIVE REPORT ---

            btn_label = "⬇️ Download FULL AI Report" if st.session_state.get("ai_notes") else "⬇️ Download Detailed Audit"
            st.download_button(btn_label, "\n".join(report_lines), f"ToxAI_Audit_{smiles_val[:8]}.md", "text/markdown", key="dl_btn", use_container_width=True)

    # TAB 1: PREDICT
    with tabs[0]:
        st.subheader("Toxicity Prediction Across All Targets")

        if not smiles:
            st.info("Enter a SMILES in the sidebar, then click **🔬 Analyze Molecule**.")
            st.session_state["results"] = None
        else:
            # Handle analysis trigger
            if analyze:
                with st.spinner("Running XGBoost models..."):
                    rows = []
                    for t in TOX21_TARGETS:
                        p = get_prediction(smiles, t)
                        if p is not None:
                            rows.append({
                                "Target": t, 
                                "Probability": float(p), 
                                "Risk": "High" if p > 0.7 else ("Medium" if p > 0.4 else "Low")
                            })
                    if rows:
                        st.session_state["results"] = rows
                        st.session_state["desc"] = smiles_to_descriptors(smiles) or {}
                        st.session_state["last_smiles"] = smiles
                        st.session_state["target_idx"] = 0
                    else:
                        st.error("No predictions generated. Check your SMILES or model paths.")
            
            # Show results if they exist for the current SMILES
            if st.session_state.get("results") and smiles.strip() == st.session_state.get("last_smiles", "").strip():
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

                if st.session_state.get("ai_notes"):
                    st.markdown(f'<div style="background:#f0f7ff; border-left:5px solid #0056b3; padding:15px; border-radius:10px; color: #111;">{st.session_state["ai_notes"]}</div>', unsafe_allow_html=True)
                else:
                    st.info("Click the button above to generate a detailed scientific report.")

    # TAB 2: ALERTS
    with tabs[1]:
        st.subheader("⚠️ Structural Alert Mapping")
        res = match_alerts(smiles)
        if res:
            st.warning(f"Found {len(res)} structural alerts.")
            cols = st.columns(2)
            with cols[0]:
                svg = render_with_highlights(smiles, res)
                if svg: svg_block(svg, "Highlighted Reactive Sites")
            with cols[1]:
                for a in res:
                    st.markdown(f"**{a['name']}** ({a['risk']})")
                    st.caption(f"Mechanism: {a['mechanism']}")
                    st.divider()
        else:
            st.success("No common toxicophore alerts matched.")

    # TAB 3: HEATMAP
    with tabs[2]:
        st.subheader("🔥 Atom-Level Toxicity Heatmap")
        res_list = st.session_state.get("results")
        if res_list:
            df_res = pd.DataFrame(res_list)
            # Add a dropdown to select which target to visualize
            bt_name = st.selectbox("Select assay target for heatmap", df_res["Target"], key="heatmap_target_sel")
            
            st.write(f"Analyzing contribution of each atom to `{bt_name}` toxicity.")
            model, imputer, meta = load_assets(bt_name)
            if model and imputer and meta:
                svg_h = quick_heatmap_from_model(
                    smiles, model, imputer, meta, compute_all_features,
                    fp_bits=meta.get("fp_bits", 2048)
                )
                if svg_h: svg_block(svg_h, f"Heatmap for {bt_name}")
                else: st.error("Failed to generate heatmap.")
            else: st.error("Model assets not available for heatmap.")
        else:
            st.info("Run an analysis first to explore atom-level heatmaps.")

    # TAB 4: COMPARE
    with tabs[3]:
        st.subheader("⚖️ Compare Two Molecules")
        s2 = st.text_input("Enter second molecule SMILES", "O=C(O)c1ccccc1")
        
        # Check if we have results to get a meaningful target or just use default
        comp_target = bt
        res_list = st.session_state.get("results")
        if res_list:
            comp_target = res_list[st.session_state.get("target_idx", 0)]["Target"]

        if st.button("📊 Compare"):
            p1 = get_prediction(smiles, comp_target)
            p2 = get_prediction(s2, comp_target)
            
            st.write(f"Comparing target: **{comp_target}**")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Molecule A** ({p1:.1%})")
                svg1 = render_plain(smiles, 300, 250)
                if svg1: svg_block(svg1)
            with c2:
                st.markdown(f"**Molecule B** ({p2:.1%})")
                svg2 = render_plain(s2, 300, 250)
                if svg2: svg_block(svg2)

    # TAB 5: OPTIMIZATION
    with tabs[4]:
        st.subheader("💊 Structural Redesign Suggestions")
        col_o1, col_o2 = st.columns([1, 1])
        
        with col_o1:
            st.markdown("### 🛠️ Rule-Based Bioisosteres")
            opts = suggest_optimizations(smiles)
            if opts:
                for o in opts:
                    with st.expander(f"**{o['original']}** ⮕ **{o['suggestion']}**"):
                        st.info(o['reason'])
            else:
                st.success("No high-priority structural risks identified by rules.")
                
        with col_o2:
            st.markdown("### 🧠 AI Scientist Redesign")
            if st.button("🚀 Ask AI for Redesign Strategies", key="ai_opt_btn"):
                res_list = st.session_state.get("results")
                if res_list:
                    with st.spinner("AI Scientist is rethinking the scaffold..."):
                        # Find highest risk target for context
                        hi_risk = max(res_list, key=lambda x: x["Probability"])
                        bt = hi_risk["Target"]
                        prob = hi_risk["Probability"]
                        
                        # Get SHAP drivers
                        df_s = get_global_shap(smiles, bt)
                        # Get alerts
                        summ = screen_summary(smiles)
                        
                        redesign_notes = get_ai_explanation(smiles, bt, float(prob), df_s, summ["alerts"])
                        st.session_state["ai_redesign"] = redesign_notes
                        st.write(redesign_notes)
                else:
                    st.warning("Run an analysis first to provide context for AI redesign.")
            
            elif "ai_redesign" in st.session_state:
                st.write(st.session_state["ai_redesign"])

    # TAB 6: GNN COMPARISON
    with tabs[5]:
        st.subheader("🧠 GNN vs XGBoost (Head-to-Head)")
        if PYG_AVAILABLE:
            gnn_user_smi = smiles or EXAMPLE_SMILES["Aspirin (safe)"]
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
