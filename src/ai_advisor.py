import os
import google.generativeai as genai
import pandas as pd

def get_ai_explanation(smiles: str, target: str, probability: float, shap_drivers: pd.DataFrame, alerts: list) -> str:
    """
    Generate a scientific explanation and design suggestions using Gemini.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ **AI explanation unavailable.** Please set the `GEMINI_API_KEY` environment variable to enable advanced AI-powered drug insights."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Format input for the LLM
        risk = "High" if probability > 0.7 else ("Medium" if probability > 0.4 else "Low")
        
        driver_text = ""
        if shap_drivers is not None and not shap_drivers.empty:
            driver_text = "\n".join([f"- {row['Feature']}: {row['SHAP Value']:.4f} ({row['Effect']})" for _, row in shap_drivers.iterrows()])
        
        alert_text = ""
        if alerts:
            alert_text = "\n".join([f"- {a['name']} ({a['risk']}): {a['mechanism']}" for a in alerts])

        prompt = f"""
        As a medicinal chemist and toxicologist, analyze this drug toxicity prediction:
        
        **Molecule (SMILES):** {smiles}
        **Assay Target:** {target}
        **Predicted Toxicity Probability:** {probability:.1%} ({risk} Risk)
        
        **Top SHAP Feature Drivers:**
        {driver_text or "No specific descriptors flagged."}
        
        **Detected Structural Alerts (Toxicophores):**
        {alert_text or "No structural alerts found."}
        
        **Your Task:**
        1. Explain the biochemical rationale for this prediction. Why do the features and alerts indicate {risk} risk for {target}?
        2. Suggest 2-3 specific structural modifications (bioisosteres or group removals) to reduce the toxicity while maintaining potential binding affinity.
        3. Provide a brief summary of the overall safety profile of this molecule.
        
        Keep the response professional, scientific, and concise (max 300 words). Use markdown formatting.
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"❌ **AI Error:** Could not generate explanation ({str(e)}). Check your API quota or network connection."

if __name__ == "__main__":
    # Quick test
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    test_drivers = pd.DataFrame({"Feature": ["MolLogP", "TPSA"], "SHAP Value": [0.1, -0.05], "Effect": ["Increases Toxicity", "Decreases Toxicity"]})
    print(get_ai_explanation(test_smiles, "NR-AR", 0.65, test_drivers, []))
