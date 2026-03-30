# 🧪 ToxAI — Advanced Drug Toxicity Predictor

> **End-to-End interpretation:** SMILES ⮕ ML Models ⮕ SHAP Explainer ⮕ AI Scientist

---

## 📌 Overview

**ToxAI** is a high-performance drug discovery platform that predicts molecular toxicity across 12 biochemical targets (Tox21 benchmark). It provides multi-level interpretability, combining classical ML (XGBoost), Deep Learning (GNN), and Large Language Models (Gemini) to explain *why* a molecule might be toxic and how to fix it.

### **✨ Key Features**
- **Dual-Model Predictions**: Head-to-head comparison between **XGBoost** (descriptors) and **GNN** (graph-level anatomy).
- **🧠 AI Scientist (Gemini)**: Automated toxicological reviews and redesign suggestions using LLM reasoning.
- **Micro-Analysis**: Atom-level SHAP heatmaps highlighting specific toxicophores.
- **Lead Discovery**: Integrated explorer for the **ZINC 250k** lead-like dataset.
- **Rule-Based Guidance**: Built-in chemistry alerts and bioisostere optimization suggestions.

---

## 🗂️ Project Structure

```
drug_toxicity_predictor/
├── data/              # Tox21 and ZINC benchmark datasets
├── models/            # XGBoost (.pkl) and GNN (.pt) assets
├── interface/
│   └── app.py         # 8-tab Streamlit Dashboard
├── src/
│   ├── ai_advisor.py  # Gemini LLM Bridge
│   ├── gnn_model.py   # PyTorch Geometric GNN (GATConv)
│   ├── atom_shap.py   # Visual interpretability engine
│   ├── toxicophores.py # Rule-based safety engine
│   ├── train.py       # ML training pipeline
│   └── feature_engineering.py
├── run_app.sh         # Intelligent launcher
├── .env.example       # API configuration template
└── requirements.txt
```

---

## ⚙️ Setup & Launch

### 1. Requirements
Ensure you have **Python 3.9+** and a working C-compiler (for RDKit/XGBoost).

### 2. Environmental Keys
To enable the **AI Scientist**, add your Gemini API key:
1. Copy `.env.example` to `.env`.
2. Insert your key: `GEMINI_API_KEY=your_key_here`.

### 3. Quick Launch
The project includes an intelligent launcher that handles dependencies for Apple Silicon and Intel Macs:
```bash
chmod +x run_app.sh
./run_app.sh
```

---

## 📊 Technical Stack

| Category | Technologies |
|---|---|
| **Core ML** | XGBoost, Scikit-learn, Optuna |
| **Deep Learning** | PyTorch, PyTorch Geometric (GATConv) |
| **Cheminformatics** | RDKit |
| **Interpretability** | SHAP (Tree & Kernel), Matplotlib |
| **AI Advisor** | Google Generative AI (Gemini 1.5 Flash) |
| **Interface** | Streamlit, Plotly, Seaborn |

---

## 🔍 Interpretability Workflow

1. **Input**: Enter a SMILES string or select from ZINC.
2. **Predict**: Review probabilities across 12 Tox21 assays.
3. **Heatmap**: Visualize which specific atoms drive the risk score.
4. **AI Review**: Click **🧪 Ask AI Scientist** to receive a structured scientific report and redesign suggestions.

---

## 📄 License
MIT License. Developed for research and early-stage drug screening.
