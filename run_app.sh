#!/bin/bash

# run_app.sh — Easy launcher for Drug Toxicity Predictor
# ------------------------------------------------------

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$BASE_DIR"

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
fi

# Ensure libomp path is set for Apple Silicon (M-series) Macs
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/homebrew/opt/libomp/lib

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    # Downgrade numpy for RDKit compatibility
    pip install "numpy<2"
    source venv/bin/activate
    # Always ensure requirements are up-to-date
    echo "Verifying dependencies..."
    pip install -q -r requirements.txt
fi

# Check for trained models
if [ ! -d "models" ] || [ -z "$(ls -A models)" ]; then
    echo "No trained models found. Running training pipeline first (this may take a few minutes)..."
    python src/train.py
fi

echo "Launching Streamlit app..."
./venv/bin/streamlit run interface/app.py
