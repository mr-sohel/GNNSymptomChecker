#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================="
echo "GNN Symptom Checker Setup & Execution"
echo "=========================================="

# 1. Set up a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment 'venv'..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment 'venv' already exists."
fi

echo "      Activating virtual environment..."
source venv/bin/activate

# 2. Install required dependencies
echo "[2/4] Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Train the model (only if it hasn't been trained yet)
echo "[3/4] Checking model status..."
if [ ! -f "model_weights.pth" ] || [ ! -f "mappings.json" ]; then
    echo "      Model not found. Training the model (this runs once)..."
    python train.py
else
    echo "      Model already trained ('model_weights.pth' found). Skipping training."
    echo "      (To force retrain, delete 'model_weights.pth' and 'mappings.json')"
fi

# 4. Start the Streamlit app
echo "[4/4] Starting the Streamlit application..."
streamlit run app.py
