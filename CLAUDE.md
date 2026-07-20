# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- **Install Dependencies**: `pip install -r requirements.txt`
- **Train the Model**: `python train.py` (Must be run first to generate `model_weights.pth` and `mappings.json`)
- **Run the Application**: `streamlit run app.py`
- **Verify Graph Construction**: `python dataset.py` (Runs the test block to print graph stats)
- **Run Headless Verification**: `python verify_ui.py` (Requires Playwright: `pip install playwright && playwright install chromium`)

## High-Level Architecture
This repository implements a Graph Neural Network (GNN) for disease prediction, functioning as a Knowledge Graph engineered with weighted edges and Explainable AI (XAI).

### Core Components
- **Data Pipeline (`dataset.py`)**: 
  - Loads 4,920 simulated patient records from `data/dataset.csv` and symptom severities from `data/Symptom-severity.csv`.
  - Calculates specific edge weights dynamically using (Probability of Symptom given Disease * Clinical Severity).
  - Constructs a PyTorch Geometric `HeteroData` graph mapping diseases to symptoms with these weighted bidirectional edges.
- **Model Definition (`model.py`)**: 
  - `SymptomCheckerModel` projects initial one-hot encodings into dense embeddings.
  - A core `GNN` class uses PyG's `GraphConv` layers to perform heterogeneous message passing, explicitly passing the `edge_weight` dictionary.
  - Link prediction relies on dot product interactions between target disease embeddings and aggregate symptom embeddings.
- **Training Loop (`train.py`)**: 
  - Generates positive and negative edge samples for link prediction.
  - Trains the network via Binary Cross Entropy (`F.binary_cross_entropy_with_logits`), successfully minimizing loss across 200 epochs.
- **Frontend / Explainable AI (XAI) (`app.py`)**: 
  - A Streamlit interface simulating real patient diagnosis.
  - **Inference logic**: Averages learned embeddings of selected symptoms to form a Patient embedding, dot-product scored against diseases.
  - **XAI logic**: Performs explicit feature attribution by calculating the dot product of each *individual* symptom embedding with the predicted disease embedding. Visualizes this contribution via a dynamic bar chart and edge-thickness mapping in NetworkX.