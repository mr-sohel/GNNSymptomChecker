# GNN Symptom Checker with Explainable AI (XAI)

An advanced medical diagnostic tool built for Knowledge Engineering and Healthcare AI research. 

This project uses a **Heterogeneous Graph Neural Network (GNN)** to predict diseases based on symptoms by formulating it as a bipartite Link Prediction problem. Unlike traditional "black box" machine learning models, this project incorporates **Explainable AI (XAI)** and **Severity-Weighted Edges** to provide transparent, mathematically sound justifications for its clinical predictions.

## Key Features & Research Contributions
- **Weighted Graph Structure**: Instead of treating symptoms as binary (present/absent), the knowledge graph utilizes real-world symptom occurrence rates combined with clinical severity to assign explicit Edge Weights.
- **Heterogeneous GraphSAGE**: Replaces standard feed-forward networks with `GraphConv` layers to effectively learn latent representations by propagating information through the Disease-Symptom topological neighborhood.
- **Latent Feature Attribution (XAI)**: Calculates the explicit logit contribution of each input symptom toward the final prediction, mapping the neural network's decision back to human-readable clinical rationale.
- **Interactive Knowledge Visualization**: Dynamically renders the active sub-graph (via NetworkX) during inference, modulating edge thickness based on symptom contribution.

## Dataset
Powered by a 4,920-row clinical dataset (synthesized from Columbia University's Disease-Symptom Knowledge Base), mapping 41 distinct diseases to 131 symptoms with explicit clinical severity weights.

## Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the GNN Model**
   Run the training pipeline to generate the embeddings, calculate edge weights, and output `model_weights.pth` and `mappings.json`.
   ```bash
   python train.py
   ```

3. **Launch the XAI Web Interface**
   Start the Streamlit application to use the Symptom Checker and view the Explainable AI charts.
   ```bash
   streamlit run app.py
   ```