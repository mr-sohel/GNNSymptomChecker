import streamlit as st
import torch
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from dataset import get_graph_dataset
from model import SymptomCheckerModel

@st.cache_resource
def load_model_and_data():
    with open("mappings.json", "r") as f:
        mappings = json.load(f)

    data, _, _, _, _ = get_graph_dataset()

    num_diseases = len(mappings["dis_to_id"])
    num_symptoms = len(mappings["symp_to_id"])

    model = SymptomCheckerModel(
        hidden_channels=64,
        num_diseases=num_diseases,
        num_symptoms=num_symptoms,
        metadata=data.metadata()
    )

    model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cpu')))
    model.eval()

    return model, data, mappings

st.set_page_config(page_title="GNN Symptom Checker with XAI", layout="wide")
st.title("GNN Symptom Checker & Explainable AI (XAI)")
st.write("""
This tool uses a **Heterogeneous Graph Neural Network (GNN)** with severity-weighted edges to predict diseases.
It also utilizes **Explainable AI (XAI)** to transparently show exactly *why* a disease was predicted based on the learned neighborhood embeddings.
""")

try:
    model, data, mappings = load_model_and_data()
except Exception as e:
    st.error(f"Error loading model. Please ensure you have trained the model first using `python train.py`.\n\nDetails: {e}")
    st.stop()

symptoms_list = list(mappings["symp_to_id"].keys())
selected_symptoms = st.multiselect("Select Symptoms (e.g. chills, high_fever)", options=symptoms_list)

if st.button("Predict Disease") and selected_symptoms:
    with torch.no_grad():
        # 1. Feature Projection
        x_dict_proj = {
            'disease': model.disease_lin(data.x_dict['disease']),
            'symptom': model.symptom_lin(data.x_dict['symptom'])
        }

        # 2. GNN Message Passing
        z_dict = model.gnn(x_dict_proj, data.edge_index_dict, data.edge_weight_dict)

        disease_embeddings = z_dict['disease']
        symptom_embeddings = z_dict['symptom']

        # 3. Patient Aggregation
        selected_ids = [mappings["symp_to_id"][symp] for symp in selected_symptoms]
        patient_symp_emb = symptom_embeddings[selected_ids].mean(dim=0)

        # 4. Link Prediction
        scores = (disease_embeddings * patient_symp_emb).sum(dim=-1)
        probabilities = torch.sigmoid(scores)

        top_probs, top_indices = torch.topk(probabilities, 3)

        st.divider()
        st.header("Diagnostic Results")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Top Predictions")
            top_disease = mappings["id_to_dis"][str(top_indices[0].item())]

            for prob, idx in zip(top_probs, top_indices):
                disease_name = mappings["id_to_dis"][str(idx.item())].replace('_', ' ').title()
                st.metric(label=disease_name, value=f"{prob.item():.2%}")

        # --- EXPLAINABLE AI (XAI) SECTION ---
        with col2:
            st.subheader(f"XAI: Why {top_disease.replace('_', ' ').title()}?")
            st.write("This chart shows the exact mathematical contribution of each selected symptom toward the top prediction, extracted from the Graph Neural Network's latent space.")

            # Calculate feature attribution for the top predicted disease
            top_disease_emb = disease_embeddings[top_indices[0]]

            contributions = {}
            for symp_id, symp_name in zip(selected_ids, selected_symptoms):
                # How much did this specific symptom's embedding align with the disease embedding?
                symp_emb = symptom_embeddings[symp_id]
                contrib = (top_disease_emb * symp_emb).sum().item() / len(selected_ids)
                contributions[symp_name.replace('_', ' ')] = contrib

            # Plot the XAI contributions
            fig, ax = plt.subplots(figsize=(8, 4))
            labels = list(contributions.keys())
            values = list(contributions.values())

            # Colors based on positive/negative contribution
            colors = ['#2ca02c' if v > 0 else '#d62728' for v in values]

            y_pos = np.arange(len(labels))
            ax.barh(y_pos, values, color=colors)
            ax.set_yticks(y_pos, labels=labels)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('Logit Contribution Score')
            ax.set_title('Symptom Contribution to Prediction')

            st.pyplot(fig)

        st.divider()

        # --- KNOWLEDGE GRAPH VISUALIZATION ---
        st.subheader("Sub-Graph Visualization")
        st.write("The active neighborhood of the graph linking your symptoms to the top prediction.")
        G = nx.Graph()

        # Add the disease node
        G.add_node(top_disease, color='#d62728', size=1500)

        # Add symptom nodes and edges with weights relative to contribution
        max_contrib = max(abs(v) for v in contributions.values()) if contributions else 1.0

        for symp_name, contrib in contributions.items():
            # Normalize edge thickness based on contribution
            edge_weight = max(0.5, (abs(contrib) / max_contrib) * 5.0) if max_contrib > 0 else 1.0

            G.add_node(symp_name, color='#1f77b4', size=800)
            G.add_edge(top_disease, symp_name, weight=edge_weight)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=42)

        colors = [node[1]['color'] for node in G.nodes(data=True)]
        sizes = [node[1]['size'] for node in G.nodes(data=True)]
        weights = [G[u][v]['weight'] for u,v in G.edges()]

        nx.draw(G, pos, with_labels=True, node_color=colors, node_size=sizes,
                width=weights, edge_color='gray', font_size=12, font_weight='bold', ax=ax2)
        st.pyplot(fig2)
