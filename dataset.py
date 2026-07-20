import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData

def clean_str(x):
    if isinstance(x, str):
        return x.strip().lower().replace(' ', '_')
    return x

def load_data(csv_path="data/dataset.csv", severity_path="data/Symptom-severity.csv"):
    df = pd.read_csv(csv_path)

    # Load severities
    severity_df = pd.read_csv(severity_path)
    severity_df['Symptom'] = severity_df['Symptom'].apply(clean_str)
    severity_dict = dict(zip(severity_df['Symptom'], severity_df['weight']))

    # Clean disease names
    df['Disease'] = df['Disease'].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Extract symptoms and clean
    symptoms_cols = [col for col in df.columns if 'Symptom' in col]

    # Build frequency map: disease -> symptom -> count
    disease_symptom_counts = {}
    disease_counts = {}

    all_symptoms = set()
    all_diseases = set(df['Disease'].unique())

    for _, row in df.iterrows():
        dis = row['Disease']
        disease_counts[dis] = disease_counts.get(dis, 0) + 1

        if dis not in disease_symptom_counts:
            disease_symptom_counts[dis] = {}

        for col in symptoms_cols:
            symp = clean_str(row[col])
            if isinstance(symp, str) and symp != "" and symp != "nan":
                all_symptoms.add(symp)
                disease_symptom_counts[dis][symp] = disease_symptom_counts[dis].get(symp, 0) + 1

    symptoms = sorted(list(all_symptoms))
    diseases = sorted(list(all_diseases))

    # Map to IDs
    symptom_to_id = {symp: i for i, symp in enumerate(symptoms)}
    id_to_symptom = {i: symp for i, symp in enumerate(symptoms)}

    disease_to_id = {dis: i for i, dis in enumerate(diseases)}
    id_to_disease = {i: dis for i, dis in enumerate(diseases)}

    # Create Edges and Weights
    disease_nodes = []
    symptom_nodes = []
    edge_weights = []

    for dis, symp_dict in disease_symptom_counts.items():
        dis_id = disease_to_id[dis]
        total_dis_occurrences = disease_counts[dis]

        for symp, count in symp_dict.items():
            symp_id = symptom_to_id[symp]

            # Probability of symptom given disease
            prob = count / total_dis_occurrences

            # Intrinsic severity (default to 1.0 if not found)
            severity = severity_dict.get(symp, 1.0)

            # Final Edge Weight: combination of prevalence and severity
            weight = prob * severity

            disease_nodes.append(dis_id)
            symptom_nodes.append(symp_id)
            edge_weights.append(weight)

    return symptom_to_id, id_to_symptom, disease_to_id, id_to_disease, disease_nodes, symptom_nodes, edge_weights

def get_graph_dataset(csv_path="data/dataset.csv", severity_path="data/Symptom-severity.csv"):
    symp_map, id_to_symp, dis_map, id_to_dis, d_nodes, s_nodes, e_weights = load_data(csv_path, severity_path)

    data = HeteroData()

    num_diseases = len(dis_map)
    num_symptoms = len(symp_map)

    data['disease'].num_nodes = num_diseases
    data['symptom'].num_nodes = num_symptoms

    # Initial features (Identity matrix)
    data['disease'].x = torch.eye(num_diseases)
    data['symptom'].x = torch.eye(num_symptoms)

    # Edges
    edge_index = torch.tensor([d_nodes, s_nodes], dtype=torch.long)
    edge_weight = torch.tensor(e_weights, dtype=torch.float)

    # Bidirectional
    data['disease', 'exhibits', 'symptom'].edge_index = edge_index
    data['disease', 'exhibits', 'symptom'].edge_weight = edge_weight

    data['symptom', 'rev_exhibits', 'disease'].edge_index = torch.tensor([s_nodes, d_nodes], dtype=torch.long)
    data['symptom', 'rev_exhibits', 'disease'].edge_weight = edge_weight

    return data, symp_map, id_to_symp, dis_map, id_to_dis

if __name__ == "__main__":
    data, symp_map, _, dis_map, _ = get_graph_dataset()
    print("Graph built successfully with Weighted Edges!")
    print(data)
    print(f"Number of diseases: {len(dis_map)}")
    print(f"Number of symptoms: {len(symp_map)}")
    print(f"Number of edges: {data['disease', 'exhibits', 'symptom'].edge_index.size(1)}")
    print(f"Sample Edge weights: {data['disease', 'exhibits', 'symptom'].edge_weight[:5]}")