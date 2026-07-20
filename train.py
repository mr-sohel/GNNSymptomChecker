import torch
import torch.nn.functional as F
from dataset import get_graph_dataset
from model import SymptomCheckerModel
import json

def train():
    # 1. Load data
    print("Loading Kaggle Dataset and building Graph...")
    data, symp_map, id_to_symp, dis_map, id_to_dis = get_graph_dataset()

    # 2. Prepare edge labels for Link Prediction
    edge_index = data['disease', 'exhibits', 'symptom'].edge_index
    num_edges = edge_index.size(1)

    # Negative sampling: random pairs of disease-symptom that don't exist
    neg_edge_index = torch.stack([
        torch.randint(0, data['disease'].num_nodes, (num_edges,)),
        torch.randint(0, data['symptom'].num_nodes, (num_edges,))
    ], dim=0)

    # Concatenate positive edges (label=1) and negative edges (label=0)
    edge_label_index = torch.cat([edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat([torch.ones(num_edges), torch.zeros(num_edges)], dim=0)

    # 3. Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SymptomCheckerModel(
        hidden_channels=64,
        num_diseases=len(dis_map),
        num_symptoms=len(symp_map),
        metadata=data.metadata()
    ).to(device)

    data = data.to(device)
    edge_label_index = edge_label_index.to(device)
    edge_label = edge_label.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 4. Training loop
    print("Starting training...")
    model.train()
    epochs = 200
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        # Pass x_dict, edge_index_dict, edge_weight_dict, and edge_label_index
        out = model(data.x_dict, data.edge_index_dict, data.edge_weight_dict, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, edge_label)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d}/{epochs}, Loss: {loss.item():.4f}")

    # 5. Save model and mappings for the Streamlit app
    torch.save(model.state_dict(), "model_weights.pth")
    with open("mappings.json", "w") as f:
        json.dump({
            "symp_to_id": symp_map,
            "id_to_symp": id_to_symp,
            "dis_to_id": dis_map,
            "id_to_dis": id_to_dis
        }, f)

    print("Training complete! Model and mappings saved.")

if __name__ == "__main__":
    train()