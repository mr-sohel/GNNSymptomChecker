import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, to_hetero

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # GraphConv supports edge weights properly
        self.conv1 = GraphConv((-1, -1), hidden_channels)
        self.conv2 = GraphConv((-1, -1), hidden_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight)
        return x

class SymptomCheckerModel(torch.nn.Module):
    def __init__(self, hidden_channels, num_diseases, num_symptoms, metadata):
        super().__init__()

        self.disease_lin = torch.nn.Linear(num_diseases, hidden_channels)
        self.symptom_lin = torch.nn.Linear(num_symptoms, hidden_channels)

        # Heterogeneous conversion
        self.gnn = to_hetero(GNN(hidden_channels), metadata, aggr='sum')

    def forward(self, x_dict, edge_index_dict, edge_weight_dict, edge_label_index):
        # Project initial features
        x_dict_proj = {
            'disease': self.disease_lin(x_dict['disease']),
            'symptom': self.symptom_lin(x_dict['symptom'])
        }

        # Pass through Hetero GNN with edge weights
        z_dict = self.gnn(x_dict_proj, edge_index_dict, edge_weight_dict)

        # Link Prediction
        row, col = edge_label_index
        disease_emb = z_dict['disease'][row]
        symptom_emb = z_dict['symptom'][col]

        # Dot product prediction
        out = (disease_emb * symptom_emb).sum(dim=-1)
        return out
