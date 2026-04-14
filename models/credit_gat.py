import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class CreditGAT(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.gat1 = GATConv(input_dim, hidden_dim, heads=4)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim)

        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index):

        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = self.gat2(x, edge_index)

        return self.fc(x)