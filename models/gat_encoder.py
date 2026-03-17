import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from models.spectral_filter import SpectralFilter
from models.temporal_lstm import TemporalLSTM


class FraudGAT(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):

        super().__init__()

        self.filter = SpectralFilter()

        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.6)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim)

        self.temporal = TemporalLSTM(hidden_dim, hidden_dim)

        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, time_steps):

        x = self.filter(x, edge_index)

        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = self.gat2(x, edge_index)

        x = self.temporal(x, time_steps)

        x = self.linear(x)

        return x