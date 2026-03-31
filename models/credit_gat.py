import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class CreditGAT(nn.Module):

    def __init__(self, input_dim, hidden_dim, heads=4, dropout=0.3):
        super().__init__()
        self.drop = dropout

        self.gat1 = GATConv(input_dim,    hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1,    dropout=dropout, concat=False)

        self.res  = nn.Linear(input_dim, hidden_dim)
        self.bn1  = nn.BatchNorm1d(hidden_dim * heads)
        self.bn2  = nn.BatchNorm1d(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x, edge_index):
        res = self.res(x)

        h = F.dropout(x,  p=self.drop, training=self.training)
        h = F.elu(self.bn1(self.gat1(h, edge_index)))

        h = F.dropout(h,  p=self.drop, training=self.training)
        h = F.elu(self.bn2(self.gat2(h, edge_index)) + res)

        return self.head(h)