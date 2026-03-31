import torch
import torch.nn as nn
import torch.nn.functional as F


class CreditLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.out_dim = hidden_dim * 2

        self.attn = nn.Linear(self.out_dim, 1)
        self.norm = nn.LayerNorm(self.out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)                        # (B, T, H*2)
        w = torch.softmax(self.attn(out), dim=1)     # (B, T, 1)
        ctx = (out * w).sum(dim=1)                   # (B, H*2)
        return self.drop(self.norm(ctx))