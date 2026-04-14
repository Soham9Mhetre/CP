import torch
import torch.nn as nn


class CreditTemporal(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x, time_steps):

        # x: [N, F]
        # Convert to sequence format (simple grouping)
        seq = x.unsqueeze(1)  # [N, 1, F]

        out, _ = self.lstm(seq)

        return out[:, -1, :]