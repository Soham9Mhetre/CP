import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, time_steps=None):

        x_seq = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x_seq)

        context = lstm_out[:, -1, :]
        context = self.proj(context)

        return x + context