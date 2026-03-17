import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, embeddings, time_steps):

        unique_times = torch.unique(time_steps)

        temporal_embeddings = []

        for t in unique_times:

            mask = time_steps == t

            step_embed = embeddings[mask].mean(dim=0)

            temporal_embeddings.append(step_embed)

        seq = torch.stack(temporal_embeddings).unsqueeze(0)

        out, _ = self.lstm(seq)

        temporal_context = out.squeeze(0)

        # broadcast temporal info back to nodes
        context = temporal_context.mean(dim=0)

        return embeddings + context