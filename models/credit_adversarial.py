import torch
import random


def inject_adversarial_edges(edge_index, num_nodes, ratio=0.01):

    num_edges = edge_index.size(1)
    num_fake = int(num_edges * ratio)

    fake_edges = []

    for _ in range(num_fake):
        src = random.randint(0, num_nodes - 1)
        dst = random.randint(0, num_nodes - 1)
        fake_edges.append([src, dst])

    fake_edges = torch.tensor(fake_edges).t().to(edge_index.device)

    return torch.cat([edge_index, fake_edges], dim=1)