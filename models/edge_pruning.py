import torch
import torch.nn.functional as F


def prune_edges(edge_index, x, threshold=0.15):
    row, col = edge_index

    sim = F.cosine_similarity(x[row], x[col])

    mask = sim > threshold

    return edge_index[:, mask]