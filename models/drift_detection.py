import torch


def detect_drift(x, normalize=True):

    mean = x.mean(dim=0, keepdim=True)
    drift = torch.abs(x - mean).mean(dim=1)

    if normalize:
        drift = drift / (drift.max() + 1e-6)

    return drift