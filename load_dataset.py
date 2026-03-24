from torch_geometric.datasets import EllipticBitcoinDataset
import torch

def load_dataset():

    dataset = EllipticBitcoinDataset(root="data/")
    data = dataset[0]

    # Extract time steps
    time_steps = data.x[:, 1].long()

    data.time_steps = time_steps

    return data
