from data.load_dataset import load_dataset

data = load_dataset()

print(data)
import torch

labels = data.y

print("Normal:", torch.sum(labels == 0))
print("Fraud:", torch.sum(labels == 1))
print("Unknown:", torch.sum(labels == -1))
print("Train nodes:", data.train_mask.sum())
print("Test nodes:", data.test_mask.sum())