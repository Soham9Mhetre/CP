import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score

from data.load_dataset import load_dataset
from models.gat_encoder import FraudGAT
from models.spectral_filter import SpectralFilter
from models.edge_pruning import prune_edges

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================
data = load_dataset().to(device)

# =========================
# FIX LABELS (ROBUST + SAFE)
# =========================
data.y = data.y.long()

unique_labels = torch.unique(data.y)
print("Original labels:", unique_labels)

# Case 1: Already binary → keep as is
if len(unique_labels) == 2:
    print("Binary dataset detected — no conversion applied")

# Case 2: Multi-class → map ONLY class 1 as fraud
else:
    print("Multi-class dataset detected — converting to binary (class 1 = fraud)")
    data.y = (data.y == 1).long()

print("Final labels:", torch.unique(data.y))
print("Label distribution:", torch.bincount(data.y))

# Safety check
assert data.y.min() >= 0 and data.y.max() < 2, "Labels must be 0 or 1"

# =========================
# CREATE SPLITS
# =========================
num_nodes = data.x.shape[0]
perm = torch.randperm(num_nodes, device=device)

train_end = int(0.7 * num_nodes)
val_end = int(0.85 * num_nodes)

train_idx = perm[:train_end]
val_idx = perm[train_end:val_end]
test_idx = perm[val_end:]

data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

data.train_mask[train_idx] = True
data.val_mask[val_idx] = True
data.test_mask[test_idx] = True

# =========================
# MODEL
# =========================
input_dim = data.x.shape[1]

model = FraudGAT(input_dim, 32, 2).to(device)
spectral = SpectralFilter(alpha=0.1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# =========================
# TRAINING LOOP
# =========================
for epoch in range(200):
    model.train()

    x = spectral(data.x, data.edge_index)
    edge_pruned = prune_edges(data.edge_index, x, threshold=0.15)

    logits = model(x, edge_pruned, data.time_steps)

    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss {loss.item():.4f}")

# =========================
# EVALUATION
# =========================
model.eval()

with torch.no_grad():
    x = spectral(data.x, data.edge_index)
    edge_pruned = prune_edges(data.edge_index, x, threshold=0.15)

    logits = model(x, edge_pruned, data.time_steps)
    preds = logits.argmax(dim=1)

    print("\n--- CRYPTO EVALUATION ---\n")
    print("Accuracy:",
          accuracy_score(
              data.y[data.test_mask].cpu(),
              preds[data.test_mask].cpu()
          ))

    print(classification_report(
        data.y[data.test_mask].cpu(),
        preds[data.test_mask].cpu()
    ))

# =========================
# TEMPERATURE CALIBRATION
# =========================
import torch.nn as nn

class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.T


def calibrate_temperature(logits, labels):
    scaler = TempScaler().to(logits.device)
    optimizer = torch.optim.LBFGS([scaler.T], lr=0.01, max_iter=50)

    loss_fn = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler.T.item()


val_logits = logits[data.val_mask]
val_labels = data.y[data.val_mask]

temperature = calibrate_temperature(val_logits, val_labels)

print(f"\nLearned temperature: {temperature:.4f}")

# =========================
# SAVE MODEL
# =========================
torch.save({
    "model_state_dict": model.state_dict(),
    "input_dim": input_dim,
    "threshold": 0.5,
    "temperature": float(temperature)
}, "models/crypto_model.pth")

print("\n✅ Crypto model saved successfully.")
