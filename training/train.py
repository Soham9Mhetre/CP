import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve
import random
import copy

from data.load_dataset import load_dataset
from models.spectral_filter import SpectralFilter
from models.gat_encoder import FraudGAT
from models.adversarial_injection import inject_adversarial_edges

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# LOAD BASE DATA
# ======================

base_data = load_dataset()


# ======================
# MODEL
# ======================

spectral = SpectralFilter(alpha=0.1).to(device)
model = FraudGAT(base_data.x.shape[1], 32, 2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


# ======================
# TRAIN
# ======================

for epoch in range(250):

    model.train()
    optimizer.zero_grad()

    # 🔥 clone data every epoch (CRITICAL FIX)
    data = copy.deepcopy(base_data)

    # ---- move to CPU for adversarial injection ----
    data = data.to("cpu")

    # ---- adversarial injection ----
    data = inject_adversarial_edges(data)

    # ---- move to GPU ----
    data = data.to(device)

    # ---- spectral filtering ----
    data.x = spectral(data.x, data.edge_index)

    # ---- forward ----
    out = model(data.x, data.edge_index, data.time_steps)

    # 🔥 use UPDATED masks
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss {loss.item()}")


# ======================
# EVALUATION
# ======================

model.eval()

with torch.no_grad():

    # clone again (same pipeline)
    data = copy.deepcopy(base_data)

    data = data.to("cpu")
    data = inject_adversarial_edges(data)
    data = data.to(device)

    data.x = spectral(data.x, data.edge_index)

    out = model(data.x, data.edge_index, data.time_steps)

    prob = torch.softmax(out, dim=1)[:, 1].cpu()
    y_true = data.y.cpu()

    mask = data.test_mask.cpu()

    # ---- dynamic threshold ----
    precision, recall, thresholds = precision_recall_curve(
        y_true[mask], prob[mask]
    )

    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = f1.argmax()

    best_threshold = thresholds[best_idx]

    print(f"\nBest threshold: {best_threshold:.3f} (F1={f1[best_idx]:.4f})")

    pred = (prob > best_threshold).long()

    print("\n--- CRYPTO FRAUD EVALUATION ---\n")
    print("Accuracy:", accuracy_score(y_true[mask], pred[mask]))
    print(classification_report(y_true[mask], pred[mask]))


# ======================
# DECISION ENGINE
# ======================

def decision(prob, threshold):

    if prob > threshold + 0.2:
        return "BLOCK"

    elif prob > threshold:
        return "OTP"

    else:
        return "ALLOW"


# ======================
# PREVENTION
# ======================

print("\n--- CRYPTO FRAUD PREVENTION ---\n")

indices = random.sample(range(len(prob)), 10)

for i in indices:
    p = prob[i].item()
    print(f"Node {i}: {p:.3f} → {decision(p, best_threshold)}")