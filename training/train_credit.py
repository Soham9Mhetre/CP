import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve
from sklearn.neighbors import NearestNeighbors
import random

from data.credit_card_loader import load_credit_card_data
from models.credit_temporal import CreditTemporal
from models.credit_spectral import CreditSpectral
from models.credit_adversarial import inject_adversarial_edges
from models.credit_gat import CreditGAT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# LOAD DATA
# ======================

X, y = load_credit_card_data("data/banksim.csv")

X = X[:50000].to(device)
y = y[:50000].to(device)


# ======================
# GRAPH
# ======================

def build_graph(x):

    X_np = x.cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=5).fit(X_np)
    _, indices = nbrs.kneighbors(X_np)

    edges = []

    for i in range(len(indices)):
        for j in indices[i]:
            if i != j:
                edges.append([i, j])

    return torch.tensor(edges).t().to(device)


edge_index = build_graph(X)


# ======================
# MODELS
# ======================

spectral = CreditSpectral(alpha=0.1)
temporal = CreditTemporal(X.shape[1], 32).to(device)
gat = CreditGAT(32, 32).to(device)

optimizer = torch.optim.Adam(
    list(temporal.parameters()) + list(gat.parameters()),
    lr=0.003
)

class_weights = torch.tensor([1.0, 10.0]).to(device)


# ======================
# TRAIN
# ======================

for epoch in range(250):

    temporal.train()
    gat.train()

    optimizer.zero_grad()

    x_f = spectral(X, edge_index)
    edge_adv = inject_adversarial_edges(edge_index, X.shape[0], 0.01)

    x_t = temporal(x_f, None)
    out = gat(x_t, edge_adv)

    loss = F.cross_entropy(out, y, weight=class_weights)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss {loss.item()}")


# ======================
# EVALUATION
# ======================

temporal.eval()
gat.eval()

with torch.no_grad():

    x_f = spectral(X, edge_index)
    edge_adv = inject_adversarial_edges(edge_index, X.shape[0], 0.01)

    x_t = temporal(x_f, None)
    out = gat(x_t, edge_adv)

    prob = torch.softmax(out, dim=1)[:, 1].cpu()
    y_true = y.cpu()

    precision, recall, thresholds = precision_recall_curve(y_true, prob)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = f1.argmax()

    best_threshold = thresholds[best_idx]

    print(f"\nBest threshold: {best_threshold:.3f} (F1={f1[best_idx]:.4f})")

    pred = (prob > best_threshold).long()

    print("\n--- CREDIT FRAUD EVALUATION ---\n")
    print("Accuracy:", accuracy_score(y_true, pred))
    print(classification_report(y_true, pred))


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

print("\n--- FRAUD PREVENTION ---\n")

indices = random.sample(range(len(prob)), 10)

for i in indices:
    p = prob[i].item()
    label = "FRAUD" if y[i] == 1 else "LEGIT"
    print(f"Txn {i} [{label}] score={p:.3f} → {decision(p, best_threshold)}")