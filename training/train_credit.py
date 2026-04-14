import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, average_precision_score, roc_auc_score
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

X = X[:50000]
y = y[:50000]

X = X.to(device)
y = y.to(device)


# ======================
# BUILD GRAPH
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

    return torch.tensor(edges).t().contiguous().to(device)


edge_index = build_graph(X)


# ======================
# MODELS
# ======================

spectral = CreditSpectral(alpha=0.1)
temporal = CreditTemporal(input_dim=X.shape[1], hidden_dim=32).to(device)
gat = CreditGAT(input_dim=32, hidden_dim=32).to(device)

optimizer = torch.optim.Adam(
    list(temporal.parameters()) + list(gat.parameters()),
    lr=0.003
)


# ======================
# CLASS WEIGHTS
# ======================

class_weights = torch.tensor([1.0, 10.0]).to(device)


# ======================
# TRAIN
# ======================

for epoch in range(100):

    temporal.train()
    gat.train()

    optimizer.zero_grad()

    # Spectral
    x_filtered = spectral(X, edge_index)

    # Adversarial
    edge_adv = inject_adversarial_edges(edge_index, X.shape[0], ratio=0.01)

    # Temporal
    x_temp = temporal(x_filtered, None)

    # GAT
    out = gat(x_temp, edge_adv)

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

    x_filtered = spectral(X, edge_index)
    edge_adv = inject_adversarial_edges(edge_index, X.shape[0], ratio=0.01)

    x_temp = temporal(x_filtered, None)

    out = gat(x_temp, edge_adv)

    prob = torch.softmax(out, dim=1)[:, 1]
    pred = (prob > 0.5).long()

    print("\n--- CREDIT CARD FRAUD EVALUATION ---\n")

    print("Accuracy:", accuracy_score(y.cpu(), pred.cpu()))
    print("AUC-ROC:", roc_auc_score(y.cpu(), prob.cpu()))
    print("AUC-PR:", average_precision_score(y.cpu(), prob.cpu()))

    print(classification_report(y.cpu(), pred.cpu()))


# ======================
# PREVENTION
# ======================

def decision(prob):

    if prob > 0.8:
        return "BLOCK TRANSACTION"

    elif prob > 0.5:
        return "REQUIRE OTP"

    else:
        return "ALLOW"


print("\n--- FRAUD PREVENTION ---\n")

indices = random.sample(range(len(prob)), 10)

for i in indices:
    p = prob[i].item()
    label = "FRAUD" if y[i] == 1 else "LEGIT"
    print(f"Txn {i} [{label}] score={p:.3f} → {decision(p)}")