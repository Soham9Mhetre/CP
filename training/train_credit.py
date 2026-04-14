import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve
from sklearn.neighbors import NearestNeighbors
import random

from data.credit_card_loader import load_credit_card_data
from models.credit_temporal import CreditTemporal
from models.credit_spectral import CreditSpectral
from models.credit_gat import CreditGAT

from models.edge_pruning import prune_edges
from models.drift_detection import detect_drift

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

for epoch in range(100):

    temporal.train()
    gat.train()

    optimizer.zero_grad()

    x_f = spectral(X, edge_index)

    edge_pruned = prune_edges(edge_index, x_f, threshold=0.05)

    drift_score = detect_drift(x_f)

    temporal_emb = temporal(x_f, None)

    x_g = gat.gat1(temporal_emb, edge_pruned)
    x_g = F.elu(x_g)
    graph_emb = gat.gat2(x_g, edge_pruned)

    out = gat.fc(graph_emb)

    loss_contrast = torch.mean((graph_emb - temporal_emb) ** 2)

    loss = F.cross_entropy(out, y, weight=class_weights) + 0.05 * loss_contrast

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

    edge_pruned = prune_edges(edge_index, x_f, threshold=0.05)

    temporal_emb = temporal(x_f, None)

    x_g = gat.gat1(temporal_emb, edge_pruned)
    x_g = F.elu(x_g)
    graph_emb = gat.gat2(x_g, edge_pruned)

    out = gat.fc(graph_emb)

    prob = torch.softmax(out, dim=1)[:, 1].cpu()
    y_true = y.cpu()

    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds[f1.argmax()]

    pred = (prob > best_threshold).long()

    print("\n--- CREDIT FRAUD EVALUATION ---\n")
    print("Accuracy:", accuracy_score(y_true, pred))
    print(classification_report(y_true, pred))


# ======================
# PREVENTION
# ======================

def decision(prob, uncertainty, threshold):

    high = threshold + 0.15

    if uncertainty > 0.6:
        return "SEND TO ANALYST"
    elif prob > high:
        return "BLOCK"
    elif prob > threshold:
        return "OTP"
    else:
        return "ALLOW"


confidence = prob
uncertainty = 1 - confidence

print("\n--- CREDIT FRAUD PREVENTION ---\n")

indices = random.sample(range(len(prob)), 10)

for i in indices:
    p = prob[i].item()
    u = uncertainty[i].item()

    print(f"Txn {i}: prob={p:.3f}, unc={u:.3f} → {decision(p, u, best_threshold)}")