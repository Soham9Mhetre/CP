import torch
import torch.nn.functional as F

from models.credit_temporal import CreditTemporal
from models.credit_spectral import CreditSpectral
from models.credit_gat import CreditGAT
from models.edge_pruning import prune_edges

from sklearn.neighbors import NearestNeighbors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD CHECKPOINT
# -------------------------
checkpoint = torch.load(
    "models/credit_model.pth",
    map_location=device,
    weights_only=True
)

input_dim = checkpoint["input_dim"]

# -------------------------
# BUILD MODEL
# -------------------------
spectral = CreditSpectral(alpha=0.1)
temporal = CreditTemporal(input_dim, 32).to(device)
gat = CreditGAT(32, 32).to(device)

temporal.load_state_dict(checkpoint["temporal_state_dict"])
gat.load_state_dict(checkpoint["gat_state_dict"])

temporal.eval()
gat.eval()

temperature = checkpoint.get("temperature", 1.0)

# -------------------------
# GLOBAL CACHE (IMPORTANT)
# -------------------------
_cached = {
    "edge_index": None,
    "x_f": None,
    "edge_pruned": None,
    "emb": None,
    "logits": None
}


# -------------------------
# BUILD GRAPH ONCE
# -------------------------
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


# -------------------------
# PRECOMPUTE EVERYTHING ONCE
# -------------------------
def initialize_credit_pipeline(X):
    global _cached

    X = X.to(device)

    edge_index = build_graph(X)
    x_f = spectral(X, edge_index)
    edge_pruned = prune_edges(edge_index, x_f, threshold=0.05)

    emb = temporal(x_f, None)

    x_g = gat.gat1(emb, edge_pruned)
    x_g = torch.relu(x_g)

    graph_emb = gat.gat2(x_g, edge_pruned)
    logits = gat.fc(graph_emb)

    logits = logits / temperature
    probs = F.softmax(logits, dim=1)

    _cached["probs"] = probs


# -------------------------
# PREDICT (FAST)
# -------------------------
def predict_credit(X, node_idx):
    if _cached.get("probs") is None:
        initialize_credit_pipeline(X)

    probs = _cached["probs"]

    fraud_prob = probs[node_idx, 1].item()
    uncertainty = 1 - fraud_prob

    return {
        "fraud_prob": fraud_prob,
        "uncertainty": uncertainty
    }