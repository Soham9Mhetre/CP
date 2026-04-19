import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from data.credit_card_loader import load_credit_card_data
from models.credit_temporal import CreditTemporal
from models.credit_spectral import CreditSpectral
from models.credit_gat import CreditGAT
from models.edge_pruning import prune_edges

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD DATA
# -------------------------
X, y = load_credit_card_data("data/banksim.csv")

X = X.to(device)
y = y.to(device)

# -------------------------
# SPLIT
# -------------------------
idx = torch.arange(len(y))
train_idx, temp_idx = train_test_split(idx.cpu(), test_size=0.3)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5)

train_idx = torch.tensor(train_idx).to(device)
val_idx = torch.tensor(val_idx).to(device)
test_idx = torch.tensor(test_idx).to(device)

# -------------------------
# MODEL
# -------------------------
input_dim = X.shape[1]

spectral = CreditSpectral(alpha=0.1)
temporal = CreditTemporal(input_dim, 32).to(device)
gat = CreditGAT(32, 32).to(device)

optimizer = torch.optim.Adam(
    list(temporal.parameters()) + list(gat.parameters()),
    lr=0.005
)

# -------------------------
# BUILD GRAPH
# -------------------------
from sklearn.neighbors import NearestNeighbors

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

# -------------------------
# TRAIN
# -------------------------
for epoch in range(200):
    temporal.train()
    gat.train()

    x_f = spectral(X, edge_index)
    edge_pruned = prune_edges(edge_index, x_f, threshold=0.05)

    emb = temporal(x_f, None)

    x_g = gat.gat1(emb, edge_pruned)
    x_g = torch.relu(x_g)

    graph_emb = gat.gat2(x_g, edge_pruned)
    logits = gat.fc(graph_emb)

    loss = F.cross_entropy(logits[train_idx], y[train_idx])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss {loss.item()}")

# -------------------------
# EVAL
# -------------------------
temporal.eval()
gat.eval()

with torch.no_grad():
    x_f = spectral(X, edge_index)
    edge_pruned = prune_edges(edge_index, x_f, threshold=0.05)

    emb = temporal(x_f, None)
    x_g = gat.gat1(emb, edge_pruned)
    x_g = torch.relu(x_g)
    graph_emb = gat.gat2(x_g, edge_pruned)

    logits = gat.fc(graph_emb)
    preds = logits.argmax(dim=1)

    print("\n--- CREDIT EVALUATION ---\n")
    print("Accuracy:", accuracy_score(y[test_idx].cpu(), preds[test_idx].cpu()))
    print(classification_report(y[test_idx].cpu(), preds[test_idx].cpu()))

# -------------------------
# CALIBRATION
# -------------------------
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


temperature = calibrate_temperature(logits[val_idx], y[val_idx])

print(f"\nLearned temperature: {temperature}")

# -------------------------
# SAVE
# -------------------------
torch.save({
    "temporal_state_dict": temporal.state_dict(),
    "gat_state_dict": gat.state_dict(),
    "input_dim": input_dim,
    "threshold": 0.5,
    "temperature": float(temperature)
}, "models/credit_model.pth")

print("\nCredit model saved.")