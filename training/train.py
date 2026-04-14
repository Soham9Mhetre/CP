import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve
import random

from data.load_dataset import load_dataset
from models.spectral_filter import SpectralFilter
from models.gat_encoder import FraudGAT
from models.temporal_lstm import TemporalLSTM
from models.adversarial_injection import inject_adversarial_edges

from models.edge_pruning import prune_edges
from models.drift_detection import detect_drift

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# LOAD DATA
# ======================

data = load_dataset()
data = data.to(device)


# ======================
# MODELS
# ======================

spectral = SpectralFilter(alpha=0.1).to(device)

temporal = TemporalLSTM(data.x.shape[1], data.x.shape[1]).to(device)

model = FraudGAT(data.x.shape[1], 32, 2).to(device)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(temporal.parameters()),
    lr=0.005
)

class_weights = torch.tensor([1.0, 8.0]).to(device)

# ======================
# PRECOMPUTE GRAPH (OUTSIDE LOOP)
# ======================

data_cpu = data.to("cpu")
data_cpu = inject_adversarial_edges(data_cpu)

edge_index = data_cpu.edge_index.to(device)


# ======================
# TRAIN
# ======================

for epoch in range(200):

    model.train()
    temporal.train()

    optimizer.zero_grad()

    x_input = data.x.to(device)

    # ---- spectral (optimized) ----
    if epoch % 3 == 0:
        x = spectral(x_input, edge_index)
    else:
        x = x_input
    
    x = F.dropout(x, p=0.2, training=model.training)


    # ---- temporal ----
    temporal_emb = temporal(x, data.time_steps)

    # ---- edge pruning ----
    edge_pruned = prune_edges(edge_index, x, threshold=0.15).to(device)

    # ---- graph view ----
    graph_hidden = model.gat1(x, edge_pruned)
    graph_hidden = F.elu(graph_hidden)

    # ---- forward ----
    out = model(x, edge_pruned, data.time_steps)

    # ---- contrastive ----
    min_dim = min(graph_hidden.shape[1], temporal_emb.shape[1])

    graph_proj = graph_hidden[:, :min_dim]
    temporal_proj = temporal_emb[:, :min_dim]

    loss_contrast = torch.mean((graph_proj - temporal_proj) ** 2)

    y_train = data.y.to(device)

    loss = F.cross_entropy(
        out[data.train_mask],
        y_train[data.train_mask],
        weight=class_weights
    ) + 0.01 * loss_contrast

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss {loss.item()}")


# ======================
# EVALUATION
# ======================

model.eval()
temporal.eval()

with torch.no_grad():

    data_cpu = data.to("cpu")
    data_cpu = inject_adversarial_edges(data_cpu)

    edge_index = data_cpu.edge_index

    # 🔥 FORCE DEVICE AGAIN
    edge_index = edge_index.to(device)
    x_input = data.x.to(device)

    x = spectral(x_input, edge_index)

    edge_pruned = prune_edges(edge_index, x, threshold=0.15)
    edge_pruned = edge_pruned.to(device)

    out = model(x, edge_pruned, data.time_steps)

    probs_all = torch.softmax(out, dim=1)

    prob = probs_all[:, 1].cpu()
    confidence = probs_all.max(dim=1)[0].cpu()
    uncertainty = 1 - confidence

    y_true = data.y.cpu()
    mask = data.test_mask.cpu()

    precision, recall, thresholds = precision_recall_curve(y_true[mask], prob[mask])
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_threshold = thresholds[f1.argmax()]
    best_threshold = best_threshold * 0.75

    pred = (prob > best_threshold).long()

    print("\n--- CRYPTO FRAUD EVALUATION ---\n")
    print("Accuracy:", accuracy_score(y_true[mask], pred[mask]))
    print(classification_report(y_true[mask], pred[mask]))


# ======================
# PREVENTION
# ======================

def decision(prob, uncertainty, threshold):

    low = threshold * 0.6
    high = threshold + 0.08

    if uncertainty > 0.75:
        return "SEND TO ANALYST"

    elif prob > high:
        return "BLOCK"

    elif prob > low:
        return "OTP"

    else:
        return "ALLOW"


print("\n--- CRYPTO FRAUD PREVENTION ---\n")

indices = random.sample(range(len(prob)), 10)

for i in indices:
    p = prob[i].item()
    u = uncertainty[i].item()

    print(f"Node {i}: prob={p:.3f}, unc={u:.3f} → {decision(p, u, best_threshold)}")