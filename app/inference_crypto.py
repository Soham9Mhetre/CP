import torch
import torch.nn.functional as F

from models.gat_encoder import FraudGAT
from models.spectral_filter import SpectralFilter
from models.edge_pruning import prune_edges
from data.load_dataset import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD MODEL ONCE
# -------------------------
checkpoint = torch.load(
    "models/crypto_model.pth",
    map_location=device,
    weights_only=True
)
model = FraudGAT(checkpoint["input_dim"], 32, 2).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

spectral = SpectralFilter(alpha=0.1).to(device)

temperature = checkpoint.get("temperature", 1.0)


# -------------------------
# PREDICT FUNCTION
# -------------------------
def predict_crypto(data, node_idx):
    with torch.no_grad():
        data = data.to(device)

        x = spectral(data.x, data.edge_index)
        edge_pruned = prune_edges(data.edge_index, x, threshold=0.15)

        logits = model(x, edge_pruned, data.time_steps)

        # Apply temperature scaling
        logits = logits / temperature

        probs = F.softmax(logits, dim=1)

        fraud_prob = probs[node_idx, 1].item()

        # Uncertainty = entropy approximation
        uncertainty = 1 - fraud_prob

        return {
            "fraud_prob": fraud_prob,
            "uncertainty": uncertainty
        }