import torch
import torch.nn.functional as F
import os

from models.gat_encoder import FraudGAT
from models.temporal_lstm import TemporalLSTM
from models.spectral_filter import SpectralFilter

from models.credit_gat import CreditGAT
from models.credit_temporal import CreditTemporal
from models.credit_spectral import CreditSpectral

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# LOAD MODELS
# ======================

crypto_model = None
crypto_threshold = 0.5

credit_temporal = None
credit_gat = None
credit_threshold = 0.5


def load_models():
    global crypto_model, crypto_threshold
    global credit_temporal, credit_gat, credit_threshold

    # ---- Crypto ----
    crypto_path = "models/crypto_model.pth"
    if os.path.exists(crypto_path):
        checkpoint = torch.load(crypto_path, map_location=device)

        crypto_model = FraudGAT(input_dim=9, hidden_dim=32, output_dim=2).to(device)
        crypto_model.load_state_dict(checkpoint["model"])
        crypto_model.eval()

        crypto_threshold = checkpoint["threshold"]

        print("✅ Crypto model loaded")

    # ---- Credit ----
    credit_path = "models/credit_model.pth"
    if os.path.exists(credit_path):
        checkpoint = torch.load(credit_path, map_location=device)

        credit_temporal = CreditTemporal(input_dim=9, hidden_dim=32).to(device)
        credit_gat = CreditGAT(input_dim=32, hidden_dim=32).to(device)

        credit_temporal.load_state_dict(checkpoint["temporal"])
        credit_gat.load_state_dict(checkpoint["gat"])

        credit_temporal.eval()
        credit_gat.eval()

        credit_threshold = checkpoint["threshold"]

        print("✅ Credit model loaded")


# Load once
load_models()


# ======================
# HELPER: BUILD SMALL GRAPH
# ======================

def build_small_graph(x):
    # x: [1, F]
    x = x.repeat(5, 1)  # create 5 nodes

    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 1, 2, 3],
        [1, 2, 3, 4, 0, 0, 1, 2]
    ], dtype=torch.long)

    return x, edge_index


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
# CRYPTO PREDICTION
# ======================

def predict_transaction(features: dict):

    if crypto_model is None:
        return {"error": "Crypto model not loaded"}

    x = torch.tensor([
        features["amount"],
        features["hour"],
        features["day_of_week"],
        features["distance_from_last"],
        features["merchant_risk"],
        features["velocity_1h"],
        features["velocity_24h"],
        features["country_risk"],
        features["device_risk"]
    ], dtype=torch.float).unsqueeze(0).to(device)

    x, edge_index = build_small_graph(x)
    x, edge_index = x.to(device), edge_index.to(device)

    time_steps = torch.zeros(x.shape[0]).long().to(device)

    out = crypto_model(x, edge_index, time_steps)
    prob = torch.softmax(out, dim=1)[:, 1][0].item()

    return {
        "fraud_probability": round(prob, 4),
        "decision": decision(prob, crypto_threshold),
        "threshold": crypto_threshold
    }


# ======================
# CREDIT PREDICTION
# ======================

def predict_credit_transaction(features: dict):

    if credit_temporal is None:
        return {"error": "Credit model not loaded"}

    x = torch.tensor([
        features["amount"],
        features["hour"],
        features["day_of_week"],
        features["distance_from_last"],
        features["merchant_risk"],
        features["velocity_1h"],
        features["velocity_24h"],
        features["country_risk"],
        features["device_risk"]
    ], dtype=torch.float).unsqueeze(0).to(device)

    x, edge_index = build_small_graph(x)
    x, edge_index = x.to(device), edge_index.to(device)

    x_temp = credit_temporal(x, None)
    out = credit_gat(x_temp, edge_index)

    prob = torch.softmax(out, dim=1)[:, 1][0].item()

    return {
        "fraud_probability": round(prob, 4),
        "decision": decision(prob, credit_threshold),
        "threshold": credit_threshold
    }