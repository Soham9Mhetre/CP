"""
Flask REST API for the Crypto Fraud Prevention System (Elliptic Bitcoin Dataset).
Serves the web UI and exposes GNN-backed inference endpoints.
"""

import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UI_DIR   = os.path.join(ROOT_DIR, "ui")

app = Flask(__name__, static_folder=UI_DIR, static_url_path="")
CORS(app)

from app.inference import (
    predict_transaction,
    get_stats,
    generate_sample_transactions,
    get_fraud_timeline,
    get_risk_distribution,
    OTP_THRESHOLD,
    BLOCK_THRESHOLD,
    # Credit card module
    predict_credit_transaction,
    get_credit_stats,
    generate_credit_sample_transactions,
    get_credit_risk_distribution,
)

# ── Elliptic entity type labels ───────────────────────────────────────────────
ENTITY_TYPES = [
    "Exchange Wallet", "Mining Pool", "Personal Wallet", "DeFi Protocol",
    "NFT Platform", "ICO Wallet", "Payment Processor", "Unknown Entity",
    "Darknet Market", "Mixing Service", "Ransomware Address", "Gambling Service",
]


# ── Serve UI ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(UI_DIR, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(UI_DIR, filename)


# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    from app.inference import _MODELS_LOADED, _TORCH_AVAILABLE
    return jsonify({
        "status":          "healthy",
        "dataset":         "Elliptic Bitcoin Transaction Dataset",
        "torch_available": _TORCH_AVAILABLE,
        "models_loaded":   _MODELS_LOADED,
        "mode":            "simulation",
        "otp_threshold":   OTP_THRESHOLD,
        "block_threshold": BLOCK_THRESHOLD,
    })


# ── Stats ─────────────────────────────────────────────────────────────────────
@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(get_stats())


# ── Single Transaction Prevention Check ───────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Body (JSON):
    {
        "time_step":    25,     // 1–49 Elliptic temporal snapshot
        "in_degree":    3,      // incoming transaction edges
        "out_degree":   5,      // outgoing transaction edges
        "btc_vol_log":  6.2,    // log-scaled BTC volume
        "is_exchange":  0,      // 1 = connected to known exchange
        "is_service":   0,      // 1 = connected to known service
        "suspicious":   0,      // 1 = connects to flagged address
        "local_edges":  12,     // edges in 1-hop neighborhood
        "lifetime":     8       // time steps the address has been active
    }
    """
    data = request.get_json(force=True) or {}

    # Build feature vector matching inference.py order
    features = [
        float(data.get("time_step",   25)),
        float(data.get("in_degree",    3)),
        float(data.get("out_degree",   3)),
        float(data.get("btc_vol_log",  5.0)),
        float(data.get("is_exchange",  0)),
        float(data.get("is_service",   0)),
        float(data.get("suspicious",   0)),
        float(data.get("local_edges", 10)),
        float(data.get("lifetime",    10)),
    ]

    result = predict_transaction(features)
    prob   = result["fraud_probability"]
    dec    = result["decision"]

    if dec == "BLOCK":
        action_text = "🚫 Block Transaction"
        risk_level  = "HIGH"
        color       = "#ff3864"
    elif dec == "OTP":
        action_text = "⚠️  Require OTP Verification"
        risk_level  = "MEDIUM"
        color       = "#f7b731"
    else:
        action_text = "✅ Allow Transaction"
        risk_level  = "LOW"
        color       = "#00f5d4"

    return jsonify({
        **result,
        "action_text":     action_text,
        "risk_level":      risk_level,
        "color":           color,
        "input_features":  features,
        "otp_threshold":   OTP_THRESHOLD,
        "block_threshold": BLOCK_THRESHOLD,
    })


# ── Sample Transaction Feed ───────────────────────────────────────────────────
@app.route("/api/sample-transactions", methods=["GET"])
def sample_transactions():
    n = min(int(request.args.get("n", 20)), 50)
    return jsonify(generate_sample_transactions(n))


# ── Batch Prevention Check ────────────────────────────────────────────────────
@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    data  = request.get_json(force=True) or {}
    txns  = data.get("transactions", [])
    results = []
    for t in txns:
        features = [
            float(t.get("time_step", 25)),
            float(t.get("in_degree",  3)),
            float(t.get("out_degree", 3)),
            float(t.get("btc_vol_log", 5.0)),
            float(t.get("is_exchange", 0)),
            float(t.get("is_service",  0)),
            float(t.get("suspicious",  0)),
            float(t.get("local_edges", 10)),
            float(t.get("lifetime",    10)),
        ]
        results.append(predict_transaction(features))
    return jsonify({"results": results, "count": len(results)})


# ── Fraud Timeline ────────────────────────────────────────────────────────────
@app.route("/api/timeline", methods=["GET"])
def timeline():
    return jsonify(get_fraud_timeline())


# ── Risk Zone Distribution ────────────────────────────────────────────────────
@app.route("/api/risk-distribution", methods=["GET"])
def risk_distribution():
    return jsonify(get_risk_distribution())


# ── Entity types (for the UI form dropdown) ──────────────────────────────────
@app.route("/api/entity-types", methods=["GET"])
def entity_types():
    return jsonify(ENTITY_TYPES)


# ════════════════════════════════════════════════════════════════════════════════
# CREDIT CARD MODULE ENDPOINTS  (/api/credit/*)
# ════════════════════════════════════════════════════════════════════════════════

@app.route("/api/credit/stats", methods=["GET"])
def credit_stats():
    return jsonify(get_credit_stats())


@app.route("/api/credit/predict", methods=["POST"])
def credit_predict():
    """
    Body (JSON):
    {
        "step":         50,    // BankSim time step (1–180)
        "amount":       250,   // transaction amount in €
        "category":     2,     // merchant category index (0–11)
        "age_norm":     0.4,   // customer age normalised (0=young, 1=old)
        "is_intl":      0,     // 1 = international transaction
        "hour":         14,    // hour of day (0–23)
        "txns_24h":     3,     // number of customer transactions in last 24h
        "avg_amt_7d":   200,   // customer avg spend over past 7 days
        "distance_km":  15     // km from customer home address
    }
    """
    data = request.get_json(force=True) or {}
    features = [
        float(data.get("step",       50)),
        float(data.get("amount",    250)),
        float(data.get("category",    2)),
        float(data.get("age_norm",  0.4)),
        float(data.get("is_intl",     0)),
        float(data.get("hour",       14)),
        float(data.get("txns_24h",    3)),
        float(data.get("avg_amt_7d", 200)),
        float(data.get("distance_km", 15)),
    ]

    result = predict_credit_transaction(features)
    prob   = result["fraud_probability"]
    dec    = result["decision"]

    if dec == "BLOCK":
        action_text = "🚫 Block Transaction"
        risk_level  = "HIGH"
        color       = "#ff3864"
    elif dec == "OTP":
        action_text = "⚠️  Require OTP Verification"
        risk_level  = "MEDIUM"
        color       = "#f7b731"
    else:
        action_text = "✅ Allow Transaction"
        risk_level  = "LOW"
        color       = "#00f5d4"

    return jsonify({
        **result,
        "action_text":     action_text,
        "risk_level":      risk_level,
        "color":           color,
        "input_features":  features,
        "otp_threshold":   0.5,
        "block_threshold": 0.8,
    })


@app.route("/api/credit/sample-transactions", methods=["GET"])
def credit_sample_transactions():
    n = min(int(request.args.get("n", 18)), 50)
    return jsonify(generate_credit_sample_transactions(n))


@app.route("/api/credit/risk-distribution", methods=["GET"])
def credit_risk_distribution():
    return jsonify(get_credit_risk_distribution())
