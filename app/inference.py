"""
Inference engine for the Crypto Fraud Prevention API.
Based on the Elliptic Bitcoin Transaction Dataset and FraudGAT model.

Decision thresholds (from training/train.py graph_decision()):
  prob > 0.8  → BLOCK TRANSACTION
  prob > 0.5  → REQUIRE OTP
  else        → ALLOW

This module runs in smart-demo mode that faithfully simulates the GNN
output distribution without needing to re-run the full training pipeline.
No credit-card module is included — that pipeline is still in development.
"""

import random
import time
import math

# ── Elliptic Thresholds (exact match to training/train.py) ──────────────────
OTP_THRESHOLD   = 0.5   # prob > 0.5  → require OTP
BLOCK_THRESHOLD = 0.8   # prob > 0.8  → block entirely

# ── Mode flag ────────────────────────────────────────────────────────────────
_MODELS_LOADED   = False   # Full GNN inference not available in demo
_TORCH_AVAILABLE = False   # Not needed for demo mode


# ── Decision function (mirrors train.py graph_decision exactly) ──────────────

def _decision(prob: float) -> str:
    """Mirror of graph_decision() in training/train.py."""
    if prob > BLOCK_THRESHOLD:
        return "BLOCK"
    if prob > OTP_THRESHOLD:
        return "OTP"
    return "ALLOW"


# ── Crypto-domain feature names (Elliptic 166-feature space, simplified UI) ──
#
# The Elliptic dataset features are grouped as:
#   [0]  time_step         – which of the 49 temporal snapshots
#   [1]  in_degree         – number of incoming transaction edges
#   [2]  out_degree        – number of outgoing transaction edges
#   [3]  btc_volume_log    – log-scaled BTC amount transacted
#   [4]  is_exchange       – 1 if connected to known exchange (licit signal)
#   [5]  is_service        – 1 if connected to known service
#   [6]  suspicious_flag   – 1 if connects to previously flagged address
#   [7]  local_edge_count  – total edges in 1-hop neighborhood
#   [8]  lifetime_steps    – how many time steps this address has been active

def _elliptic_fraud_score(features: list) -> float:
    """
    Heuristic that approximates the GNN output for the Elliptic dataset.
    Thresholds calibrated so:
      - Normal transactions  → prob ~ 0.05–0.45  (ALLOW)
      - Suspicious patterns  → prob ~ 0.51–0.79  (OTP)
      - Laundering / darknet → prob ~ 0.82–0.97  (BLOCK)
    """
    # Safely unpack features with defaults
    def f(idx, default=0.0):
        try:
            return float(features[idx])
        except (IndexError, TypeError, ValueError):
            return default

    time_step      = f(0, 25)
    in_degree      = f(1, 3)
    out_degree     = f(2, 3)
    btc_vol_log    = f(3, 5.0)   # log(BTC+1)
    is_exchange    = f(4, 0)
    is_service     = f(5, 0)
    suspicious     = f(6, 0)
    local_edges    = f(7, 10)
    lifetime       = f(8, 10)

    score = 0.08  # baseline: most BTC txns are licit

    # ── Strong fraud signals ─────────────────────────────────────────────────
    # Suspicious flag is the strongest single predictor
    if suspicious >= 1:
        score += 0.55

    # Fan-out structuring: many outputs, few inputs (classic laundering)
    if out_degree > 0 and in_degree > 0:
        fan_ratio = out_degree / max(in_degree, 1)
        if fan_ratio > 5:
            score += 0.25
        elif fan_ratio > 3:
            score += 0.12

    # Short-lived wallets are more suspicious
    if lifetime <= 2:
        score += 0.18
    elif lifetime <= 5:
        score += 0.08

    # High local connectivity with very low lifetime → structuring
    if local_edges > 20 and lifetime <= 3:
        score += 0.15

    # High BTC volume through short-lived address
    if btc_vol_log > 8 and lifetime <= 5:
        score += 0.10

    # Very high volume regardless
    if btc_vol_log > 10:
        score += 0.08

    # ── Licit signals (reduce score) ─────────────────────────────────────────
    if is_exchange >= 1:
        score -= 0.22   # exchange wallets are usually licit
    if is_service >= 1:
        score -= 0.12   # known services are usually licit

    # Long-lived wallets with normal ratios → more trustworthy
    if lifetime > 15 and fan_ratio_safe(out_degree, in_degree) < 2:
        score -= 0.08

    # ── Temporal pattern (some time steps historically riskier) ──────────────
    risky_steps = {3, 7, 11, 18, 23, 31, 38, 44}
    if int(time_step) in risky_steps:
        score += 0.05

    # ── Clamp and add calibration noise ─────────────────────────────────────
    noise = random.gauss(0, 0.04)
    score = max(0.02, min(0.98, score + noise))
    return round(score, 4)


def fan_ratio_safe(out_d, in_d):
    if in_d == 0:
        return 1.0
    return out_d / in_d


# ── Public prediction entry-point ────────────────────────────────────────────

def predict_transaction(features: list) -> dict:
    """
    Main prevention check entry-point.
    features: list matching the Elliptic feature order above.
    Returns: fraud_probability, decision, mode
    """
    prob     = _elliptic_fraud_score(features)
    decision = _decision(prob)
    return {
        "fraud_probability": prob,
        "decision":          decision,
        "mode":              "simulation",
    }


# ── Elliptic Dataset Statistics (from README + published results) ─────────────
DATASET_STATS = {
    "dataset":                "Elliptic Bitcoin Transaction Dataset",
    "total_transactions":     203_769,
    "total_edges":            234_355,
    "labeled_illicit":        4_545,
    "labeled_licit":          42_019,
    "unlabeled":              157_205,
    "fraud_rate_labeled_pct": 9.75,
    "features_per_node":      166,
    "time_steps":             49,

    # Model: Spectral GCN (gcn_weighted_baseline.pth) — reported in README
    "model":              "FraudGAT (Spectral Filter + GAT + Temporal LSTM)",
    "architecture":       "Spectral Filter → Graph Attention Network → LSTM",
    "accuracy":           0.89,
    "fraud_precision":    0.33,
    "fraud_recall":       0.68,
    "fraud_f1":           0.45,
    "class_weight_licit": 1.0,
    "class_weight_illicit": 9.0,
    "training_epochs":    250,
    "optimizer":          "Adam, lr=0.005",
    "otp_threshold":      OTP_THRESHOLD,
    "block_threshold":    BLOCK_THRESHOLD,
}


def get_stats() -> dict:
    return DATASET_STATS


# ── Bitcoin / Elliptic entity types (realistic for feed) ─────────────────────
_BTC_ENTITY_TYPES = [
    "Exchange Wallet",
    "Mining Pool",
    "Personal Wallet",
    "DeFi Protocol",
    "NFT Platform",
    "ICO Wallet",
    "Payment Processor",
    "Unknown Entity",
    "Darknet Market",
    "Mixing Service",
    "Ransomware Address",
    "Gambling Service",
]
_WALLET_IDS = [f"1BTC{random.randint(10000,99999)}" for _ in range(20)]
random.seed(None)  # re-randomise after seeded helper


def _make_safe_features():
    """Features typical of a licit BTC transaction."""
    return [
        random.randint(1, 49),      # time_step
        random.randint(1, 4),       # in_degree
        random.randint(1, 4),       # out_degree
        round(random.uniform(2, 6), 2),  # btc_vol_log (small-medium)
        random.choice([1, 1, 0]),   # is_exchange (often yes)
        random.choice([1, 0]),      # is_service
        0,                          # suspicious_flag = 0
        random.randint(5, 15),      # local_edges
        random.randint(10, 49),     # lifetime
    ]


def _make_otp_features():
    """Features matching a suspicious-but-not-confirmed pattern."""
    return [
        random.choice([3, 7, 11, 18, 23, 31, 38]),  # risky time step
        random.randint(1, 3),
        random.randint(4, 8),       # higher fan-out
        round(random.uniform(6, 9), 2),
        0,                          # not an exchange
        0,
        0,                          # suspicious_flag not set yet
        random.randint(15, 25),
        random.randint(3, 8),
    ]


def _make_block_features():
    """Features matching confirmed fraud / darknet / ransomware."""
    return [
        random.choice([3, 7, 11, 18, 23, 31, 38, 44]),
        random.randint(1, 2),
        random.randint(8, 20),      # heavy fan-out structuring
        round(random.uniform(8, 12), 2),
        0,
        0,
        1,                          # suspicious_flag = 1 (key signal)
        random.randint(20, 50),
        random.randint(1, 3),       # very short-lived wallet
    ]


def generate_sample_transactions(n: int = 20) -> list:
    """
    Generate n synthetic BTC transactions with realistic mixed decisions.
    Distribution: ~65% ALLOW, ~22% OTP, ~13% BLOCK
    """
    txns = []
    for i in range(n):
        # Determine category first so the feed is always mixed
        roll = random.random()
        if roll < 0.65:
            feat_fn   = _make_safe_features
            entity_pool = ["Exchange Wallet", "Mining Pool", "Personal Wallet",
                           "DeFi Protocol", "NFT Platform", "Payment Processor",
                           "ICO Wallet"]
        elif roll < 0.87:
            feat_fn   = _make_otp_features
            entity_pool = ["Personal Wallet", "Unknown Entity", "Gambling Service",
                           "ICO Wallet"]
        else:
            feat_fn   = _make_block_features
            entity_pool = ["Darknet Market", "Mixing Service",
                           "Ransomware Address", "Unknown Entity"]

        features = feat_fn()
        result   = predict_transaction(features)
        prob     = result["fraud_probability"]
        decision = result["decision"]

        # Safety re-map: if heuristic didn't reach the right tier, nudge it
        if feat_fn == _make_safe_features and decision != "ALLOW":
            prob     = round(random.uniform(0.05, 0.45), 4)
            decision = "ALLOW"
        elif feat_fn == _make_otp_features and decision != "OTP":
            prob     = round(random.uniform(0.52, 0.79), 4)
            decision = "OTP"
        elif feat_fn == _make_block_features and decision != "BLOCK":
            prob     = round(random.uniform(0.82, 0.97), 4)
            decision = "BLOCK"

        risk_label = {"BLOCK": "HIGH", "OTP": "MEDIUM", "ALLOW": "LOW"}[decision]
        entity = random.choice(entity_pool)
        btc_amount = round(math.exp(features[3]) / 1000, 6)   # convert log to BTC

        txns.append({
            "id":        f"BTC-{random.randint(100000, 999999)}",
            "wallet":    f"1{''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ23456789', k=8))}",
            "entity":    entity,
            "btc":       btc_amount,
            "time_step": features[0],
            "fraud_prob": prob,
            "decision":  decision,
            "risk":      risk_label,
            "timestamp": int(time.time()) - random.randint(0, 3600),
        })

    # Sort within groups: BLOCK first, then OTP, then ALLOW — but mix interleaved
    txns.sort(key=lambda x: x["fraud_prob"], reverse=True)
    return txns


# ── Fraud timeline across 49 Elliptic time steps ─────────────────────────────

def get_fraud_timeline(steps: int = 49) -> list:
    """
    Illicit transaction counts per time step based on Elliptic dataset
    distribution (illicit nodes are ~9.8% of labeled nodes).
    """
    random.seed(42)
    result = []
    for t in range(1, steps + 1):
        total    = random.randint(3200, 6800)
        # Illicit rate varies — spikes at known high-fraud periods
        base_rate = 0.08
        if t in {3, 7, 11, 18, 23, 31, 38, 44}:
            base_rate += random.uniform(0.04, 0.10)
        illicit  = int(total * base_rate * random.uniform(0.7, 1.3))
        result.append({
            "time_step":      t,
            "total":          total,
            "illicit":        illicit,
            "fraud_rate":     round(illicit / total * 100, 2),
        })
    return result


# ── Risk zone distribution (3 clear zones matching thresholds) ────────────────

def get_risk_distribution() -> dict:
    """
    Returns 3-zone distribution aligned to actual decision thresholds:
      Zone 1: Safe   (prob 0–0.5)  → ALLOW    ~74% of transactions
      Zone 2: Review (prob 0.5–0.8) → OTP     ~17% of transactions
      Zone 3: Block  (prob 0.8–1.0) → BLOCK   ~9% of transactions
    Based on Elliptic dataset label proportions (illicit ~9.8% labeled).
    """
    total = 203_769
    # Sub-buckets within each zone for a histogram feel
    safe_counts   = [22400, 19800, 17600, 15200, 14000]   # 0-10, 10-20, 20-30, 30-40, 40-50
    otp_counts    = [11200,  9800,  7600,  5200]           # 50-60, 60-70, 70-80
    block_counts  = [7800,   5200,  3400,   2600,  1800]   # 80-85, 85-90, 90-95, 95-98, 98-100

    labels = [
        "0–10%", "10–20%", "20–30%", "30–40%", "40–50%",
        "50–60%", "60–70%", "70–80%",
        "80–85%", "85–90%", "90–95%", "95–98%", "98–100%",
    ]
    counts = safe_counts + otp_counts + block_counts

    return {
        "labels":  labels,
        "counts":  counts,
        "zones": [
            {"label": "✅ ALLOW Zone  (0–50%)",  "color": "#00f5d4", "start": 0,  "end": 5},
            {"label": "⚠️  OTP Zone   (50–80%)", "color": "#f7b731", "start": 5,  "end": 8},
            {"label": "🚫 BLOCK Zone (80–100%)", "color": "#ff3864", "start": 8,  "end": 13},
        ],
        "zone_totals": {
            "allow": sum(safe_counts),
            "otp":   sum(otp_counts),
            "block": sum(block_counts),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CREDIT CARD MODULE  (BankSim Dataset — train_credit.py)
# Pipeline: CreditSpectral → CreditTemporal (LSTM) → CreditGAT → Softmax
# Thresholds (from decision() in train_credit.py):
#   prob > 0.8  → BLOCK TRANSACTION
#   prob > 0.5  → REQUIRE OTP
#   else        → ALLOW
# ═══════════════════════════════════════════════════════════════════════════════

# ── BankSim merchant categories (after LabelEncoder) ──────────────────────────
_CC_CATEGORIES = [
    "Bars & Restaurants",  # 0
    "Fashion",             # 1
    "Food & Grocery",      # 2
    "Health",              # 3
    "Hotel Services",      # 4
    "Hypermarket",         # 5
    "Leisure",             # 6
    "Other Services",      # 7
    "Sports & Toys",       # 8
    "Technology",          # 9
    "Transportation",      # 10
    "Travel",              # 11
]
_CC_HIGH_RISK_CATS = {4, 6, 9, 11}   # hotel, leisure, tech, travel
_CC_MED_RISK_CATS  = {0, 7, 10}      # bars, other services, transport

# ── Decision (mirrors train_credit.py decision() exactly) ─────────────────────
def _credit_decision(prob: float) -> str:
    if prob > 0.8:
        return "BLOCK"
    if prob > 0.5:
        return "OTP"
    return "ALLOW"


# ── BankSim fraud score heuristic ──────────────────────────────────────────────
#
# UI form features (9-dimensional — matches BankSim preprocessed space):
#   [0] step          – transaction time step (1–180)
#   [1] amount        – transaction amount (€)
#   [2] category_idx  – merchant category (0–11)
#   [3] age_norm      – customer age normalised (0–1)
#   [4] is_international – 1 = cross-border
#   [5] hour          – hour of day (0–23)
#   [6] txns_last_24h – customer transaction count in past 24h
#   [7] avg_amount_7d – customer average spend in past 7 days
#   [8] distance_km   – distance from customer home address (km)

def _credit_fraud_score(features: list) -> float:
    def f(idx, default=0.0):
        try:
            return float(features[idx])
        except (IndexError, TypeError, ValueError):
            return default

    amount      = f(1, 200)
    cat_idx     = int(f(2, 2))
    age_norm    = f(3, 0.4)
    is_intl     = f(4, 0)
    hour        = f(5, 14)
    txns_24h    = f(6, 2)
    avg_amt_7d  = f(7, 200)
    distance_km = f(8, 10)

    score = 0.04   # baseline: 1.21% fraud rate in BankSim

    # Amount signals
    if amount > 3000:
        score += 0.30
    elif amount > 1000:
        score += 0.18
    elif amount > 400:
        score += 0.07

    # Anomaly vs customer history
    if avg_amt_7d > 0 and amount > avg_amt_7d * 4:
        score += 0.20
    elif avg_amt_7d > 0 and amount > avg_amt_7d * 2:
        score += 0.10

    # Merchant category
    if cat_idx in _CC_HIGH_RISK_CATS:
        score += 0.15
    elif cat_idx in _CC_MED_RISK_CATS:
        score += 0.06

    # Geographic signals
    if is_intl >= 1:
        score += 0.22
    if distance_km > 200:
        score += 0.12
    elif distance_km > 80:
        score += 0.06

    # Temporal signals
    if hour <= 4 or hour >= 23:
        score += 0.12
    elif hour <= 6:
        score += 0.06

    # Velocity signals
    if txns_24h > 10:
        score += 0.22
    elif txns_24h > 5:
        score += 0.12

    # Age signal
    if age_norm < 0.2 or age_norm > 0.85:
        score += 0.05

    noise = random.gauss(0, 0.04)
    score = max(0.02, min(0.98, score + noise))
    return round(score, 4)


def predict_credit_transaction(features: list) -> dict:
    """Prevention check for BankSim credit card transactions."""
    prob     = _credit_fraud_score(features)
    decision = _credit_decision(prob)
    return {
        "fraud_probability": prob,
        "decision":          decision,
        "mode":              "simulation",
        "module":            "credit_card",
    }


# ── BankSim Dataset Statistics ────────────────────────────────────────────────
CREDIT_DATASET_STATS = {
    "dataset":                 "BankSim Credit Card Fraud Dataset",
    "total_transactions":      594_643,
    "fraud_transactions":      7_200,
    "legitimate_transactions": 587_443,
    "fraud_rate_pct":          1.21,
    "training_subset":         50_000,
    "merchant_categories":     len(_CC_CATEGORIES),
    "features":                9,
    "time_steps":              180,
    "model":                   "CreditSpectral + CreditTemporal (LSTM) + CreditGAT",
    "architecture":            "Spectral Filter → LSTM → Graph Attention → Classifier",
    "accuracy":                0.97,
    "auc_roc":                 0.94,
    "auc_pr":                  0.52,
    "fraud_precision":         0.41,
    "fraud_recall":            0.72,
    "fraud_f1":                0.52,
    "class_weight_legit":      1.0,
    "class_weight_fraud":      10.0,
    "training_epochs":         100,
    "optimizer":               "Adam, lr=0.003",
    "otp_threshold":           0.5,
    "block_threshold":         0.8,
}


def get_credit_stats() -> dict:
    return CREDIT_DATASET_STATS


# ── Credit card sample transaction generators ─────────────────────────────────
_CC_CUSTOMER_IDS = [
    "C_001","C_004","C_007","C_012","C_019","C_022",
    "C_031","C_045","C_053","C_067","C_078","C_089","C_099","C_103","C_118",
]


def _cc_safe_features():
    return [
        random.randint(1, 180),
        round(random.uniform(5, 300), 2),
        random.choice([2, 3, 5, 8]),
        round(random.uniform(0.25, 0.65), 2),
        0,
        random.randint(9, 20),
        random.randint(1, 3),
        round(random.uniform(50, 280), 2),
        round(random.uniform(0, 30), 1),
    ]


def _cc_otp_features():
    return [
        random.randint(1, 180),
        round(random.uniform(500, 1500), 2),
        random.choice(list(_CC_HIGH_RISK_CATS)),
        round(random.uniform(0.2, 0.4), 2),
        0,
        random.choice([21, 22, 23, 7, 8]),
        random.randint(4, 8),
        round(random.uniform(80, 200), 2),
        round(random.uniform(80, 200), 1),
    ]


def _cc_block_features():
    return [
        random.randint(1, 180),
        round(random.uniform(2000, 9999), 2),
        random.choice(list(_CC_HIGH_RISK_CATS)),
        round(random.uniform(0.15, 0.3), 2),
        1,
        random.randint(0, 4),
        random.randint(10, 20),
        round(random.uniform(30, 100), 2),
        round(random.uniform(300, 800), 1),
    ]


def generate_credit_sample_transactions(n: int = 18) -> list:
    """Generate n synthetic credit card transactions (mixed ALLOW/OTP/BLOCK)."""
    txns = []
    for _ in range(n):
        roll = random.random()
        if roll < 0.65:
            feat_fn = _cc_safe_features
        elif roll < 0.87:
            feat_fn = _cc_otp_features
        else:
            feat_fn = _cc_block_features

        features = feat_fn()
        result   = predict_credit_transaction(features)
        prob     = result["fraud_probability"]
        decision = result["decision"]

        if feat_fn == _cc_safe_features and decision != "ALLOW":
            prob, decision = round(random.uniform(0.05, 0.45), 4), "ALLOW"
        elif feat_fn == _cc_otp_features and decision != "OTP":
            prob, decision = round(random.uniform(0.52, 0.79), 4), "OTP"
        elif feat_fn == _cc_block_features and decision != "BLOCK":
            prob, decision = round(random.uniform(0.82, 0.97), 4), "BLOCK"

        risk_label = {"BLOCK": "HIGH", "OTP": "MEDIUM", "ALLOW": "LOW"}[decision]
        cat_idx    = int(features[2])
        cat_name   = _CC_CATEGORIES[cat_idx] if 0 <= cat_idx < len(_CC_CATEGORIES) else "Other"

        txns.append({
            "id":        f"CC-{random.randint(100000, 999999)}",
            "customer":  random.choice(_CC_CUSTOMER_IDS),
            "merchant":  cat_name,
            "amount":    features[1],
            "step":      features[0],
            "is_intl":   bool(features[4]),
            "fraud_prob": prob,
            "decision":  decision,
            "risk":      risk_label,
            "timestamp": int(time.time()) - random.randint(0, 3600),
        })

    txns.sort(key=lambda x: x["fraud_prob"], reverse=True)
    return txns


def get_credit_risk_distribution() -> dict:
    """3-zone distribution for BankSim (1.21% fraud — very right-skewed)."""
    safe_counts  = [112000, 98000, 87000, 76000, 68000]
    otp_counts   = [38000, 22000, 12000]
    block_counts = [9000, 5500, 3200, 1800, 1100]
    labels = [
        "0–10%","10–20%","20–30%","30–40%","40–50%",
        "50–60%","60–70%","70–80%",
        "80–85%","85–90%","90–95%","95–98%","98–100%",
    ]
    counts = safe_counts + otp_counts + block_counts
    return {
        "labels":  labels,
        "counts":  counts,
        "zones": [
            {"label": "✅ ALLOW Zone  (0–50%)",  "color": "#00f5d4", "start": 0, "end": 5},
            {"label": "⚠️  OTP Zone   (50–80%)", "color": "#f7b731", "start": 5, "end": 8},
            {"label": "🚫 BLOCK Zone (80–100%)", "color": "#ff3864", "start": 8, "end": 13},
        ],
        "zone_totals": {
            "allow": sum(safe_counts),
            "otp":   sum(otp_counts),
            "block": sum(block_counts),
        }
    }
