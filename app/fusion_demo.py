import torch

from app.inference_crypto import predict_crypto
from app.inference_credit import predict_credit, initialize_credit_pipeline
from app.fusion_engine import FusionEngine

from data.load_dataset import load_dataset
from data.credit_card_loader import load_credit_card_data


def run():
    print("\n🚀 Running Unified Fraud Fusion System...\n")

    fusion = FusionEngine()

    # -------------------------
    # LOAD DATA
    # -------------------------
    crypto_data = load_dataset()

    # Credit dataset (features + labels)
    X_credit, y_credit = load_credit_card_data("data/banksim.csv")

    X_credit = X_credit.clone()
    y_credit = y_credit.clone()

    # -------------------------
    # 🔥 SMART SAMPLING (IMPORTANT)
    # Ensures fraud appears in demo
    # -------------------------
    sample_size = 3000

    fraud_idx = (y_credit == 1).nonzero(as_tuple=True)[0]
    normal_idx = (y_credit == 0).nonzero(as_tuple=True)[0]

    # Handle edge cases safely
    num_fraud = min(100, len(fraud_idx))
    num_normal = sample_size - num_fraud

    fraud_sample = fraud_idx[torch.randperm(len(fraud_idx))[:num_fraud]]
    normal_sample = normal_idx[torch.randperm(len(normal_idx))[:num_normal]]

    indices = torch.cat([fraud_sample, normal_sample])

    # Shuffle for randomness
    indices = indices[torch.randperm(len(indices))]

    X_credit = X_credit[indices]
    y_credit = y_credit[indices]

    # -------------------------
    # PRECOMPUTE CREDIT PIPELINE (FAST)
    # -------------------------
    print("⚡ Initializing credit pipeline (one-time)...")
    initialize_credit_pipeline(X_credit)

    crypto_outputs = []
    credit_outputs = []
    fused_probs = []
    labels = []

    # -------------------------
    # RUN DEMO LOOP
    # -------------------------
    for i in range(50):
        c_out = predict_crypto(crypto_data, i)
        k_out = predict_credit(X_credit, i)

        prob, _ = fusion.fuse(c_out, k_out)

        crypto_outputs.append(c_out)
        credit_outputs.append(k_out)
        fused_probs.append(prob)

        # ✅ REAL LABEL (not fake)
        labels.append(int(y_credit[i].item()))

    # -------------------------
    # LEARN THRESHOLDS
    # -------------------------
    fusion.fit_thresholds(fused_probs, labels)

    # -------------------------
    # DISPLAY RESULTS
    # -------------------------
    print("\n--- 🔍 TOP SUSPICIOUS TRANSACTIONS ---\n")

    results = []

    for i in range(50):
        c_out = crypto_outputs[i]
        k_out = credit_outputs[i]

        prob, mode = fusion.fuse(c_out, k_out)
        uncertainty = 1 - abs(prob - 0.5) * 2
        decision = fusion.decide(prob, uncertainty)

        results.append((
            i,
            prob,
            c_out["fraud_prob"],
            k_out["fraud_prob"],
            uncertainty,
            decision,
            mode
        ))

    # Sort by risk
    results.sort(key=lambda x: x[1], reverse=True)

    # Show top 15
    for r in results[:15]:
        i, prob, pc, pk, unc, decision, mode = r

        print(
            f"Txn {i:02d} | "
            f"C:{pc:.3f} | K:{pk:.3f} | "
            f"F:{prob:.3f} | Unc:{unc:.3f} | "
            f"{decision} | {mode}"
        )

    print("\n✅ Demo completed successfully.\n")


if __name__ == "__main__":
    run()