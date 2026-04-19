import torch

from app.inference_crypto import predict_crypto
from app.inference_credit import predict_credit, initialize_credit_pipeline
from app.fusion_engine import FusionEngine


class UnifiedFusionSystem:

    def __init__(self):
        self.fusion = FusionEngine()
        self.initialized = False

    def _initialize(self, crypto_data, credit_data):
        initialize_credit_pipeline(credit_data)
        self.initialized = True

    def predict(self, crypto_data, credit_data, num_samples=50):

        if not self.initialized:
            self._initialize(crypto_data, credit_data)

        results = []
        fused_probs = []

        # -------------------------
        # FIRST PASS (threshold learning)
        # -------------------------
        for i in range(num_samples):

            c_out = predict_crypto(crypto_data, i)
            k_out = predict_credit(credit_data, i)

            prob, _ = self.fusion.fuse(c_out, k_out)
            fused_probs.append(prob)

        # Dummy labels (same as demo logic)
        labels = [1 if p > 0.7 else 0 for p in fused_probs]
        self.fusion.fit_thresholds(fused_probs, labels)

        # -------------------------
        # FINAL PASS
        # -------------------------
        for i in range(num_samples):

            c_out = predict_crypto(crypto_data, i)
            k_out = predict_credit(credit_data, i)

            prob, mode = self.fusion.fuse(c_out, k_out)

            uncertainty = 1 - abs(prob - 0.5) * 2
            decision = self.fusion.decide(prob, uncertainty)

            results.append({
                "txn_id": i,
                "crypto_prob": float(c_out["fraud_prob"]),
                "credit_prob": float(k_out["fraud_prob"]),
                "final_prob": float(prob),
                "uncertainty": float(uncertainty),
                "decision": decision,
                "mode": mode
            })

        return results