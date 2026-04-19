import torch

from app.inference_crypto import predict_crypto
from app.inference_credit import predict_credit, initialize_credit_pipeline
from app.fusion_engine import FusionEngine

from data.load_dataset import load_dataset
from data.credit_card_loader import load_credit_card_data


class UnifiedFusionSystem:

    def __init__(self):
        self.fusion = FusionEngine()
        self.initialized = False

    def _initialize(self, crypto_data, credit_data):
        # Initialize credit pipeline once
        initialize_credit_pipeline(credit_data)
        self.initialized = True

    def predict(self, crypto_data, credit_data, num_samples=50):

        if not self.initialized:
            self._initialize(crypto_data, credit_data)

        crypto_outputs = []
        credit_outputs = []
        fused_probs = []
        labels = []

        # -------------------------
        # RUN PREDICTIONS
        # -------------------------
        for i in range(num_samples):

            c_out = predict_crypto(crypto_data, i)
            k_out = predict_credit(credit_data, i)

            prob, mode = self.fusion.fuse(c_out, k_out)

            crypto_outputs.append(c_out)
            credit_outputs.append(k_out)
            fused_probs.append(prob)

            # dummy labels (only for threshold fitting)
            labels.append(1 if prob > 0.7 else 0)

        # -------------------------
        # FIT THRESHOLDS
        # -------------------------
        self.fusion.fit_thresholds(fused_probs, labels)

        # -------------------------
        # FINAL RESULTS
        # -------------------------
        results = []

        for i in range(num_samples):

            c_out = crypto_outputs[i]
            k_out = credit_outputs[i]

            prob, mode = self.fusion.fuse(c_out, k_out)

            # ✅ FIXED uncertainty
            uncertainty = 1 - abs(prob - 0.5) * 2

            decision = self.fusion.decide(prob, uncertainty)

            results.append({
                "txn_id": i,
                "crypto_prob": c_out["fraud_prob"],
                "credit_prob": k_out["fraud_prob"],
                "final_prob": prob,
                "uncertainty": uncertainty,
                "decision": decision,
                "mode": mode
            })

        return results