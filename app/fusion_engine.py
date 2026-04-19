import numpy as np


class FusionEngine:
    def __init__(self):
        self.block_threshold = 0.7
        self.analyst_threshold = 0.4

    def calibrate(self, prob):
    # Stretch probability distribution
    # maps low values upward without breaking high ones
        return prob ** 0.5   # sqrt scaling
    # -------------------------
    # FUSION    
    # -------------------------
    def fuse(self, crypto_out, credit_out):
        pc = crypto_out["fraud_prob"]
        uc = crypto_out["uncertainty"]

        pk = credit_out["fraud_prob"]
        uk = credit_out["uncertainty"]

        conf_c = 1 - uc
        conf_k = 1 - uk

        total_conf = conf_c + conf_k + 1e-6

        w_c = conf_c / total_conf
        w_k = conf_k / total_conf

        final_prob = w_c * pc + w_k * pk

        if w_k > 0.7:
            mode = "CREDIT_HEAVY"
        elif w_c > 0.7:
            mode = "CRYPTO_HEAVY"
        else:
            mode = "BALANCED"

        final_prob = self.calibrate(final_prob)
        return final_prob, mode

    # -------------------------
    # ROBUST THRESHOLD LEARNING
    # -------------------------
    def fit_thresholds(self, probs, labels):
        probs = np.array(probs)

        p90 = np.percentile(probs, 90)
        p70 = np.percentile(probs, 70)

        block_floor = 0.4
        analyst_floor = 0.2

        self.block_threshold = max(p90, block_floor)
        self.analyst_threshold = max(p70, analyst_floor)

        print("\n🔥 Learned thresholds (robust):")
        print(f"BLOCK: {self.block_threshold:.3f}")
        print(f"ANALYST: {self.analyst_threshold:.3f}")

    # -------------------------
    # DECISION (FIXED)
    # -------------------------
    def decide(self, prob, uncertainty=None):

        # 1. High risk → BLOCK
        if prob >= self.block_threshold:
            return "BLOCK"

        # 2. Moderate risk → OTP
        if prob >= self.analyst_threshold:
            return "OTP"

        # 3. Very high uncertainty ONLY → ANALYST
        if uncertainty is not None and uncertainty > 0.7:
            return "ANALYST"

        # 4. Otherwise → ALLOW
        return "ALLOW"