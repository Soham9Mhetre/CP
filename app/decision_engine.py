import numpy as np

def compute_uncertainty(prob):
    # Confidence-based uncertainty (better for fraud systems)
    return min(prob, 1 - prob)

def make_decision(prob, threshold, uncertainty):

    if uncertainty > 0.4:
        return "ANALYST"

    elif prob > threshold + 0.1:
        return "BLOCK"

    elif prob > threshold:
        return "OTP"

    elif prob > 0.2:   #  increased from 0.05
        return "OTP"

    else:
        return "ALLOW"