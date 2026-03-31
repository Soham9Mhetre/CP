import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np


def load_credit_card_data(path):

    df = pd.read_csv(path)

    # Strip quotes
    df = df.map(lambda x: str(x).replace("'", "") if isinstance(x, str) else x)

    # Drop useless columns
    for col in ["zipcodeOri", "zipMerchant"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Encode categoricals
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Sort by time
    if "step" in df.columns:
        df = df.sort_values("step").reset_index(drop=True)

    # Feature engineering
    if "amount" in df.columns:
        df["log_amount"] = np.log1p(df["amount"].astype(float))

    y = df["fraud"].astype(int).values
    X = df.drop("fraud", axis=1).astype(float)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return (
        torch.tensor(X, dtype=torch.float),
        torch.tensor(y, dtype=torch.long),
    )