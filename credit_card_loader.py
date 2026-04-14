import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_credit_card_data(path):

    # ======================
    # LOAD DATA
    # ======================
    df = pd.read_csv(path)

    # ======================
    # CLEAN STRINGS (VERY IMPORTANT)
    # ======================
    df = df.map(lambda x: str(x).replace("'", "") if isinstance(x, str) else x)

    # ======================
    # DROP USELESS COLUMNS
    # ======================
    drop_cols = ["zipcodeOri", "zipMerchant"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # ======================
    # HANDLE MISSING VALUES
    # ======================
    df = df.fillna(0)

    # ======================
    # ENCODE CATEGORICAL
    # ======================
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # ======================
    # SORT BY TIME (CRITICAL)
    # ======================
    if "step" in df.columns:
        df = df.sort_values("step").reset_index(drop=True)

    # ======================
    # TARGET
    # ======================
    y = df["fraud"].astype(int).values

    # ======================
    # FEATURES
    # ======================
    X = df.drop("fraud", axis=1)

    # Ensure numeric
    X = X.astype(float)

    # ======================
    # NORMALIZE FEATURES
    # ======================
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ======================
    # CONVERT TO TENSORS
    # ======================
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    return X, y