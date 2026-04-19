import streamlit as st
import pandas as pd

from app.unified_fusion_service import UnifiedFusionSystem
from data.load_dataset import load_dataset
from data.credit_card_loader import load_credit_card_data


st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("💳 Unified Fraud Detection System")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    crypto_data = load_dataset()
    X_credit, y_credit = load_credit_card_data("data/banksim.csv")
    return crypto_data, X_credit, y_credit


crypto_data, X_credit, y_credit = load_data()

# -------------------------
# RUN SYSTEM
# -------------------------
system = UnifiedFusionSystem()

if st.button("🚀 Run Fraud Detection"):

    with st.spinner("Running models..."):

        results = system.predict(crypto_data, X_credit)

        df = pd.DataFrame(results)

    st.success("✅ Detection Completed")

    # -------------------------
    # DISPLAY TABLE
    # -------------------------
    st.dataframe(df.sort_values("final_prob", ascending=False))

    # -------------------------
    # METRICS
    # -------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("🚨 High Risk", (df["decision"] == "BLOCK").sum())
    col2.metric("⚠️ OTP", (df["decision"] == "OTP").sum())
    col3.metric("🧠 Analyst", (df["decision"] == "ANALYST").sum())

    # -------------------------
    # FILTER VIEW
    # -------------------------
    st.subheader("🔍 Filter Results")

    option = st.selectbox("Decision Type", ["ALL", "BLOCK", "OTP", "ANALYST", "ALLOW"])

    if option != "ALL":
        st.dataframe(df[df["decision"] == option])