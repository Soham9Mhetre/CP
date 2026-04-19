import streamlit as st
import pandas as pd

from app.unified_fusion_service import UnifiedFusionSystem
from data.credit_card_loader import load_credit_card_data
from data.load_dataset import load_dataset

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("🚨 Omni Fraud Prevention System")
st.markdown("Real-time Multi-Model Fraud Detection Dashboard")

# -------------------------
# INIT SYSTEM
# -------------------------
system = UnifiedFusionSystem()

# -------------------------
# BUTTON
# -------------------------
if st.button("🔍 Run Fusion Demo"):

    with st.spinner("Running fraud detection..."):

        crypto_data = load_dataset()
        credit_data = load_credit_card_data("data/banksim.csv")

        results = system.predict(crypto_data, credit_data)

    st.success("✅ Analysis Complete")

    # -------------------------
    # FORMAT DATA
    # -------------------------
    table_data = []

    for i, r in enumerate(results[:20]):

        table_data.append({
            "Txn": i,
            "Crypto Prob": round(r["crypto_prob"], 3),
            "Credit Prob": round(r["credit_prob"], 3),
            "Final Risk": round(r["final_prob"], 3),
            "Uncertainty": round(r["uncertainty"], 3),
            "Decision": r["decision"],
            "Mode": r["mode"]
        })

    df = pd.DataFrame(table_data)

    # -------------------------
    # COLOR CODING
    # -------------------------
    def highlight_decision(val):
        if val == "BLOCK":
            return "background-color: #ff4d4d; color: white;"
        elif val == "OTP":
            return "background-color: #ffa500; color: black;"
        elif val == "ANALYST":
            return "background-color: #6c757d; color: white;"
        elif val == "ALLOW":
            return "background-color: #28a745; color: white;"
        return ""

    styled_df = df.style.applymap(highlight_decision, subset=["Decision"])

    # -------------------------
    # DISPLAY TABLE
    # -------------------------
    st.subheader("📊 Transaction Risk Analysis")
    st.dataframe(styled_df, use_container_width=True)

    # -------------------------
    # SUMMARY METRICS
    # -------------------------
    st.subheader("📈 Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Blocked", sum(df["Decision"] == "BLOCK"))
    col2.metric("OTP Triggered", sum(df["Decision"] == "OTP"))
    col3.metric("Analyst Review", sum(df["Decision"] == "ANALYST"))
    col4.metric("Allowed", sum(df["Decision"] == "ALLOW"))

    # -------------------------
    # TOP FRAUD ALERT
    # -------------------------
    st.subheader("🚨 Top Risk Transaction")

    top_txn = df.sort_values("Final Risk", ascending=False).iloc[0]

    st.error(
        f"""
        **Txn {top_txn['Txn']} flagged as HIGH RISK**

        • Final Risk: {top_txn['Final Risk']}  
        • Crypto Score: {top_txn['Crypto Prob']}  
        • Credit Score: {top_txn['Credit Prob']}  
        • Decision: {top_txn['Decision']}  
        • Mode: {top_txn['Mode']}
        """
    )