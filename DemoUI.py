# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 10:17:25 2025
@author: vidya
"""

import streamlit as st
import pandas as pd
import numpy as np
import io, re, os, warnings, joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from Model import (
    engineer_dataframe,
    sb_label,
    categorize_score_numeric,
    parse_num
)

warnings.filterwarnings("ignore")

# ============================================================
#                     STREAMLIT UI SETUP
# ============================================================
st.title("📊 Financial Health (FH) & SB Rating Prediction Tool")
st.write("Upload datasets → Click Calculate → Training + Prediction + Dashboard.")

st.sidebar.header("⚙ Settings")


# ============================================================
#   SECTION 1 — UPLOAD MASTER DATASET (Training Data)
# ============================================================
st.subheader("1️⃣ Upload Master Dataset (Historic FY Data)")

master_file = st.file_uploader("Upload master Excel file", type=["xlsx"], key="master")

if master_file is not None:
    df_master_raw = pd.read_excel(master_file, engine="openpyxl", dtype=object)
    st.success(f"Master dataset loaded — {len(df_master_raw)} rows")

    # Engineer & store
    with st.spinner("Engineering master dataset..."):
        df_master_eng = engineer_dataframe(df_master_raw)

    st.session_state["MASTER_ENG"] = df_master_eng


# ============================================================
#   SECTION 2 — UPLOAD INPUT DATASET (Prediction Input)
# ============================================================
st.subheader("2️⃣ Upload Input Excel for Prediction")

input_file = st.file_uploader("Upload input Excel file", type=["xlsx"], key="input")

if input_file is not None:
    df_input_raw = pd.read_excel(input_file, engine="openpyxl", dtype=object)
    st.success(f"Input dataset loaded — {len(df_input_raw)} rows")
    st.session_state["INPUT_RAW"] = df_input_raw


# ============================================================
#   SECTION 3 — SINGLE BUTTON → TRAIN + PREDICT + DASHBOARD
# ============================================================
st.subheader("3️⃣ Run Complete Risk Analysis")

if st.button("🚀 Calculate Risk"):

    # -------------------------------------------
    # Safety Checks
    # -------------------------------------------
    if "MASTER_ENG" not in st.session_state:
        st.error("Upload Master Dataset first.")
        st.stop()

    if "INPUT_RAW" not in st.session_state:
        st.error("Upload Input File first.")
        st.stop()

    df_master = st.session_state["MASTER_ENG"]
    df_input_raw = st.session_state["INPUT_RAW"]

    # -------------------------------------------
    # Feature List
    # -------------------------------------------
    FEATURES = [
        "Debt_Equity","EBITDA_Margin","PAT_Margin",
        "DSCR","Current Ratio","ROCE (%)","ROE (%)","Credit Utilization (%)",
        "DPD_CurrentLoan","Bounced_Cheques","SMA_Flag","CRILC_Exposure_num",
        "Loan_Type_Code","Collateral_Value_num","LTV_num","Loan_Amount_num","Loan_Tenure_Months",
        "Existing_Loan_Sanctioned_num","Existing_Loan_Outstanding_num",
        "Promoter_Risk","Management_Risk","Industry_Risk","ESG_Risk",
        "Document_Quality_Score","Loan_Type_EWS",
        "Growth_1Y","Growth_3Y_Avg","Trend_Slope"
    ]

    # Ensure missing feature columns exist
    for f in FEATURES:
        if f not in df_master.columns:
            df_master[f] = np.nan

    # -------------------------------------------
    # TRAIN MODEL ON MASTER
    # -------------------------------------------
    with st.spinner("Training ML Model using Master Dataset..."):

        X_train_full = df_master[FEATURES].applymap(lambda x: parse_num(x) if not pd.isna(x) else np.nan)
        feature_medians = X_train_full.median(numeric_only=True)
        X_train_full = X_train_full.fillna(feature_medians)

        y_train_full = df_master["FH_Score"].astype(float)

        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train_full, y_train_full)

        st.success("Training Completed!")

    # -------------------------------------------
    # ENGINEER INPUT DATA FOR PREDICTION
    # -------------------------------------------
    with st.spinner("Engineering input dataset..."):
        df_input_eng = engineer_dataframe(df_input_raw)

    X_pred = df_input_eng[FEATURES].applymap(lambda x: parse_num(x) if not pd.isna(x) else np.nan)
    X_pred = X_pred.fillna(feature_medians)

    # -------------------------------------------
    # PREDICT
    # -------------------------------------------
    results = []

    for idx, row in df_input_eng.iterrows():

        fh_formula = row["FH_Score"]
        sb_f, sb_desc_f, sb_range_f = sb_label(fh_formula)
        rb_formula = categorize_score_numeric(fh_formula)

        fh_ml = ridge.predict(X_pred.iloc[[idx]])[0]
        sb_ml, sb_desc_ml, sb_range_ml = sb_label(fh_ml)
        rb_ml = categorize_score_numeric(fh_ml)

        results.append({
            "Company Name": row.get("Company Name",""),
            "FH_Score_Formula": fh_formula,
            "SB_Formula": sb_f,
            "RiskBand_Formula": rb_formula,
            "FH_Score_Ridge": fh_ml,
            "SB_Ridge": sb_ml,
            "RiskBand_Ridge": rb_ml
        })

    df_results = pd.DataFrame(results)

    # ============================================================
    #   SECTION 4 — DASHBOARD & GRAPHS
    # ============================================================
    st.subheader("📌 Company Dashboard")

    company_list = df_input_eng["Company Name"].dropna().unique()
    selected_company = st.selectbox("Select a company", company_list)

    comp_results = df_results[df_results["Company Name"] == selected_company].iloc[-1]
    fh_pred = comp_results["FH_Score_Ridge"]

    # -------------------------------
    # COLOR CARDS
    # -------------------------------
    st.markdown("### 🧩 Score Summary")

    def get_color(rb):
        return "#2ECC71" if rb=="Low" else "#F1C40F" if rb=="Moderate" else "#E74C3C"

    col1, col2 = st.columns(2)

    with col1:
        st.metric("FH Score (Formula)", f"{comp_results['FH_Score_Formula']:.2f}")
        st.metric("SB Rating (Formula)", comp_results["SB_Formula"])
        st.metric("Risk Band (Formula)", comp_results["RiskBand_Formula"])

    with col2:
        st.metric("FH Score (ML)", f"{fh_pred:.2f}")
        st.metric("SB Rating (ML)", comp_results["SB_Ridge"])
        st.metric("Risk Band (ML)", comp_results["RiskBand_Ridge"])


    # -----------------------------------------------------------
    # FORMULA FH TREND (Historical only)
    # -----------------------------------------------------------
    st.markdown("### 📈 Formula FH Trend (Historical)")

    hist = df_master[df_master["Company Name"] == selected_company].sort_values("FY_num")

    if not hist.empty:
        plt.figure(figsize=(8,4))
        plt.plot(hist["FY_num"], hist["FH_Score"], marker="o")
        plt.grid(alpha=0.3)
        plt.xlabel("Financial Year")
        plt.ylabel("FH Score (Formula)")
        plt.title(f"Historical FH Score Trend - {selected_company}")
        st.pyplot(plt)
    else:
        st.info("No historical data found for this company.")


    # -----------------------------------------------------------
    # PREDICTED TREND (Current + Next Year)
    # -----------------------------------------------------------
    st.markdown("### 🤖 ML Predicted Trend (Including Next FY)")

    if not hist.empty:
        years = hist["FY_num"].tolist()
        ml_scores = hist["FH_Score"].tolist()

        next_year = max(years) + 1
        years.append(next_year)
        ml_scores.append(fh_pred)

        plt.figure(figsize=(8,4))
        plt.plot(years, ml_scores, marker="o", color="purple")
        plt.grid(alpha=0.3)
        plt.xlabel("Financial Year")
        plt.ylabel("Predicted FH Score (ML)")
        plt.title(f"Predicted Trend (Including {next_year}) - {selected_company}")
        st.pyplot(plt)


    # -----------------------------------------------------------
    # DOWNLOAD RESULTS
    # -----------------------------------------------------------
    st.subheader("📊 Download Results")

    output = io.BytesIO()
    df_results.to_excel(output, index=False, engine="openpyxl")

    st.download_button(
        label="📥 Download Full Results",
        data=output.getvalue(),
        file_name="FH_Scoring_Results.xlsx",
        mime="application/vnd.ms-excel"
    )
