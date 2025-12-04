# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 10:17:25 2025
@author: vidya
"""

import streamlit as st
import pandas as pd
import numpy as np
import io, warnings
from sklearn.linear_model import Ridge
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
st.subheader(" Upload Master Dataset ")

master_file = st.file_uploader("Upload master Excel file", type=["xlsx"], key="master")

if master_file is not None:
    df_master_raw = pd.read_excel(master_file, engine="openpyxl", dtype=object)
    st.success(f"Master dataset loaded — {len(df_master_raw)} rows")

    with st.spinner("Engineering master dataset..."):
        df_master_eng = engineer_dataframe(df_master_raw)

    st.session_state["MASTER_ENG"] = df_master_eng


# ============================================================
#   SECTION 2 — UPLOAD INPUT DATASET (Prediction Input)
# ============================================================
st.subheader("Upload Input Excel for Prediction")

input_file = st.file_uploader("Upload input Excel file", type=["xlsx"], key="input")

if input_file is not None:
    df_input_raw = pd.read_excel(input_file, engine="openpyxl", dtype=object)
    st.success(f"Input dataset loaded — {len(df_input_raw)} rows")
    st.session_state["INPUT_RAW"] = df_input_raw


# ============================================================
#   SECTION 3 — SINGLE BUTTON → TRAIN + PREDICT + DASHBOARD
# ============================================================
st.subheader("Run Complete Risk Analysis")

if st.button("Calculate Risk"):

    # ==================== SAFETY CHECKS ====================
    if "MASTER_ENG" not in st.session_state:
        st.error("Upload Master Dataset first.")
        st.stop()

    if "INPUT_RAW" not in st.session_state:
        st.error("Upload Input File first.")
        st.stop()

    df_master = st.session_state["MASTER_ENG"]
    df_input_raw = st.session_state["INPUT_RAW"]

    # ==================== FEATURE LIST ====================
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

    for f in FEATURES:
        if f not in df_master.columns:
            df_master[f] = np.nan

    # ==================== TRAIN MODEL ====================
    with st.spinner("Training ML Model using Master Dataset..."):

        X_train_full = df_master[FEATURES].applymap(
            lambda x: parse_num(x) if not pd.isna(x) else np.nan
        )
        feature_medians = X_train_full.median(numeric_only=True)
        X_train_full = X_train_full.fillna(feature_medians)

        y_train_full = df_master["FH_Score"].astype(float)

        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train_full, y_train_full)

        st.success("Training Completed!")

    # ==================== PREP INPUT ====================
    with st.spinner("Engineering input dataset..."):
        df_input_eng = engineer_dataframe(df_input_raw)

    X_pred = df_input_eng[FEATURES].applymap(
        lambda x: parse_num(x) if not pd.isna(x) else np.nan
    )
    X_pred = X_pred.fillna(feature_medians)

    # ==================== PREDICT ====================
    results = []
    for idx, row in df_input_eng.iterrows():

        fh_formula = row["FH_Score"]
        sb_f, _, _ = sb_label(fh_formula)
        rb_formula = categorize_score_numeric(fh_formula)

        fh_ml = ridge.predict(X_pred.iloc[[idx]])[0]
        sb_ml, _, _ = sb_label(fh_ml)
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
    #                     DASHBOARD
    # ============================================================
    st.subheader("Company Dashboard")

    company_list = df_input_eng["Company Name"].dropna().unique()
    selected_company = st.selectbox("Select a company", company_list)

    comp_result = df_results[df_results["Company Name"] == selected_company].iloc[-1]
    fh_pred = comp_result["FH_Score_Ridge"]

    # ------------------------ SCORE CARDS ------------------------
    st.markdown(" Score Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("FH Score (Formula)", f"{comp_result['FH_Score_Formula']:.2f}")
        st.metric("SB Rating (Formula)", comp_result["SB_Formula"])
        st.metric("Risk Band (Formula)", comp_result["RiskBand_Formula"])

    with col2:
        st.metric("FH Score (ML Prediction)", f"{fh_pred:.2f}")
        st.metric("SB Rating (ML)", comp_result["SB_Ridge"])
        st.metric("Risk Band (ML)", comp_result["RiskBand_Ridge"])

    # ============================================================
    #     FORMULA TREND GRAPH (Historical)
    # ============================================================
    st.markdown(" 📈 Formula FH Trend (Historical)")

    hist = df_master[df_master["Company Name"] == selected_company].sort_values("FY_num")

    if not hist.empty:

        years_hist = hist["FY_num"].astype(int).tolist()

        plt.figure(figsize=(8,4))
        plt.plot(years_hist, hist["FH_Score"], marker="o")
        plt.xticks(years_hist)
        plt.xlabel("Financial Year")
        plt.ylabel("FH Score (Formula)")
        plt.title(f"Historical FH Trend - {selected_company}")
        plt.grid(alpha=0.3)
        st.pyplot(plt)

    else:
        st.info("No historical records found.")

    # ============================================================
    #     ML TREND GRAPH (Next FY Prediction)
    # ============================================================
    st.markdown(" ML Predicted Trend ( Next FY)")

    if not hist.empty:

        years_ml = hist["FY_num"].astype(int).tolist()
        scores_ml = hist["FH_Score"].tolist()

        next_year = max(years_ml) + 1
        years_ml.append(next_year)
        scores_ml.append(fh_pred)

        plt.figure(figsize=(8,4))
        plt.plot(years_ml, scores_ml, marker="o", color="purple")
        plt.xticks(years_ml)
        plt.xlabel("Financial Year")
        plt.ylabel("Predicted FH Score")
        plt.title(f"Predicted FH Trend Including {next_year} - {selected_company}")
        plt.grid(alpha=0.3)
        st.pyplot(plt)

    # ============================================================
    # DOWNLOAD RESULTS
    # ============================================================
    st.subheader("📊 Download Results")

    output = io.BytesIO()
    df_results.to_excel(output, index=False, engine="openpyxl")

    st.download_button(
        label="📥 Download Results",
        data=output.getvalue(),
        file_name="FH_Scoring_Results.xlsx",
        mime="application/vnd.ms-excel"
    )

