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
from engine import (
    engineer_dataframe,
    sb_label,
    categorize_score_numeric,
    parse_num
)


warnings.filterwarnings("ignore")

# -------------------------
# IMPORT FUNCTIONS FROM YOUR LOGIC
# -------------------------
# (Copy all your helper functions here exactly as they are)
# SAFE HELPERS
# LOAN TYPE EWS
# FINANCIAL HEALTH SCORING
# SB LABEL
# ENGINEER DATAFRAME
# (Paste everything above this Streamlit code)
# -------------------------


# -------------------------
# STREAMLIT UI
# -------------------------
st.title("📊 Financial Health (FH) & SB Rating Prediction Tool")
st.write("Upload master dataset → Train model → Upload input companies → Generate FH + SB + ML scores.")

st.sidebar.header("⚙ Settings")


# -----------------------------------------------------------
# STEP 1 — UPLOAD MASTER DATASET
# -----------------------------------------------------------
st.subheader("1️⃣ Upload Master Dataset (FY expanded)")

master_file = st.file_uploader("Upload master Excel file", type=["xlsx"])

if master_file is not None:
    df_hist = pd.read_excel(master_file, engine="openpyxl", dtype=object)
    st.success(f"Master dataset loaded — {len(df_hist)} rows")

    # ENGINEER DATA
    with st.spinner("Engineering dataset..."):
        df_eng = engineer_dataframe(df_hist)

    # FEATURE SET
    FEATURES = [
        "Debt_Equity","EBITDA_Margin","PAT_Margin",
        "DSCR","Current Ratio","ROCE (%)","ROE (%)","Credit Utilization (%)",
        "DPD_CurrentLoan","Bounced_Cheques","SMA_Flag","CRILC_Exposure_num",
        "Loan_Type_Code","Collateral_Value_num","LTV_num","Loan_Amount_num","Loan_Tenure_Months",
        "Existing_Loan_Sanctioned_num","Existing_Loan_Outstanding_num",
        "Promoter_Risk","Management_Risk","Industry_Risk","ESG_Risk",
        "Document_Quality_Score","Loan_Type_EWS"
    ]

    for f in FEATURES:
        if f not in df_eng.columns:
            df_eng[f] = np.nan

    X_fh = df_eng[FEATURES].applymap(lambda x: parse_num(x) if not pd.isna(x) else np.nan)
    feature_medians = X_fh.median(numeric_only=True)
    X_fh = X_fh.fillna(feature_medians)

    y_fh = df_eng["FH_Score"].astype(float)

    # TRAINING
    if st.button("Train Ridge Regression Model"):
        if X_fh.shape[0] >= 10 and y_fh.nunique() > 1:
            with st.spinner("Training Ridge model..."):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_fh, y_fh, test_size=0.2, random_state=42
                )

                ridge = Ridge(alpha=1.0, random_state=42)
                ridge.fit(X_train, y_train)

                st.success("Model trained successfully!")

                # SHOW METRICS
                y_pred = ridge.predict(X_test)
                st.write("### 📈 Model Performance")
                st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")
                st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.4f}")
                st.write(f"**RMSE:** {mean_squared_error(y_test, y_pred)**0.5:.4f}")

                st.session_state["ridge_model"] = ridge
                st.session_state["feature_medians"] = feature_medians

        else:
            st.error("Insufficient data to train the model.")



# -----------------------------------------------------------
# STEP 2 — UPLOAD INPUT FILE FOR PREDICTION
# -----------------------------------------------------------
st.subheader("2️⃣ Upload Input Excel for Prediction")

input_file = st.file_uploader("Upload input Excel file", type=["xlsx"], key="input")

if input_file is not None:

    if "ridge_model" not in st.session_state:
        st.error("Please train the model first.")
    else:
        ridge = st.session_state["ridge_model"]
        feature_medians = st.session_state["feature_medians"]

        df_input = pd.read_excel(input_file, engine="openpyxl", dtype=object)
        st.success(f"Input file loaded — {len(df_input)} rows")

        with st.spinner("Engineering input data..."):
            df_input_eng = engineer_dataframe(df_input)

        X_pred = df_input_eng[FEATURES].applymap(lambda x: parse_num(x) if not pd.isna(x) else np.nan)
        X_pred = X_pred.fillna(feature_medians)

        results = []

        for idx, row in df_input_eng.iterrows():
            loan_ews_val = row["Loan_Type_EWS"]
            fh_formula = row["FH_Score"]

            sb_f, sb_desc_f, sb_range_f = sb_label(fh_formula)
            rb_formula = categorize_score_numeric(fh_formula)

            fh_ml = ridge.predict(X_pred.iloc[[idx]])[0]
            sb_ml, sb_desc_ml, sb_range_ml = sb_label(fh_ml)
            rb_ml = categorize_score_numeric(fh_ml)

            results.append({
                "Company Name": row.get("Company Name",""),
                "Loan_Type_EWS": loan_ews_val,
                "FH_Score_Formula": fh_formula,
                "SB_Formula": sb_f,
                "SB_Description_Formula": sb_desc_f,
                "SB_Range_Formula": sb_range_f,
                "RiskBand_Formula": rb_formula,
                "FH_Score_Ridge": fh_ml,
                "SB_Ridge": sb_ml,
                "SB_Description_Ridge": sb_desc_ml,
                "SB_Range_Ridge": sb_range_ml,
                "RiskBand_Ridge": rb_ml
            })

        df_results = pd.DataFrame(results)

        st.subheader("📊 Prediction Results")
        st.dataframe(df_results)

        # Download button
        output = io.BytesIO()
        df_results.to_excel(output, index=False, engine="openpyxl")
        st.download_button(
            label="📥 Download Results as Excel",
            data=output.getvalue(),
            file_name="FH_Scoring_Results.xlsx",
            mime="application/vnd.ms-excel"
        )


