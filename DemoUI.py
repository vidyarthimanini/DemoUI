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

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("📊 Financial Health (FH) & SB Rating Prediction Tool")
st.write("Upload master dataset → Train model → Upload input companies → Generate FH + SB + ML scores.")

st.sidebar.header("⚙ Settings")


# -----------------------------------------------------------
# STEP 1 — UPLOAD MASTER DATASET & TRAIN MODEL
# -----------------------------------------------------------
st.subheader("1️⃣ Upload Master Dataset (FY expanded)")

master_file = st.file_uploader("Upload master Excel file", type=["xlsx"])

if master_file is not None:
    df_hist = pd.read_excel(master_file, engine="openpyxl", dtype=object)
    st.success(f"Master dataset loaded — {len(df_hist)} rows")

    # ENGINEER DATA
    with st.spinner("Engineering dataset..."):
        df_eng = engineer_dataframe(df_hist)

    # 🔥 STORE HISTORICAL ENGINEERED DATA FOR TREND ANALYSIS
    st.session_state["history_eng"] = df_eng

    # -------------------------------------------------------------------
    # FEATURE SET (INCLUDE HISTORICAL FEATURES)
    # -------------------------------------------------------------------
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
        if f not in df_eng.columns:
            df_eng[f] = np.nan

    X_fh = df_eng[FEATURES].applymap(lambda x: parse_num(x) if not pd.isna(x) else np.nan)
    feature_medians = X_fh.median(numeric_only=True)
    X_fh = X_fh.fillna(feature_medians)

    y_fh = df_eng["FH_Score"].astype(float)

    if st.button("Train Ridge Regression Model"):
        if X_fh.shape[0] >= 10 and y_fh.nunique() > 1:
            with st.spinner("Training Ridge model..."):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_fh, y_fh, test_size=0.2, random_state=42
                )

                ridge = Ridge(alpha=1.0, random_state=42)
                ridge.fit(X_train, y_train)

                st.success("Model trained successfully!")

                y_pred = ridge.predict(X_test)
                st.write("### 📈 Model Performance")
                st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")
                st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.4f}")
                st.write(f"**RMSE:** {mean_squared_error(y_test, y_pred)**0.5:.4f}")

                st.session_state["ridge_model"] = ridge
                st.session_state["feature_medians"] = feature_medians
                st.session_state["FEATURES"] = FEATURES
        else:
            st.error("Insufficient data to train the model.")



# -----------------------------------------------------------
# STEP 2 — UPLOAD INPUT FILE FOR PREDICTION
# -----------------------------------------------------------
st.subheader("2️⃣ Upload Input Excel for Prediction")

input_file = st.file_uploader("Upload input Excel file", type=["xlsx"], key="input")

if input_file is not None:

    if "ridge_model" not in st.session_state:
        st.error("Please train the model first in Step 1.")
    else:
        df_input = pd.read_excel(input_file, engine="openpyxl", dtype=object)
        st.success(f"Input file loaded — {len(df_input)} rows")

        if st.button("Calculate Results"):
            ridge = st.session_state["ridge_model"]
            feature_medians = st.session_state["feature_medians"]
            FEATURES = st.session_state["FEATURES"]

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

            # -----------------------------------------------------------
            # COMPANY DASHBOARD
            # -----------------------------------------------------------
            st.subheader("📌 Company-wise Dashboard")

            company_list = df_input_eng["Company Name"].dropna().unique()
            selected_company = st.selectbox("Select a company", company_list)

            company_results = df_results[df_results["Company Name"] == selected_company]
            latest_row = company_results.iloc[-1]

            # -----------------------------------------------------------
            # COLOR CODED SCORE CARDS
            # -----------------------------------------------------------
            st.markdown("### 🧩 Score Summary (Color Coded)")

            def get_color(risk_band):
                if risk_band == "Low": return "#2ECC71"
                if risk_band == "Moderate": return "#F1C40F"
                return "#E74C3C"

            def card(title, value, color):
                st.markdown(f"""
                <div style="
                    background-color:{color};
                    padding:15px;
                    border-radius:10px;
                    margin-bottom:10px;
                    color:white;
                    font-weight:bold;
                ">
                    {title}: {value}
                </div>
                """, unsafe_allow_html=True)

            rb_color_formula = get_color(latest_row["RiskBand_Formula"])
            rb_color_ml = get_color(latest_row["RiskBand_Ridge"])

            col1, col2 = st.columns(2)

            with col1:
                card("FH Score (Formula)", f"{latest_row['FH_Score_Formula']:.2f}", rb_color_formula)
                card("SB Rating (Formula)", latest_row["SB_Formula"], rb_color_formula)
                card("Risk Band (Formula)", latest_row["RiskBand_Formula"], rb_color_formula)

            with col2:
                card("FH Score (ML)", f"{latest_row['FH_Score_Ridge']:.2f}", rb_color_ml)
                card("SB Rating (ML)", latest_row["SB_Ridge"], rb_color_ml)
                card("Risk Band (ML)", latest_row["RiskBand_Ridge"], rb_color_ml)

            # -----------------------------------------------------------
            # ADVANCED TREND GRAPH — HISTORICAL FROM MASTER + ML PREDICTION
            # -----------------------------------------------------------
            st.markdown("### 📈 FH Score Trend (2021–2026)")

            # Load historical engineered dataset
            if "history_eng" in st.session_state:
                df_history = st.session_state["history_eng"]
            else:
                df_history = df_input_eng  # fallback

            hist_company = df_history[df_history["Company Name"] == selected_company].sort_values("FY_num")

            years = hist_company["FY_num"].tolist()
            scores = hist_company["FH_Score"].tolist()

            # Append predicted year
            next_year = max(years) + 1
            years.append(next_year)
            scores.append(latest_row["FH_Score_Ridge"])

            plot_df = pd.DataFrame({"FY": years, "FH_Score": scores})

            def score_to_color(s):
                if s >= 75: return "#2ECC71"
                elif s >= 50: return "#F1C40F"
                return "#E74C3C"

            plot_df["Color"] = plot_df["FH_Score"].apply(score_to_color)

            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(plot_df["FY"], plot_df["FH_Score"], marker="o", linewidth=2)

            for i in range(len(plot_df)):
                ax.scatter(
                    plot_df["FY"].iloc[i],
                    plot_df["FH_Score"].iloc[i],
                    color=plot_df["Color"].iloc[i],
                    s=120
                )

            ax.annotate("Predicted",
                        (next_year, latest_row["FH_Score_Ridge"]),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=9)

            ax.set_xlabel("Financial Year")
            ax.set_ylabel("FH Score")
            ax.set_title(f"FH Score Trend for {selected_company}")
            ax.grid(True, linestyle="--", alpha=0.4)

            st.pyplot(fig)

            # -----------------------------------------------------------
            # INSIGHTS
            # -----------------------------------------------------------
            st.markdown("### 📝 Insights")

            insights = []

            if latest_row["FH_Score_Ridge"] > latest_row["FH_Score_Formula"]:
                insights.append("📌 ML predicts **better financial health** than formula.")
            else:
                insights.append("📌 ML predicts **weaker financial health** than formula.")

            if latest_row["RiskBand_Ridge"] != latest_row["RiskBand_Formula"]:
                insights.append(f"🔄 Risk Band changed: **{latest_row['RiskBand_Formula']} → {latest_row['RiskBand_Ridge']}**")
            else:
                insights.append("✔ Risk Band same across formula & ML.")

            # Historical trend insight
            hist_company_sorted = hist_company.sort_values("FY_num")
            if len(hist_company_sorted) >= 2:
                if hist_company_sorted["FH_Score"].iloc[-1] > hist_company_sorted["FH_Score"].iloc[-2]:
                    insights.append("📈 Financial health improved vs last FY.")
                else:
                    insights.append("📉 Financial health declined vs last FY.")

            for i in insights:
                st.write(i)

            # -----------------------------------------------------------
            # DOWNLOAD RESULTS
            # -----------------------------------------------------------
            st.subheader("📊 Prediction Results (All Companies)")
            st.dataframe(df_results)

            output = io.BytesIO()
            df_results.to_excel(output, index=False, engine="openpyxl")

            st.download_button(
                label="📥 Download Results as Excel",
                data=output.getvalue(),
                file_name="FH_Scoring_Results.xlsx",
                mime="application/vnd.ms-excel"
            )
