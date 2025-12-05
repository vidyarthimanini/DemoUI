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

    # Pre-compute stats for driver analysis
    feature_means = X_train_full.mean()
    feature_stds = X_train_full.std().replace(0, 1)

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

    # Get latest record for selected company
    comp_mask = df_results["Company Name"] == selected_company
    comp_result = df_results[comp_mask].iloc[-1]
    fh_pred = comp_result["FH_Score_Ridge"]

    # ------------------------ SCORE CARDS ------------------------
    st.markdown("### 📌 Score Summary")

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
    #     AI MODEL FEEDBACK & SCORECARD (Big Card + SB bands)
    # ============================================================
    st.markdown("### 🧠 AI Model Feedback & Scorecard")

    sb_code, sb_text, sb_range = sb_label(fh_pred)

    score_col, band_col = st.columns([1.2, 1.8])

    with score_col:
        st.markdown(
            f"""
            <div style="
                background-color:#f5f3ff;
                padding:24px;
                border-radius:16px;
                text-align:center;
                border:1px solid #e0ddff;
            ">
                <div style="font-size:16px;font-weight:600;color:#555;">
                    Risk Score
                </div>
                <div style="font-size:40px;font-weight:800;color:#4f46e5;margin-top:4px;">
                    {fh_pred:.0f}
                </div>
                <div style="font-size:16px;font-weight:600;color:#f97373;margin-top:4px;">
                    {sb_code} - {sb_text}
                </div>
                <div style="font-size:12px;color:#777;margin-top:2px;">
                    Range {sb_range}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with band_col:
        st.markdown("**Risk Band Classification**")
        band_data = [
            ("SB1","Excellent","90-100"),
            ("SB2","Very Good","85-89"),
            ("SB3","Good","80-84"),
            ("SB4","Good","75-79"),
            ("SB5","Satisfactory","70-74"),
            ("SB6","Satisfactory","65-69"),
            ("SB7","Acceptable","60-64"),
            ("SB8","Acceptable","55-59"),
            ("SB9","Marginal","50-54"),
            ("SB10","Marginal","45-49"),
            ("SB11","Weak","40-44"),
            ("SB12","Poor","35-39"),
            ("SB13","Poor","30-34"),
            ("SB14","Very Poor","25-29"),
            ("SB15","Very Poor","20-24"),
            ("SB16","Unacceptable","0-19"),
        ]
        df_bands = pd.DataFrame(band_data, columns=["SB Code","Description","Score Range"])
        st.table(df_bands)

    # ============================================================
    #          DECISION RECOMMENDATION (Approve / Reject)
    # ============================================================
    st.markdown("### 🧾 Decision Recommendation")

    risk_band_ml = comp_result["RiskBand_Ridge"]

    if risk_band_ml == "Low":
        decision = "Approve"
        decision_color = "#16a34a"
        decision_desc = "Application meets minimum risk criteria for approval."
    elif risk_band_ml == "Moderate":
        decision = "Review"
        decision_color = "#f59e0b"
        decision_desc = "Application is borderline and requires further assessment."
    else:
        decision = "Reject"
        decision_color = "#dc2626"
        decision_desc = "Application does not meet minimum risk criteria for approval."

    st.markdown(
        f"""
        <div style="
            background-color:#fef2f2;
            border:1px solid #fecaca;
            padding:18px;
            border-radius:12px;
            max-width:600px;
        ">
            <div style="font-size:14px;color:#b91c1c;font-weight:600;">
                Decision Recommendation
            </div>
            <div style="font-size:22px;font-weight:800;color:{decision_color};margin-top:4px;">
                {decision}
            </div>
            <div style="font-size:13px;color:#7f1d1d;margin-top:4px;">
                {decision_desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    #                KEY RISK DRIVERS (Pseudo-SHAP)
    # ============================================================
   import plotly.graph_objects as go

# Top drivers: already in top_drivers (sorted by AbsImpact)
dfd = top_drivers.sort_values("Impact")

colors = ["#d62728" if v < 0 else "#2ca02c" for v in dfd["Impact"]]

fig = go.Figure()

fig.add_trace(go.Bar(
    x=dfd["Impact"],
    y=dfd["Feature"],
    orientation='h',
    marker=dict(
        color=colors,
        line=dict(color="black", width=0.5)
    ),
    text=[f"{v:.2f}" for v in dfd["Impact"]],
    textposition="auto",
))

fig.update_layout(
    title=f"<b>Top Risk Drivers – {selected_company}</b>",
    title_x=0.5,
    height=400,
    margin=dict(l=80, r=40, t=60, b=40),
    plot_bgcolor="#f8f9fa",
    paper_bgcolor="#ffffff",
    xaxis=dict(
        title="Impact on Risk Score",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="#999",
        gridcolor="#ddd"
    ),
    yaxis=dict(
        title="",
        automargin=True
    ),
)

st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    #              MODEL PERFORMANCE METRICS (TRAIN)
    # ============================================================
    st.markdown("### 🔍 Model Performance Metrics")
    
    y_pred_train = ridge.predict(X_train_full)
    model_accuracy = r2_score(y_train_full, y_pred_train)
    mae = mean_absolute_error(y_train_full, y_pred_train)
    
    # RMSE calculation for older sklearn versions
    rmse = mean_squared_error(y_train_full, y_pred_train) ** 0.5
    
    precision_rate = np.mean(np.abs(y_train_full - y_pred_train) < 5)
    auc_score = np.corrcoef(y_train_full, y_pred_train)[0, 1]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Model Accuracy (R²)", f"{model_accuracy*100:.1f}%")
    c2.metric("AUC Score (proxy)", f"{auc_score:.2f}")
    c3.metric("Precision Rate (±5 pts)", f"{precision_rate*100:.1f}%")
    
    st.caption(f"MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    

    # ============================================================
    #              RISK ASSESSMENT SUMMARY
    # ============================================================
    st.markdown("### 📋 Risk Assessment Summary")

    if len(selected_indices) > 0:
        # Use same driver_df from above if exists, else recompute quickly
        if 'driver_df' not in locals():
            feature_row = X_pred.loc[selected_idx]
            z_values = (feature_row - feature_means) / feature_stds
            impacts = z_values * ridge.coef_
            driver_df = pd.DataFrame({
                "Feature": FEATURES,
                "Impact": impacts.values
            })

        positive = driver_df[driver_df["Impact"] > 0].sort_values("Impact", ascending=False).head(3)
        negative = driver_df[driver_df["Impact"] < 0].sort_values("Impact", ascending=True).head(3)

        col_pos, col_neg = st.columns(2)

        with col_pos:
            st.write("#### ✅ Positive Factors")
            if len(positive) == 0:
                st.write("- None")
            for _, r in positive.iterrows():
                st.write(f"- **{r['Feature']}** : +{r['Impact']:.2f}")

        with col_neg:
            st.write("#### ❌ Risk Concerns")
            if len(negative) == 0:
                st.write("- None")
            for _, r in negative.iterrows():
                st.write(f"- **{r['Feature']}** : {r['Impact']:.2f}")
    else:
        st.info("No driver information available for summary.")

    # ============================================================
    #     FORMULA TREND GRAPH (Historical)
    # ============================================================
    st.markdown("### 📈 Formula FH Trend (Historical)")

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
    st.markdown("### 🤖 ML Predicted Trend (Next FY)")

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


