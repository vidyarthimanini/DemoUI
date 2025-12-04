import io, re, math, sys, warnings, os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
import joblib


warnings.filterwarnings("ignore")

# ------------------------------------------------------
# SAFE HELPERS
# ------------------------------------------------------
def safe_div(a, b, eps=1e-9):
    try: return float(a)/(float(b)+eps)
    except: return np.nan

def parse_num(x):
    if pd.isna(x): return np.nan
    s = str(x).replace(",","").replace("₹","").strip()
    s = re.sub(r"[^0-9.\-]","", s)
    if s in ["","-",".","-."]: return np.nan
    try: return float(s)
    except: return np.nan

def parse_int(x):
    v = parse_num(x)
    return int(round(v)) if v is not None and not pd.isna(v) else 0

def parse_yesno_to_num(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in ["yes","y","true","1","t"]: return 1.0
    if s in ["no","n","false","0","f"]: return 0.0
    return parse_num(s)

def dpd_from_text(x):
    if pd.isna(x): return 0
    s = str(x)
    m = re.findall(r"(\d{1,3})\s*DPD", s, flags=re.IGNORECASE)
    if m: return max(map(int,m))
    m2 = re.findall(r"(\d{1,3})", s)
    return max(map(int,m2)) if m2 else 0

def sma_flag(x):
    if pd.isna(x): return 0
    s = str(x).lower()
    if "sma-2" in s: return 2
    if "sma-1" in s: return 1
    return 0

def text_risk_score(x):
    if pd.isna(x): return 50
    s = str(x).lower()
    if any(w in s for w in ["good","clean","strong","compliant","yes","ok","clear","excellent","positive"]): return 80
    if any(w in s for w in ["bad","poor","weak","issue","non","no","fraud","concern","negative"]): return 20
    return 50

def scale(v, domain, rng):
    try:
        v = float(v)
    except:
        v = domain[1]
    if np.isnan(v): v = domain[1]
    return float(np.clip(np.interp(v, domain, rng), min(rng), max(rng)))

def score_behavior(value, good, mid, bad):
    try: v = float(value)
    except: v = mid
    if v <= good: return 100
    if v <= mid: return 70
    if v <= bad: return 40
    return 20

def categorize_score_numeric(s):
    if s >= 75: return "Low"
    if s >= 50: return "Moderate"
    return "High"

def loan_type_code(lt):
    if pd.isna(lt): return 0
    s = str(lt).strip().upper()
    if s.startswith("WC"): return 0
    if s.startswith("TL"): return 1
    if s.startswith("SL") or s.startswith("ST"): return 2
    if s.startswith("CC"): return 3
    if s.startswith("OD"): return 4
    return 0

# ------------------------------------------------------
# LOAN TYPE EWS
# ------------------------------------------------------
def compute_loan_type_ews_row(r):
    cc = parse_num(r.get("Credit Utilization (%)", np.nan))
    dpd = dpd_from_text(r.get("DPD Flags & History","0"))
    bounced = parse_int(r.get("Bounced Cheques (Count)",0))
    sma = sma_flag(r.get("SMA Classification","SMA-0"))
    crilc = parse_yesno_to_num(r.get("CRILC Exposure",0))
    ltv = parse_num(r.get("LTV Ratio",np.nan))
    coll = parse_num(r.get("Collateral Value",np.nan))

    wc = np.mean([
        score_behavior(cc if not np.isnan(cc) else 60, 40,75,95),
        score_behavior(dpd, 0,30,90),
        score_behavior(bounced, 0,1,3)
    ])

    tl = np.mean([
        score_behavior(ltv if not np.isnan(ltv) else 70, 50,70,90),
        score_behavior(dpd, 0,30,90),
        score_behavior(coll if not np.isnan(coll) else 0, 0,50,80)
    ])

    sl = np.mean([
        score_behavior(crilc if not np.isnan(crilc) else 0, 0,10,50),
        score_behavior(sma,0,1,2)
    ])

    lt = str(r.get("Loan Type","WC")).upper()
    base = wc if lt.startswith("WC") else (tl if lt.startswith("TL") else sl)
    credit_comp = np.mean([score_behavior(sma,0,1,2), score_behavior(dpd,0,30,90)])

    return float(np.clip(base*0.6 + credit_comp*0.4,0,100))

# ------------------------------------------------------
# FINANCIAL HEALTH SCORING
# ------------------------------------------------------
def compute_fh_row(r, loan_ews, turnover_history=None):
    turnover = parse_num(r.get("Turnover (₹ Crore)", np.nan))
    ebitda = parse_num(r.get("EBITDA (₹ Crore)", np.nan))
    netprofit = parse_num(r.get("Net Profit (₹ Crore)", np.nan))
    networth = parse_num(r.get("Net Worth (₹ Crore)", np.nan))
    total_debt = parse_num(r.get("Total Debt (₹ Crore)", np.nan))

    debt_equity = safe_div(total_debt, networth)
    ebitda_margin = 100 * safe_div(ebitda, turnover)
    pat_margin = 100 * safe_div(netprofit, turnover)

    dscr = parse_num(r.get("DSCR",np.nan))
    current_ratio = parse_num(r.get("Current Ratio",np.nan))
    roce = parse_num(r.get("ROCE (%)",np.nan))
    roe = parse_num(r.get("ROE (%)",np.nan))
    credit_util = parse_num(r.get("Credit Utilization (%)",np.nan))

    dpd = dpd_from_text(r.get("DPD Flags & History","0"))
    bounced = parse_int(r.get("Bounced Cheques (Count)",0))
    sma = sma_flag(r.get("SMA Classification","SMA-0"))
    overdrafts = parse_int(r.get("Overdrafts (Count)",0))
    avg_bal = parse_num(r.get("Average Bank Balance (Last 6 Months)",np.nan))

    leverage = np.mean([
        scale(debt_equity if not np.isnan(debt_equity) else 1.0, [0,1,3],[100,80,40]),
        scale(total_debt if not np.isnan(total_debt) else 10, [0,50,200],[100,70,20])
    ])

    liquidity = np.mean([
        scale(current_ratio if not np.isnan(current_ratio) else 1.0, [0.5,1,2],[40,70,100]),
        scale(credit_util if not np.isnan(credit_util) else 60, [20,60,100],[100,60,40])
    ])

    coverage = np.mean([
        scale(dscr if not np.isnan(dscr) else 1.5, [0.8,1.2,2.0],[40,70,100]),
        scale(avg_bal if not np.isnan(avg_bal) else 10, [0,20,50],[40,70,100])
    ])

    profitability = np.mean([
        scale(ebitda_margin if not np.isnan(ebitda_margin) else 10,[5,15,30],[40,70,100]),
        scale(pat_margin if not np.isnan(pat_margin) else 7,[1,8,20],[40,70,100]),
        scale(roce if not np.isnan(roce) else 10,[5,10,20],[40,70,100]),
        scale(roe if not np.isnan(roe) else 10,[5,10,20],[40,70,100])
    ])

    cashflow = np.mean([
        scale(avg_bal if not np.isnan(avg_bal) else 10,[0,20,50],[40,70,100]),
        scale(overdrafts if not np.isnan(overdrafts) else 0,[0,1,3],[100,70,40])
    ])

    growth = 50
    if turnover_history and len(turnover_history)>=2:
        try:
            growth = 100 * safe_div((turnover_history[-1]-turnover_history[-2]), turnover_history[-2])
        except:
            growth = 50

    crilc_num = parse_yesno_to_num(r.get("CRILC Exposure",0))

    contingent = np.mean([
        scale(crilc_num if not np.isnan(crilc_num) else 0,[0,5,20],[100,60,20]),
        score_behavior(sma,0,1,2)
    ])

    behaviour = np.mean([
        score_behavior(dpd,0,14,30),
        score_behavior(bounced,0,1,3)
    ])

    fraud = np.mean([
        text_risk_score(r.get("Promoter Background Check","")),
        text_risk_score(r.get("Regulatory Compliance Status",""))
    ])

    financial_health = (
        0.25*leverage + 0.20*liquidity + 0.15*coverage +
        0.20*profitability + 0.10*cashflow + 0.05*growth + 0.05*contingent
    )

    return float(np.clip(0.55*financial_health + 0.25*behaviour + 0.10*fraud + 0.10*(loan_ews if loan_ews else 50),0,100))

# ------------------------------------------------------
# SB LABEL
# ------------------------------------------------------
def sb_label(score):
    s = float(score)
    if 90<=s<=100: return ("SB1","Excellent","90-100")
    if 85<=s<=89: return ("SB2","Very Good","85-89")
    if 80<=s<=84: return ("SB3","Good","80-84")
    if 75<=s<=79: return ("SB4","Good","75-79")
    if 70<=s<=74: return ("SB5","Satisfactory","70-74")
    if 65<=s<=69: return ("SB6","Satisfactory","65-69")
    if 60<=s<=64: return ("SB7","Acceptable","60-64")
    if 55<=s<=59: return ("SB8","Acceptable","55-59")
    if 50<=s<=54: return ("SB9","Marginal","50-54")
    if 45<=s<=49: return ("SB10","Marginal","45-49")
    if 40<=s<=44: return ("SB11","Weak","40-44")
    if 35<=s<=39: return ("SB12","Poor","35-39")
    if 30<=s<=34: return ("SB13","Poor","30-34")
    if 25<=s<=29: return ("SB14","Very Poor","25-29")
    if 20<=s<=24: return ("SB15","Very Poor","20-24")
    return ("SB16","Unacceptable","0-19")

# ------------------------------------------------------
# ENGINEER DATAFRAME
# ------------------------------------------------------
def engineer_dataframe(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    num_cols = [
        "Turnover (₹ Crore)","EBITDA (₹ Crore)","Net Profit (₹ Crore)",
        "Net Worth (₹ Crore)","Total Debt (₹ Crore)","DSCR",
        "Current Ratio","ROCE (%)","ROE (%)","Credit Utilization (%)",
        "Average Bank Balance (Last 6 Months)",
        "Collateral Value","LTV Ratio","Loan Amount","Tenure (Months)"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].apply(parse_num)

    df["CRILC_Exposure_num"] = df["CRILC Exposure"].apply(parse_yesno_to_num).fillna(0) if "CRILC Exposure" in df.columns else 0
    df["Loan_Type_EWS"] = df.apply(compute_loan_type_ews_row,axis=1)

    df["Turnover_num"] = df["Turnover (₹ Crore)"].apply(parse_num) if "Turnover (₹ Crore)" in df.columns else np.nan
    df["FY_num"] = df["FY"].apply(parse_num) if "FY" in df.columns else np.nan

    fh=[]
    for _, row in df.iterrows():
        comp = row.get("Company Name",None)
        if comp and "FY_num" in df.columns:
            same = df[(df["Company Name"]==comp)&(df["FY_num"].notna())]
            turnover_history = same.sort_values("FY_num")["Turnover_num"].tolist() if len(same)>=2 else None
        else:
            turnover_history = None
        fh.append(compute_fh_row(row, row["Loan_Type_EWS"], turnover_history))
    df["FH_Score"] = fh

    df["Debt_Equity"] = df.apply(lambda r: safe_div(parse_num(r.get("Total Debt (₹ Crore)",np.nan)), parse_num(r.get("Net Worth (₹ Crore)",np.nan))), axis=1)
    df["EBITDA_Margin"] = df.apply(lambda r: 100*safe_div(parse_num(r.get("EBITDA (₹ Crore)",np.nan)), parse_num(r.get("Turnover (₹ Crore)",np.nan))), axis=1)
    df["PAT_Margin"] = df.apply(lambda r: 100*safe_div(parse_num(r.get("Net Profit (₹ Crore)",np.nan)), parse_num(r.get("Turnover (₹ Crore)",np.nan))), axis=1)

    df["DPD_CurrentLoan"] = df["DPD Flags & History"].apply(dpd_from_text)
    df["Bounced_Cheques"] = df["Bounced Cheques (Count)"].apply(parse_int) if "Bounced Cheques (Count)" in df.columns else 0
    df["SMA_Flag"] = df["SMA Classification"].apply(sma_flag)

    df["Loan_Type_Code"] = df["Loan Type"].apply(loan_type_code) if "Loan Type" in df.columns else 0
    df["Collateral_Value_num"] = df["Collateral Value"].apply(parse_num) if "Collateral Value" in df.columns else np.nan
    df["LTV_num"] = df["LTV Ratio"].apply(parse_num) if "LTV Ratio" in df.columns else np.nan
    df["Loan_Amount_num"] = df["Loan Amount"].apply(parse_num) if "Loan Amount" in df.columns else np.nan
    df["Loan_Tenure_Months"] = df["Tenure (Months)"].apply(parse_int) if "Tenure (Months)" in df.columns else np.nan

    df["Existing_Loan_Sanctioned_num"] = df["Existing Loan - Sanctioned (₹Cr)"].apply(parse_num) if "Existing Loan - Sanctioned (₹Cr)" in df.columns else np.nan
    df["Existing_Loan_Outstanding_num"] = df["Existing Loan - Outstanding (₹Cr)"].apply(parse_num) if "Existing Loan - Outstanding (₹Cr)" in df.columns else np.nan

    df["Promoter_Risk"] = df["Promoter Background Check"].apply(text_risk_score) if "Promoter Background Check" in df.columns else 50
    df["Management_Risk"] = df["Management Track Record Rating"].apply(text_risk_score) if "Management Track Record Rating" in df.columns else 50
    df["Industry_Risk"] = df["Industry Risk Outlook"].apply(text_risk_score) if "Industry Risk Outlook" in df.columns else 50
    df["ESG_Risk"] = df["ESG Compliance Risk Level"].apply(text_risk_score) if "ESG Compliance Risk Level" in df.columns else 50

    doc_cols = [c for c in df.columns if "Uploaded" in c]
    df["Document_Quality_Score"] = df[doc_cols].notna().sum(axis=1)/len(doc_cols)*100 if len(doc_cols)>0 else 50

    return df

# ------------------------------------------------------


# ------------------------------------------------------
# TRAIN RIDGE REGRESSION
# ------------------------------------------------------
df_eng = engineer_dataframe(df_hist)
print("Engineered shape:", df_eng.shape)

if "FH_Score" not in df_eng.columns:
    raise SystemExit("FH_Score missing after engineering.")

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

if X_fh.shape[0] >= 10 and y_fh.nunique() > 1:
    X_train, X_test, y_train, y_test = train_test_split(X_fh, y_fh, test_size=0.2, random_state=42)

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)

    joblib.dump({
        "model": ridge,
        "feature_medians": feature_medians.to_dict(),
        "model_type": "ridge_regression"
    }, "fh_ridge_regressor.pkl")

    print("Trained Ridge model saved as fh_ridge_regressor.pkl")

    y_pred = ridge.predict(X_test)
    print("\nMODEL PERFORMANCE")
    print("R²   :", r2_score(y_test, y_pred))
    print("MAE  :", mean_absolute_error(y_test, y_pred))
    print("RMSE :", mean_squared_error(y_test, y_pred)**0.5)
else:
    ridge = None
    print("Skipping model training — insufficient data.")

# ------------------------------------------------------
# 🎯 NEW EXCEL INPUT FOR PREDICTION (FINAL)
# ------------------------------------------------------
print("\nPlease upload the Excel file for prediction:")
uploaded_pred = files.upload()

pred_filename = list(uploaded_pred.keys())[0]
df_input = pd.read_excel(io.BytesIO(uploaded_pred[pred_filename]), engine="openpyxl", dtype=object)

print("\nRows loaded for prediction:", len(df_input))

df_input_eng = engineer_dataframe(df_input)

X_pred = df_input_eng[FEATURES].applymap(lambda x: parse_num(x) if not pd.isna(x) else np.nan)
X_pred = X_pred.fillna(feature_medians)

results = []

for idx, row in df_input_eng.iterrows():

    loan_ews_val = row["Loan_Type_EWS"]
    fh_formula = row["FH_Score"]

    sb_f, sb_desc_f, sb_range_f = sb_label(fh_formula)
    rb_formula = categorize_score_numeric(fh_formula)

    if ridge is not None:
        fh_ml = ridge.predict(X_pred.iloc[[idx]])[0]
        sb_ml, sb_desc_ml, sb_range_ml = sb_label(fh_ml)
        rb_ml = categorize_score_numeric(fh_ml)
    else:
        fh_ml = sb_ml = sb_desc_ml = sb_range_ml = rb_ml = None

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

output_file = "FH_Scoring_Results.xlsx"
df_results.to_excel(output_file, index=False)

files.download(output_file)

print("\nPrediction complete. Output saved as:", output_file)
