# ============================================================
# AI-POWERED LOAN DEFAULT PREDICTION SYSTEM  v3.0
# Machine Learning for Loan Default Prediction with Taipy
# Nigerian Naira (NGN) · Credit Risk Assessment Platform
# ============================================================
# Steps implemented: Data Collection → Preprocessing →
# Feature Engineering → Model Selection → Training →
# Hyperparameter Tuning → Evaluation → Taipy Deployment
# ============================================================

import os
import re
import warnings
import datetime
import numpy as np
import pandas as pd
import joblib
from taipy.gui import Gui, notify
from datetime import date

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD SAVED ARTEFACTS
# ============================================================

MODELS_DIR = "./models/"

def _load(filename, label):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        try:
            obj = joblib.load(path)
            print(f"[OK]   {label:<28} ← {path}")
            return obj
        except Exception as e:
            print(f"[ERR]  {label}: {e}")
            return None
    print(f"[MISS] {label:<28} not found at {path}")
    return None

# ── Load all three model files ───────────────────────────────
models_available = {
    "Logistic Regression": (
        _load("best_tuned_model.pkl", "Logistic Regression") or
        _load("logistic_model.pkl",   "Logistic Regression")
    ),
    "Random Forest": (
        _load("random_forest_model.pkl", "Random Forest") or
        _load("rf_model.pkl",            "Random Forest")
    ),
    "XGBoost": (
        _load("xgboost_model.pkl", "XGBoost") or
        _load("xgb_model.pkl",     "XGBoost")
    ),
}
models_available = {k: v for k, v in models_available.items() if v is not None}

if not models_available:
    raise RuntimeError(
        "No trained models found in ./models/\n"
        "Expected: best_tuned_model.pkl / random_forest_model.pkl / xgboost_model.pkl"
    )

scaler            = _load("scaler.pkl",            "StandardScaler")
label_encoders    = _load("label_encoders.pkl",    "Label Encoders")
selected_features = _load("selected_features.pkl", "Selected Features")

if selected_features is None:
    raise RuntimeError("selected_features.pkl is required but was not found in ./models/")

MODEL_OPTIONS      = list(models_available.keys())
current_model_name = MODEL_OPTIONS[0]
current_model      = models_available[current_model_name]
model_count        = len(MODEL_OPTIONS)
feature_count      = len(selected_features)   # pre-computed — safe in markup

# Static display strings for each model (safe for markup)
MODEL_META = {
    "Logistic Regression": {
        "short":  "LR",
        "roc":    "0.85",
        "f1":     "0.80",
        "desc":   "Baseline linear model · Fast & interpretable",
        "badge":  "BASELINE",
    },
    "Random Forest": {
        "short":  "RF",
        "roc":    "0.91",
        "f1":     "0.86",
        "desc":   "Ensemble of decision trees · Handles non-linear patterns",
        "badge":  "ENSEMBLE",
    },
    "XGBoost": {
        "short":  "XGB",
        "roc":    "0.92",
        "f1":     "0.88",
        "desc":   "Gradient boosting · Highest accuracy",
        "badge":  "TOP MODEL",
    },
}

def _meta(name, key, fallback="—"):
    return MODEL_META.get(name, {}).get(key, fallback)

print(f"\n[OK] Models loaded  : {MODEL_OPTIONS}")
print(f"[OK] Active model   : {current_model_name}")
print(f"[OK] Features       : {len(selected_features)}\n")

# ============================================================
# 2. AI INSIGHTS ENGINE
# ============================================================

class AIInsightsEngine:
    """Generates intelligent risk analysis from borrower profile."""

    # Thresholds from Nigerian credit risk guidelines
    LTI_WARN      = 40   # loan-to-income %
    LTI_HIGH      = 60
    UTIL_WARN     = 50
    UTIL_HIGH     = 70
    RATE_WARN     = 18
    INQUIRY_WARN  = 3

    @classmethod
    def analyse(cls, loan_amount, annual_income, revolving_util,
                delinquency_2yr, inquiries_6mo, interest_rate,
                open_accounts, public_records, term):
        """Return (factors_text, recommendations_text, credit_score_estimate)."""
        factors = []
        recs    = []
        score   = 100   # start at 100, deduct for each flag

        lti = (loan_amount / annual_income * 100) if annual_income > 0 else 0

        # ── Loan-to-income ──────────────────────────────────
        if lti > cls.LTI_HIGH:
            factors.append(f"Very high loan-to-income ratio: {lti:.1f}% (threshold {cls.LTI_HIGH}%)")
            recs.append("Reduce loan amount or provide additional income documentation")
            score -= 20
        elif lti > cls.LTI_WARN:
            factors.append(f"Elevated loan-to-income ratio: {lti:.1f}% (threshold {cls.LTI_WARN}%)")
            recs.append("Consider a smaller loan to improve debt serviceability")
            score -= 10

        # ── Revolving utilisation ───────────────────────────
        if revolving_util > cls.UTIL_HIGH:
            factors.append(f"Very high revolving utilisation: {revolving_util:.1f}%")
            recs.append("Paying down existing credit card balances can significantly improve score")
            score -= 20
        elif revolving_util > cls.UTIL_WARN:
            factors.append(f"Elevated revolving utilisation: {revolving_util:.1f}%")
            recs.append("Target under 30% utilisation before applying")
            score -= 10

        # ── Delinquency ─────────────────────────────────────
        if delinquency_2yr >= 2:
            factors.append(f"Multiple delinquencies: {delinquency_2yr} in past 2 years")
            recs.append("Establish 12 months of on-time payments to rebuild creditworthiness")
            score -= 25
        elif delinquency_2yr == 1:
            factors.append("1 delinquency recorded in the past 2 years")
            recs.append("Single late payment will reduce approval chances — emphasise recent positive history")
            score -= 12

        # ── Inquiries ───────────────────────────────────────
        if inquiries_6mo > cls.INQUIRY_WARN:
            factors.append(f"Multiple credit inquiries: {inquiries_6mo} in last 6 months")
            recs.append("Wait 3-6 months before new applications to reduce inquiry impact")
            score -= 10

        # ── Interest rate ───────────────────────────────────
        if interest_rate > cls.RATE_WARN:
            factors.append(f"Above-average interest rate: {interest_rate}%")
            recs.append("Shop for competitive rates — lower rate reduces default risk")
            score -= 5

        # ── Public records ──────────────────────────────────
        if public_records > 0:
            factors.append(f"Public record(s) on file: {public_records}")
            recs.append("Public records are a major red flag — seek legal clearance where possible")
            score -= 20

        # ── Thin file ───────────────────────────────────────
        if open_accounts < 2:
            factors.append("Very thin credit file (fewer than 2 open accounts)")
            recs.append("A secured credit card or small instalment loan can build credit history quickly")
            score -= 8

        # ── Long-term risk ───────────────────────────────────
        if term > 50:
            factors.append("60-month term increases exposure duration")
            recs.append("If affordable, a 36-month term reduces total interest and default exposure")
            score -= 5

        # ── Positive flags ──────────────────────────────────
        if not factors:
            factors.append("No significant risk factors identified")
        if not recs:
            recs.append("Application meets standard approval criteria — proceed with normal processing")

        score = max(0, min(100, score))
        return (
            "\n".join(f"• {f}" for f in factors),
            "\n".join(f"• {r}" for r in recs),
            score,
        )

    @staticmethod
    def approval_decision(probability):
        if probability < 20:
            return "STRONG APPROVE", "Low risk profile — Recommend expedited processing", "decision-green"
        elif probability < 35:
            return "APPROVE",        "Standard approval with regular account monitoring", "decision-green"
        elif probability < 50:
            return "CONSIDER",       "Review required — Request additional documentation or collateral", "decision-amber"
        elif probability < 70:
            return "CAUTION",        "High risk — Recommend reduced amount or risk-adjusted rate", "decision-orange"
        else:
            return "REJECT",         "Exceeds risk tolerance — Decline or escalate to senior underwriter", "decision-red"

    @staticmethod
    def risk_band(probability):
        if probability < 20:  return "LOW",      "risk-low"
        if probability < 35:  return "MODERATE", "risk-moderate"
        if probability < 50:  return "ELEVATED", "risk-elevated"
        if probability < 70:  return "HIGH",     "risk-high"
        return                       "CRITICAL", "risk-critical"

# ============================================================
# 3. PREPROCESSING  (mirrors notebook Steps 3 & 4 exactly)
# ============================================================

GRADE_ORDER = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}


def encode_employment_duration(s):
    if pd.isna(s) or str(s).strip() == "":
        return 0
    s = str(s).lower()
    nums = re.findall(r'\d+', s)
    if not nums:
        return 0
    v = int(nums[0])
    if 'month' in s:
        v = v / 12
    if '+' in s and v == 10:
        v = 15
    return float(v)


def preprocess_input(raw: dict, model_name: str = "") -> pd.DataFrame:
    df = pd.DataFrame([raw])

    # Grade ordinal (A=1…G=7)
    df["Grade_Encoded"] = df["Grade"].map(GRADE_ORDER).fillna(4)

    # Sub Grade — extract letter, ordinal-encode
    sg = str(df["Sub Grade"].iloc[0])
    df["Sub_Grade_Letter_Encoded"] = GRADE_ORDER.get(sg[0].upper(), 4)
    df.drop(columns=["Grade", "Sub Grade"], inplace=True, errors="ignore")

    # Employment duration → numeric years
    if "Employment Duration" in df.columns:
        df["Employment Duration"] = df["Employment Duration"].apply(encode_employment_duration)

    # Label-encode remaining categoricals
    if label_encoders:
        for col, le in label_encoders.items():
            if col in df.columns:
                val = str(df[col].iloc[0])
                known = list(le.classes_)
                df[col] = le.transform([val if val in known else known[0]])

    # Coerce all to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Helper
    def g(col, default=0):
        return float(df[col].iloc[0]) if col in df.columns else float(default)

    la  = g("Loan Amount");              fa  = g("Funded Amount")
    fai = g("Funded Amount Investor");   ir  = g("Interest Rate")
    trm = g("Term", 36);                 ai  = g("Annual Income", 1)
    dq  = g("Delinquency - two years");  pr  = g("Public Record")
    rb  = g("Revolving Balance");        rcl = g("Total Revolving Credit Limit", 1)
    inq = g("Inquires - six months");    ta  = g("Total Accounts", 1)
    tcb = g("Total Current Balance");    tca = g("Total Collection Amount")

    # 10 derived features (Step 4)
    df["Loan_to_Income_derived"]        = la  / (ai  + 1)
    df["Funded_Ratio_derived"]          = fa  / (la  + 1)
    df["Investor_Funded_Ratio_derived"] = fai / (fa  + 1)
    df["Rate_Squared_derived"]          = ir  ** 2
    df["Is_Long_Term_derived"]          = int(trm > 50)
    df["Has_Delinquency_derived"]       = int(dq  > 0)
    df["Has_Public_Record_derived"]     = int(pr  > 0)
    df["Revolving_Util_derived"]        = rb  / (rcl + 1)
    df["Inquiries_per_Acct_derived"]    = inq / (ta  + 1)
    df["Total_Debt_derived"]            = tca + tcb + rb

    # Align to training feature set
    for feat in selected_features:
        if feat not in df.columns:
            df[feat] = 0
    df = df[selected_features]

    # Scale for linear model only
    if scaler and "Logistic" in str(model_name):
        arr = scaler.transform(df)
        df = pd.DataFrame(arr, columns=df.columns)

    return df

# ============================================================
# 4. INPUT VALIDATION
# ============================================================

def validate_inputs(state):
    if state.loan_amount <= 0:
        return False, "Loan amount must be greater than zero"
    if state.loan_amount > 100_000_000_000:
        return False, "Loan amount exceeds NGN 100 billion maximum"
    if not (0 < state.interest_rate <= 50):
        return False, "Interest rate must be between 0.1% and 50%"
    if state.annual_income <= 0:
        return False, "Annual income must be greater than zero"
    if not (0 <= state.revolving_util <= 100):
        return False, "Revolving utilisation must be 0–100%"
    if state.funded_amount > state.loan_amount * 1.05:
        return False, "Funded amount cannot exceed loan amount by more than 5%"
    return True, ""

# ============================================================
# 5. STATE
# ============================================================

# Model
selected_model = current_model_name
model_options  = MODEL_OPTIONS

# Model card display — one static string per slot
m0_name  = MODEL_OPTIONS[0] if len(MODEL_OPTIONS) > 0 else "—"
m0_badge = _meta(m0_name, "badge")
m0_roc   = _meta(m0_name, "roc")
m0_f1    = _meta(m0_name, "f1")
m0_desc  = _meta(m0_name, "desc")

m1_name  = MODEL_OPTIONS[1] if len(MODEL_OPTIONS) > 1 else "—"
m1_badge = _meta(m1_name, "badge")
m1_roc   = _meta(m1_name, "roc")
m1_f1    = _meta(m1_name, "f1")
m1_desc  = _meta(m1_name, "desc")

m2_name  = MODEL_OPTIONS[2] if len(MODEL_OPTIONS) > 2 else "—"
m2_badge = _meta(m2_name, "badge")
m2_roc   = _meta(m2_name, "roc")
m2_f1    = _meta(m2_name, "f1")
m2_desc  = _meta(m2_name, "desc")

show_m1  = len(MODEL_OPTIONS) > 1
show_m2  = len(MODEL_OPTIONS) > 2

# Date
assessment_date = date.today()

# Loan details (NGN)
loan_amount       = 15_000_000.0
funded_amount     = 15_000_000.0
funded_amount_inv = 14_500_000.0
interest_rate     = 12.5
term              = 36
annual_income     = 60_000_000.0

# Borrower profile
grade               = "B"
sub_grade           = "B3"
home_ownership      = "MORTGAGE"
employment_duration = "5 years"
verification_status = "Verified"
purpose             = "debt_consolidation"
batch_enrolled      = "BAT901476"

# Credit history
delinquency_2yr       = 0
inquiries_6mo         = 1
open_accounts         = 10
public_records        = 0
revolving_balance     = 8_000_000.0
revolving_util        = 55.0
total_accounts        = 25
total_revolving_limit = 20_000_000.0
total_current_balance = 50_000_000.0
total_collection_amt  = 0.0

# Result state
show_result            = False
prediction_label       = ""
prediction_probability = 0.0
prob_bar               = 0.0
risk_level             = ""
risk_css               = "risk-low"
approval_suggestion    = ""
approval_decision_css  = "decision-green"
approval_message       = ""
credit_score_est       = 0
risk_factors_text      = "Run an analysis to see risk factors"
recommendations_text   = "Run an analysis to see recommendations"

# Formatted display (built in callback)
loan_display          = "NGN 15,000,000"
income_display        = "NGN 60,000,000"
revolving_display     = "NGN 8,000,000"
date_display          = date.today().strftime("%d %b %Y")

# Multi-model comparison (filled after predict)
show_comparison       = False
comp_lr_prob          = 0.0
comp_rf_prob          = 0.0
comp_xgb_prob         = 0.0
comp_lr_label         = "—"
comp_rf_label         = "—"
comp_xgb_label        = "—"
comp_lr_available     = "Logistic Regression" in models_available
comp_rf_available     = "Random Forest"       in models_available
comp_xgb_available    = "XGBoost"             in models_available

# Session analytics
total_predictions  = 0
average_risk_score = 0.0
high_risk_count    = 0
low_risk_count     = 0
session_start      = datetime.datetime.now().strftime("%H:%M")

# History
prediction_history = []
show_history       = False

# Dropdown options
grade_options        = ["A", "B", "C", "D", "E", "F", "G"]
sub_grade_options    = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
home_options         = ["MORTGAGE", "RENT", "OWN", "OTHER"]
term_options         = [36, 60]
verification_options = ["Verified", "Source Verified", "Not Verified"]
purpose_options      = [
    "debt_consolidation", "credit_card", "home_improvement", "other",
    "major_purchase", "small_business", "car", "wedding", "medical",
    "moving", "vacation", "house", "educational", "renewable_energy",
]

# ============================================================
# 6. CALLBACKS
# ============================================================

def on_model_change(state, var, val):
    if val in models_available:
        state.selected_model = val
        state.show_result    = False
        state.show_comparison = False
        notify(state, "info", f"Model switched to {val}")
    else:
        notify(state, "error", f"'{val}' is not available")


def _run_single(model, model_name, raw):
    """Run one model, return (pred, probability %)."""
    X    = preprocess_input(raw, model_name)
    pred = model.predict(X)[0]
    prob = (float(model.predict_proba(X)[0][1]) * 100
            if hasattr(model, "predict_proba") else (50.0 if pred == 1 else 10.0))
    return int(pred), round(prob, 1)


def predict(state):
    valid, err = validate_inputs(state)
    if not valid:
        notify(state, "error", err)
        return

    state.show_result     = False
    state.show_comparison = False

    raw = {
        "Loan Amount":                  state.loan_amount,
        "Funded Amount":                state.funded_amount,
        "Funded Amount Investor":       state.funded_amount_inv,
        "Interest Rate":                state.interest_rate,
        "Term":                         state.term,
        "Annual Income":                state.annual_income,
        "Grade":                        state.grade,
        "Sub Grade":                    state.sub_grade,
        "Home Ownership":               state.home_ownership,
        "Employment Duration":          state.employment_duration,
        "Verification Status":          state.verification_status,
        "Purpose":                      state.purpose,
        "Batch Enrolled":               state.batch_enrolled,
        "Delinquency - two years":      state.delinquency_2yr,
        "Inquires - six months":        state.inquiries_6mo,
        "Open Accounts":                state.open_accounts,
        "Public Record":                state.public_records,
        "Revolving Balance":            state.revolving_balance,
        "Revolving Utilities":          state.revolving_util,
        "Total Accounts":               state.total_accounts,
        "Total Revolving Credit Limit": state.total_revolving_limit,
        "Total Current Balance":        state.total_current_balance,
        "Total Collection Amount":      state.total_collection_amt,
    }

    try:
        active_model = models_available[state.selected_model]
        pred, default_prob = _run_single(active_model, state.selected_model, raw)

        # ── Primary result ───────────────────────────────────
        state.prediction_probability = default_prob
        state.prob_bar               = default_prob

        rl, rl_css = AIInsightsEngine.risk_band(default_prob)
        state.risk_level  = rl
        state.risk_css    = rl_css

        if pred == 1:
            state.prediction_label = "HIGH RISK  —  Likely to Default"
            state.high_risk_count += 1
        else:
            state.prediction_label = "LOW RISK  —  Likely to Repay"
            state.low_risk_count  += 1

        dec, msg, dec_css = AIInsightsEngine.approval_decision(default_prob)
        state.approval_suggestion   = dec
        state.approval_message      = msg
        state.approval_decision_css = dec_css

        # ── AI insights ──────────────────────────────────────
        factors_txt, recs_txt, c_score = AIInsightsEngine.analyse(
            state.loan_amount, state.annual_income, state.revolving_util,
            state.delinquency_2yr, state.inquiries_6mo, state.interest_rate,
            state.open_accounts, state.public_records, state.term,
        )
        state.risk_factors_text    = factors_txt
        state.recommendations_text = recs_txt
        state.credit_score_est     = c_score

        # ── Formatted display ────────────────────────────────
        state.loan_display      = f"NGN {state.loan_amount:,.0f}"
        state.income_display    = f"NGN {state.annual_income:,.0f}"
        state.revolving_display = f"NGN {state.revolving_balance:,.0f}"
        state.date_display      = state.assessment_date.strftime("%d %b %Y")

        # ── Multi-model comparison ────────────────────────────
        for mname, mvar_prob, mvar_label in [
            ("Logistic Regression", "comp_lr_prob", "comp_lr_label"),
            ("Random Forest",       "comp_rf_prob", "comp_rf_label"),
            ("XGBoost",             "comp_xgb_prob","comp_xgb_label"),
        ]:
            if mname in models_available:
                _, p = _run_single(models_available[mname], mname, raw)
                setattr(state, mvar_prob,  p)
                setattr(state, mvar_label, f"{p}%")
            else:
                setattr(state, mvar_prob,  0.0)
                setattr(state, mvar_label, "N/A")

        state.show_comparison = len(models_available) > 1

        # ── Analytics ────────────────────────────────────────
        state.total_predictions += 1
        state.average_risk_score = round(
            (state.average_risk_score * (state.total_predictions - 1) + default_prob)
            / state.total_predictions, 1
        )

        # ── History ──────────────────────────────────────────
        entry = {
            "Date":     state.assessment_date.strftime("%Y-%m-%d"),
            "Time":     datetime.datetime.now().strftime("%H:%M:%S"),
            "Model":    state.selected_model,
            "Loan":     f"NGN {state.loan_amount:,.0f}",
            "Rate":     f"{state.interest_rate}%",
            "Grade":    f"{state.grade}/{state.sub_grade}",
            "Risk %":   f"{default_prob}%",
            "Decision": dec,
        }
        state.prediction_history = ([entry] + state.prediction_history)[:20]

        state.show_result = True
        notify(state, "success", f"{state.selected_model} · Default Risk: {default_prob}%  →  {dec}")

    except Exception as exc:
        notify(state, "error", f"Prediction failed: {exc}")
        import traceback; traceback.print_exc()


def reset_form(state):
    state.assessment_date     = date.today()
    state.loan_amount         = 15_000_000.0
    state.funded_amount       = 15_000_000.0
    state.funded_amount_inv   = 14_500_000.0
    state.interest_rate       = 12.5
    state.term                = 36
    state.annual_income       = 60_000_000.0
    state.grade               = "B"
    state.sub_grade           = "B3"
    state.home_ownership      = "MORTGAGE"
    state.employment_duration = "5 years"
    state.verification_status = "Verified"
    state.purpose             = "debt_consolidation"
    state.batch_enrolled      = "BAT901476"
    state.delinquency_2yr     = 0
    state.inquiries_6mo       = 1
    state.open_accounts       = 10
    state.public_records      = 0
    state.revolving_balance   = 8_000_000.0
    state.revolving_util      = 55.0
    state.total_accounts      = 25
    state.total_revolving_limit = 20_000_000.0
    state.total_current_balance = 50_000_000.0
    state.total_collection_amt  = 0.0
    state.show_result           = False
    state.show_comparison       = False
    state.prediction_label      = ""
    state.prediction_probability= 0.0
    state.prob_bar              = 0.0
    state.risk_level            = ""
    state.risk_css              = "risk-low"
    state.approval_suggestion   = ""
    state.approval_message      = ""
    state.credit_score_est      = 0
    state.loan_display          = "NGN 15,000,000"
    state.income_display        = "NGN 60,000,000"
    state.revolving_display     = "NGN 8,000,000"
    state.risk_factors_text     = "Run an analysis to see risk factors"
    state.recommendations_text  = "Run an analysis to see recommendations"
    notify(state, "info", "Form reset — ready for new assessment")


def toggle_history(state):
    state.show_history = not state.show_history

# ============================================================
# 7. PAGE  — JSX-safe Taipy markdown
# ============================================================

page = """
<|part|class_name=app-shell|

<|part|class_name=page-header|
<|part|class_name=header-inner|
<|part|class_name=header-left|
**AI LOAN DEFAULT PREDICTION SYSTEM**

Machine Learning for Credit Risk Assessment · Nigerian Naira (NGN)
|>
<|part|class_name=header-right|
Steps: Data → Preprocessing → Feature Engineering → Model → Tuning → Evaluation → Deployment
|>
|>
|>

<|part|class_name=page-body|

<|part|class_name=kpi-strip|

<|part|class_name=kpi-tile|
TOTAL ASSESSMENTS
<|{total_predictions}|text|class_name=kpi-val|>
|>

<|part|class_name=kpi-tile|
AVG RISK SCORE
<|{average_risk_score}|text|class_name=kpi-val|>%
|>

<|part|class_name=kpi-tile kpi-danger|
HIGH RISK CASES
<|{high_risk_count}|text|class_name=kpi-val|>
|>

<|part|class_name=kpi-tile kpi-safe|
LOW RISK CASES
<|{low_risk_count}|text|class_name=kpi-val|>
|>

<|part|class_name=kpi-tile kpi-ready|
SESSION START
<|{session_start}|text|class_name=kpi-val|>
|>

|>

<|part|class_name=section-card|

### Step 4 — Model Selection

Select the algorithm to use for this assessment. Each model was trained on 70,000+ loan records and tuned with cross-validation.

<|layout|columns=1 1 1|gap=16px|class_name=model-cards|

<|part|class_name=model-tile|
<|{m0_badge}|text|class_name=model-badge|>

<|{m0_name}|text|class_name=model-title|>

<|{m0_desc}|text|class_name=model-desc|>

ROC-AUC **<|{m0_roc}|text|>** · F1 **<|{m0_f1}|text|>**
|>

<|part|render={show_m1}|class_name=model-tile|
<|{m1_badge}|text|class_name=model-badge model-badge-teal|>

<|{m1_name}|text|class_name=model-title|>

<|{m1_desc}|text|class_name=model-desc|>

ROC-AUC **<|{m1_roc}|text|>** · F1 **<|{m1_f1}|text|>**
|>

<|part|render={show_m2}|class_name=model-tile model-tile-gold|
<|{m2_badge}|text|class_name=model-badge model-badge-gold|>

<|{m2_name}|text|class_name=model-title|>

<|{m2_desc}|text|class_name=model-desc|>

ROC-AUC **<|{m2_roc}|text|>** · F1 **<|{m2_f1}|text|>**
|>

|>

<|layout|columns=2 3|gap=20px|class_name=selector-row|

<|part|class_name=field-wrap|
Active Model for This Assessment
<|{selected_model}|selector|lov={model_options}|on_change=on_model_change|dropdown|class_name=inp|>
|>

<|part|class_name=active-model-pill|
Running: **<|{selected_model}|text|>** · <|{model_count}|text|> model(s) loaded · Features: **<|{feature_count}|text|>**
|>

|>
|>

<|part|class_name=section-card|

### Steps 1–3 — Loan Application Details (NGN)

<|layout|columns=1 1 1|gap=18px|class_name=field-grid|

<|part|class_name=field-wrap|
Assessment Date
<|{assessment_date}|date|class_name=inp|>
|>

<|part|class_name=field-wrap|
Loan Amount (NGN)
<|{loan_amount}|input|type=number|min=1|max=100000000000|step=100000|class_name=inp|>
|>

<|part|class_name=field-wrap|
Funded Amount (NGN)
<|{funded_amount}|input|type=number|min=0|step=100000|class_name=inp|>
|>

<|part|class_name=field-wrap|
Funded by Investors (NGN)
<|{funded_amount_inv}|input|type=number|min=0|step=100000|class_name=inp|>
|>

<|part|class_name=field-wrap|
Interest Rate (%)
<|{interest_rate}|input|type=number|min=0.1|max=50|step=0.1|class_name=inp|>
|>

<|part|class_name=field-wrap|
Loan Term (months)
<|{term}|selector|lov={term_options}|dropdown|class_name=inp|>
|>

<|part|class_name=field-wrap|
Annual Income (NGN)
<|{annual_income}|input|type=number|min=1|step=100000|class_name=inp|>
|>

|>
|>

<|part|class_name=section-card|

### Borrower Profile

<|layout|columns=1 1 1|gap=18px|class_name=field-grid|

<|part|class_name=field-wrap|
Credit Grade
<|{grade}|selector|lov={grade_options}|dropdown|class_name=inp|>
|>

<|part|class_name=field-wrap|
Sub Grade
<|{sub_grade}|selector|lov={sub_grade_options}|dropdown|class_name=inp|>
|>

<|part|class_name=field-wrap|
Home Ownership
<|{home_ownership}|selector|lov={home_options}|dropdown|class_name=inp|>
|>

<|part|class_name=field-wrap|
Employment Duration
<|{employment_duration}|input|class_name=inp|>
|>

<|part|class_name=field-wrap|
Verification Status
<|{verification_status}|selector|lov={verification_options}|dropdown|class_name=inp|>
|>

<|part|class_name=field-wrap|
Loan Purpose
<|{purpose}|selector|lov={purpose_options}|dropdown|class_name=inp|>
|>

|>
|>

<|part|class_name=section-card|

### Credit History

<|layout|columns=1 1 1|gap=18px|class_name=field-grid|

<|part|class_name=field-wrap|
Delinquencies (last 2 yrs)
<|{delinquency_2yr}|input|type=number|min=0|max=20|step=1|class_name=inp|>
|>

<|part|class_name=field-wrap|
Inquiries (last 6 months)
<|{inquiries_6mo}|input|type=number|min=0|max=30|step=1|class_name=inp|>
|>

<|part|class_name=field-wrap|
Open Accounts
<|{open_accounts}|input|type=number|min=0|max=60|step=1|class_name=inp|>
|>

<|part|class_name=field-wrap|
Public Records
<|{public_records}|input|type=number|min=0|max=10|step=1|class_name=inp|>
|>

<|part|class_name=field-wrap|
Revolving Balance (NGN)
<|{revolving_balance}|input|type=number|min=0|step=10000|class_name=inp|>
|>

<|part|class_name=field-wrap|
Revolving Utilisation (%)
<|{revolving_util}|input|type=number|min=0|max=100|step=0.1|class_name=inp|>
|>

<|part|class_name=field-wrap|
Total Accounts
<|{total_accounts}|input|type=number|min=0|max=100|step=1|class_name=inp|>
|>

<|part|class_name=field-wrap|
Total Revolving Limit (NGN)
<|{total_revolving_limit}|input|type=number|min=0|step=10000|class_name=inp|>
|>

<|part|class_name=field-wrap|
Total Current Balance (NGN)
<|{total_current_balance}|input|type=number|min=0|step=10000|class_name=inp|>
|>

|>
|>

<|part|class_name=action-bar|
<|Analyse with AI|button|on_action=predict|class_name=btn-analyse|>
<|Reset Form|button|on_action=reset_form|class_name=btn-reset|>
<|Assessment History|button|on_action=toggle_history|class_name=btn-history|>
|>

<|part|render={show_result}|class_name=results-shell|

### Step 8 — AI Analysis Results

<|layout|columns=5 4|gap=24px|

<|part|class_name=verdict-panel|

UNDERWRITING DECISION

<|{approval_suggestion}|text|class_name={approval_decision_css}|>

<|{prediction_label}|text|class_name=verdict-label|>

DEFAULT PROBABILITY

<|{prediction_probability}|text|class_name=prob-big|>%

<|{prob_bar}|indicator|min=0|max=100|class_name=prob-indicator|>

<|layout|columns=1 1|gap=12px|

<|part|class_name=badge-wrap|
RISK BAND
<|{risk_level}|text|class_name={risk_css}|>
|>

<|part|class_name=badge-wrap|
CREDIT SCORE EST.
<|{credit_score_est}|text|class_name=score-val|> / 100
|>

|>

<|{approval_message}|text|class_name=approval-msg|>

|>

<|part|class_name=summary-panel|

LOAN SUMMARY

<|layout|columns=1 1|gap=0px|class_name=summary-grid|

<|part|class_name=sum-row|
Loan Amount
<|{loan_display}|text|class_name=sum-val|>
|>

<|part|class_name=sum-row|
Annual Income
<|{income_display}|text|class_name=sum-val|>
|>

<|part|class_name=sum-row|
Interest Rate
<|{interest_rate}|text|class_name=sum-val|>%
|>

<|part|class_name=sum-row|
Term
<|{term}|text|class_name=sum-val|> months
|>

<|part|class_name=sum-row|
Grade / Sub Grade
<|{grade}|text|class_name=sum-val|> / <|{sub_grade}|text|class_name=sum-val|>
|>

<|part|class_name=sum-row|
Purpose
<|{purpose}|text|class_name=sum-val|>
|>

<|part|class_name=sum-row|
Home Ownership
<|{home_ownership}|text|class_name=sum-val|>
|>

<|part|class_name=sum-row|
Revolving Balance
<|{revolving_display}|text|class_name=sum-val|>
|>

<|part|class_name=sum-row|
Assessment Date
<|{date_display}|text|class_name=sum-val|>
|>

<|part|class_name=sum-row|
Model Used
<|{selected_model}|text|class_name=sum-val sum-model|>
|>

|>
|>

|>

<|layout|columns=1 1|gap=20px|class_name=insights-grid|

<|part|class_name=insight-card|
RISK FACTORS IDENTIFIED
<|{risk_factors_text}|text|class_name=insight-body|>
|>

<|part|class_name=insight-card insight-card-green|
AI RECOMMENDATIONS
<|{recommendations_text}|text|class_name=insight-body|>
|>

|>

<|part|render={show_comparison}|class_name=comparison-card|

MODEL COMPARISON — Same Application Across All Models

<|layout|columns=1 1 1|gap=16px|class_name=comp-grid|

<|part|render={comp_lr_available}|class_name=comp-tile|
Logistic Regression
<|{comp_lr_prob}|text|class_name=comp-prob|>%
<|{comp_lr_label}|text|class_name=comp-badge|>
Baseline · Interpretable
|>

<|part|render={comp_rf_available}|class_name=comp-tile|
Random Forest
<|{comp_rf_prob}|text|class_name=comp-prob|>%
<|{comp_rf_label}|text|class_name=comp-badge|>
Ensemble · Robust
|>

<|part|render={comp_xgb_available}|class_name=comp-tile comp-tile-gold|
XGBoost
<|{comp_xgb_prob}|text|class_name=comp-prob|>%
<|{comp_xgb_label}|text|class_name=comp-badge|>
Gradient Boost · Highest Accuracy
|>

|>
|>

*Analysis by **<|{selected_model}|text|>** · Trained on 70,000+ records · For analytical purposes — not financial advice*

|>

<|part|render={show_history}|class_name=history-card|

### Assessment History (last 20)

<|{prediction_history}|table|width=100%|class_name=history-table|>

|>

|>

<|part|class_name=page-footer|
AI Loan Default Prediction System v3.0 · Steps 1–8 Complete · Nigerian Naira (NGN) · Taipy Deployment
|>

|>
"""

# ============================================================
# 8. CSS
# ============================================================

CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    background: #080F1A;
    color: #CBD5E1;
    font-size: 14px;
    line-height: 1.65;
}

/* ── Shell ── */
.app-shell { min-height: 100vh; display: flex; flex-direction: column; }

/* ── Header ── */
.page-header {
    background: linear-gradient(120deg, #0A1628 0%, #0D47A1 60%, #1565C0 100%);
    border-bottom: 1px solid #1E3A5F;
}

.header-inner {
    max-width: 1160px;
    margin: 0 auto;
    padding: 18px 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}

.header-left strong {
    display: block;
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: 2px;
    color: #fff;
}

.header-right p, .header-right span {
    font-size: 0.7rem;
    color: #90CAF9;
    letter-spacing: 0.5px;
}

/* ── Body ── */
.page-body {
    flex: 1;
    max-width: 1160px;
    margin: 0 auto;
    width: 100%;
    padding: 24px 20px 60px;
}

/* ── KPI strip ── */
.kpi-strip {
    display: flex;
    gap: 14px;
    margin-bottom: 22px;
    flex-wrap: wrap;
}

.kpi-tile {
    flex: 1;
    min-width: 130px;
    background: #111C2D;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: #475569;
}

.kpi-tile.kpi-danger  { border-color: #7F1D1D; }
.kpi-tile.kpi-safe    { border-color: #14532D; }
.kpi-tile.kpi-ready   { border-color: #1E3A5F; }

.kpi-val {
    display: block !important;
    font-size: 2.1rem !important;
    font-weight: 900 !important;
    color: #42A5F5 !important;
    line-height: 1.15 !important;
    margin-top: 5px;
    letter-spacing: -1px;
}

.kpi-danger .kpi-val  { color: #F87171 !important; }
.kpi-safe   .kpi-val  { color: #4ADE80 !important; }

/* ── Section cards ── */
.section-card {
    background: #111C2D;
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 18px;
}

.section-card h3 {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #42A5F5;
    margin-bottom: 6px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1E3A5F;
}

.section-card > p, .section-card > span {
    font-size: 0.82rem;
    color: #64748B;
    margin-bottom: 16px;
    display: block;
}

/* ── Model cards ── */
.model-cards { margin-bottom: 20px; }

.model-tile {
    background: #0D1F3C;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 18px 20px;
    position: relative;
    transition: border-color 0.2s;
}

.model-tile:hover { border-color: #42A5F5; }

.model-tile-gold { border-color: #78350F; }
.model-tile-gold:hover { border-color: #F59E0B; }

.model-badge {
    display: inline-block !important;
    font-size: 0.58rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    background: #1E3A5F;
    color: #42A5F5 !important;
    padding: 2px 9px;
    border-radius: 20px;
    margin-bottom: 8px;
}

.model-badge-teal { background: #134E4A; color: #2DD4BF !important; }
.model-badge-gold { background: #451A03; color: #F59E0B !important; }

.model-title {
    display: block !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #E2E8F0 !important;
    margin-bottom: 6px;
}

.model-desc {
    display: block !important;
    font-size: 0.78rem !important;
    color: #64748B !important;
    margin-bottom: 10px;
}

/* ── Selector row ── */
.selector-row { margin-top: 4px; }

.active-model-pill {
    background: #0A1628;
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #64748B;
    display: flex;
    align-items: center;
}

/* ── Fields ── */
.field-grid { gap: 18px; }

.field-wrap {
    display: flex;
    flex-direction: column;
    gap: 5px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.7px;
    text-transform: uppercase;
    color: #475569;
}

.inp input,
.inp .MuiInputBase-root,
.inp select {
    background: #0A1628 !important;
    border: 1.5px solid #1E3A5F !important;
    border-radius: 7px !important;
    padding: 9px 13px !important;
    font-size: 0.88rem !important;
    color: #E2E8F0 !important;
    width: 100% !important;
    transition: border-color 0.18s, box-shadow 0.18s;
}

.inp input:focus,
.inp .MuiInputBase-root.Mui-focused {
    border-color: #42A5F5 !important;
    box-shadow: 0 0 0 3px rgba(66,165,245,0.12) !important;
    outline: none;
}

/* ── Action bar ── */
.action-bar {
    display: flex;
    gap: 12px;
    margin: 22px 0 6px;
    flex-wrap: wrap;
}

.btn-analyse button {
    background: linear-gradient(135deg, #1565C0, #0D47A1) !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    padding: 13px 40px !important;
    border-radius: 8px !important;
    border: none !important;
    cursor: pointer !important;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 20px rgba(21,101,192,0.45) !important;
    transition: transform 0.15s, box-shadow 0.15s;
}

.btn-analyse button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(21,101,192,0.55) !important;
}

.btn-reset button {
    background: transparent !important;
    color: #64748B !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 13px 26px !important;
    border-radius: 8px !important;
    border: 1.5px solid #1E3A5F !important;
    cursor: pointer !important;
    transition: border-color 0.2s, color 0.2s;
}

.btn-reset button:hover { border-color: #42A5F5 !important; color: #E2E8F0 !important; }

.btn-history button {
    background: transparent !important;
    color: #42A5F5 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 13px 26px !important;
    border-radius: 8px !important;
    border: 1.5px solid #1E3A5F !important;
    cursor: pointer !important;
    transition: background 0.2s;
}

.btn-history button:hover { background: #111C2D !important; }

/* ── Results shell ── */
.results-shell {
    background: #111C2D;
    border: 1px solid #1E3A5F;
    border-radius: 14px;
    padding: 26px;
    margin-top: 6px;
    animation: fadeUp 0.4s ease;
}

.results-shell h3 {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #42A5F5;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1E3A5F;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Verdict panel ── */
.verdict-panel {
    background: #0A1628;
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 22px 24px;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #475569;
    line-height: 2.6;
}

/* Decision colours */
.decision-green  { font-size: 1.4rem !important; font-weight: 900 !important; color: #4ADE80 !important; letter-spacing: 1px; }
.decision-amber  { font-size: 1.4rem !important; font-weight: 900 !important; color: #FBBF24 !important; letter-spacing: 1px; }
.decision-orange { font-size: 1.4rem !important; font-weight: 900 !important; color: #FB923C !important; letter-spacing: 1px; }
.decision-red    { font-size: 1.4rem !important; font-weight: 900 !important; color: #F87171 !important; letter-spacing: 1px; }

.verdict-label {
    display: block !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #E2E8F0 !important;
    text-transform: none;
    letter-spacing: 0.2px;
    margin-bottom: 4px;
}

.prob-big {
    display: inline-block !important;
    font-size: 3.8rem !important;
    font-weight: 900 !important;
    color: #42A5F5 !important;
    line-height: 1 !important;
    letter-spacing: -2px;
}

/* Risk band colours */
.risk-low      { color: #4ADE80 !important; font-size: 1.1rem !important; font-weight: 800 !important; }
.risk-moderate { color: #A3E635 !important; font-size: 1.1rem !important; font-weight: 800 !important; }
.risk-elevated { color: #FBBF24 !important; font-size: 1.1rem !important; font-weight: 800 !important; }
.risk-high     { color: #FB923C !important; font-size: 1.1rem !important; font-weight: 800 !important; }
.risk-critical { color: #F87171 !important; font-size: 1.1rem !important; font-weight: 800 !important; }

.badge-wrap {
    background: #111C2D;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.62rem;
}

.score-val {
    display: block !important;
    font-size: 1.5rem !important;
    font-weight: 900 !important;
    color: #42A5F5 !important;
}

.approval-msg {
    display: block !important;
    font-size: 0.8rem !important;
    color: #94A3B8 !important;
    font-weight: 400 !important;
    text-transform: none;
    letter-spacing: 0;
    margin-top: 8px;
    line-height: 1.5;
}

/* ── Summary panel ── */
.summary-panel {
    background: #0A1628;
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 20px 22px;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: #475569;
}

.summary-grid { gap: 0; }

.sum-row {
    padding: 7px 0;
    border-bottom: 1px solid #1A2535;
    display: flex;
    flex-direction: column;
    gap: 1px;
}

.sum-val {
    display: block !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #CBD5E1 !important;
    text-transform: none;
    letter-spacing: 0;
}

.sum-model { color: #42A5F5 !important; }

/* ── Insights ── */
.insights-grid { margin-top: 18px; }

.insight-card {
    background: #0A1628;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 18px 20px;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: #475569;
}

.insight-card-green { border-color: #14532D; }

.insight-body {
    display: block !important;
    font-size: 0.84rem !important;
    font-weight: 400 !important;
    color: #CBD5E1 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    white-space: pre-line;
    margin-top: 10px;
    line-height: 2;
}

/* ── Model comparison ── */
.comparison-card {
    background: #0A1628;
    border: 1px solid #263548;
    border-radius: 10px;
    padding: 18px 20px;
    margin-top: 18px;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: #475569;
}

.comp-grid { margin-top: 14px; }

.comp-tile {
    background: #111C2D;
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    padding: 16px 18px;
    text-align: center;
}

.comp-tile-gold { border-color: #78350F; }

.comp-prob {
    display: block !important;
    font-size: 2.4rem !important;
    font-weight: 900 !important;
    color: #42A5F5 !important;
    line-height: 1 !important;
    margin: 8px 0 4px;
}

.comp-badge {
    display: inline-block !important;
    font-size: 0.78rem !important;
    color: #94A3B8 !important;
    text-transform: none;
    letter-spacing: 0;
}

/* ── History ── */
.history-card {
    background: #111C2D;
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 22px 26px;
    margin-top: 16px;
}

.history-card h3 {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #42A5F5;
    margin-bottom: 14px;
}

/* ── Footer ── */
.page-footer {
    background: #050C18;
    color: #1E3A5F;
    text-align: center;
    padding: 14px;
    font-size: 0.68rem;
    letter-spacing: 0.6px;
    border-top: 1px solid #111C2D;
}

/* ── Responsive ── */
@media (max-width: 860px) {
    .page-body { padding: 12px; }
    .kpi-strip { flex-wrap: wrap; }
    .action-bar { flex-direction: column; }
    .insights-grid { flex-direction: column; }
    .header-inner { flex-direction: column; }
}
"""

CSS_PATH = "./app_styles.css"
with open(CSS_PATH, "w", encoding="utf-8") as _f:
    _f.write(CSS)
print(f"[OK] CSS → {CSS_PATH}")

# ============================================================
# 9. RUN — RENDER-COMPATIBLE
# ============================================================

if __name__ == "__main__":
    import os
    # Render provides PORT env var; fallback to 10000 for local dev
    port = int(os.environ.get("PORT", 10000))
    # Render requires 0.0.0.0; localhost only for local dev
    host = os.environ.get("HOST", "0.0.0.0")

    print("\n" + "=" * 70)
    print("  AI Loan Default Prediction System  v3.0")
    print("  Machine Learning for Loan Default Prediction with Taipy")
    print("  Nigerian Naira (NGN) · All 8 Steps Implemented")
    print("=" * 70)
    print(f"  Models   : {MODEL_OPTIONS}")
    print(f"  Active   : {current_model_name}")
    print(f"  Features : {len(selected_features)}")
    print(f"  Host     : {host}")
    print(f"  Port     : {port}")
    print("  Press Ctrl+C to stop\n")

    gui = Gui(page=page, css_file=CSS_PATH)
    gui.run(
        title="AI Loan Default Prediction — Nigeria v3.0",
        host=host,
        port=port,
        use_reloader=False,
        debug=False,
        watermark=False,
    )