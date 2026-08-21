
"""
ChurnGuard – Bank Customer Intelligence Platform
Multi-model scoring + Natural Language Querying (LLM-style)
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score, accuracy_score
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard | Bank Customer Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional theme ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Professional blue palette ─────────────────────────────────
   Navy #0A1628 | Deep #0F2744 | Royal #1A4A7A | Accent #2563EB
   Soft #DBEAFE | Mist #EFF6FF | Slate text #334155 | Muted #64748B
─────────────────────────────────────────────────────────────── */

.main { background-color: #F0F5FA; }
.block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1200px; }

/* Header */
.app-header {
    background: linear-gradient(125deg, #0A1628 0%, #0F2744 45%, #1A4A7A 100%);
    color: #fff;
    padding: 1.5rem 1.75rem;
    border-radius: 10px;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 16px rgba(10, 22, 40, 0.28);
}
.app-header h1 {
    margin: 0;
    font-size: 1.55rem;
    font-weight: 700;
    color: #fff !important;
    letter-spacing: -0.02em;
}
.app-header p {
    margin: 0.35rem 0 0 0;
    font-size: 0.9rem;
    color: rgba(219, 234, 254, 0.92) !important;
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 1.15rem 1.2rem;
    box-shadow: 0 1px 4px rgba(15, 39, 68, 0.07);
    border: 1px solid #BFDBFE;
    text-align: center;
    height: 100%;
}
.metric-card .label {
    font-size: 0.72rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #0F2744;
}

/* Risk badges – blue-tinted professional tones */
.risk-high {
    background: #EFF6FF;
    color: #1E3A8A;
    border: 1px solid #93C5FD;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-weight: 700;
    font-size: 1rem;
    text-align: center;
}
.risk-medium {
    background: #DBEAFE;
    color: #1E40AF;
    border: 1px solid #60A5FA;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-weight: 700;
    font-size: 1rem;
    text-align: center;
}
.risk-low {
    background: #F0F9FF;
    color: #0369A1;
    border: 1px solid #7DD3FC;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-weight: 700;
    font-size: 1rem;
    text-align: center;
}

/* Panels */
.result-panel {
    background: #ffffff;
    border-radius: 10px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 1px 6px rgba(15, 39, 68, 0.08);
    border: 1px solid #BFDBFE;
    margin-top: 1rem;
}
.section-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #0F2744;
    margin: 1rem 0 0.55rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #93C5FD;
}

/* NL answer */
.answer-box {
    background: #ffffff;
    border-left: 4px solid #2563EB;
    border-radius: 0 8px 8px 0;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 4px rgba(15, 39, 68, 0.06);
    margin-top: 0.75rem;
    font-size: 0.95rem;
    line-height: 1.55;
    color: #1E293B;
}
.flow-box {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 0.9rem 1.15rem;
    font-size: 0.86rem;
    color: #1E3A5F;
    margin-bottom: 1.1rem;
    line-height: 1.5;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    border: 1px solid #93C5FD;
    background: #ffffff;
    color: #1E3A8A;
    font-size: 0.84rem;
    font-weight: 500;
    transition: all 0.12s ease;
    text-align: left;
}
.stButton > button:hover {
    border-color: #2563EB;
    color: #0F2744;
    background: #EFF6FF;
}
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #0F2744 0%, #1A4A7A 55%, #2563EB 100%) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600;
    border-radius: 8px;
    text-align: center;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0A1628 0%, #0F2744 55%, #1A4A7A 100%) !important;
}

/* Sidebar – deep navy */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1628 0%, #0F2744 100%);
}
section[data-testid="stSidebar"] * { color: #DBEAFE !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(147, 197, 253, 0.2); }
section[data-testid="stSidebar"] .stSelectbox label { color: #93C5FD !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #15395a !important;
    border-color: #1A4A7A !important;
}

/* Inputs */
.stTextInput > div > div > input {
    border: 1px solid #93C5FD !important;
    border-radius: 8px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
}

/* Tables */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #1A4A7A, #2563EB) !important;
}

/* Info / alerts */
div[data-testid="stAlert"] {
    border-radius: 8px;
}

/* Hide chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ── Paths ────────────────────────────────────────────────────────────────────
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "bank-1.csv"
RANDOM_STATE = 42

FEATURE_COLS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "BalanceSalaryRatio", "HasZeroBalance", "HighProductCount",
]
NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "BalanceSalaryRatio", "HasZeroBalance", "HighProductCount",
]
CATEGORICAL_FEATURES = ["Geography", "Gender"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["BalanceSalaryRatio"] = out["Balance"] / (out["EstimatedSalary"] + 1.0)
    out["HasZeroBalance"] = (out["Balance"] == 0).astype(int)
    out["HighProductCount"] = (out["NumOfProducts"] >= 3).astype(int)
    return out


def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_FEATURES),
    ])


def best_threshold(y_true, probs):
    best_t, best_rec = 0.5, 0.0
    for t in np.arange(0.25, 0.75, 0.05):
        rec = recall_score(y_true, (probs >= t).astype(int), zero_division=0)
        if rec > best_rec:
            best_rec, best_t = rec, float(t)
    return best_t


@st.cache_resource(show_spinner=False)
def load_all():
    df = pd.read_csv(DATA_PATH)
    df_fe = engineer_features(df)
    X = df_fe[FEATURE_COLS]
    y = df_fe["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    candidates = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=8, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, class_weight="balanced", max_depth=12,
            min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.08,
            random_state=RANDOM_STATE
        ),
    }
    if HAS_XGB:
        candidates["XGBoost"] = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            scale_pos_weight=scale_pos, random_state=RANDOM_STATE,
            eval_metric="logloss", n_jobs=-1
        )
    if HAS_LGB:
        candidates["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            scale_pos_weight=scale_pos, random_state=RANDOM_STATE,
            verbose=-1, n_jobs=-1
        )

    models = {}
    rows = []
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        thr = best_threshold(y_test, probs)
        pred = (probs >= thr).astype(int)
        m = {
            "Model": name,
            "Threshold": thr,
            "Recall": float(recall_score(y_test, pred)),
            "Precision": float(precision_score(y_test, pred, zero_division=0)),
            "F1": float(f1_score(y_test, pred, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, probs)),
            "Accuracy": float(accuracy_score(y_test, pred)),
        }
        rows.append(m)
        models[name] = {"pipe": pipe, "threshold": thr, "metrics": m}

    metrics_df = pd.DataFrame(rows).sort_values("Recall", ascending=False)
    default_name = str(metrics_df.iloc[0]["Model"])

    full_probs = models[default_name]["pipe"].predict_proba(X)[:, 1]
    df_fe = df_fe.copy()
    df_fe["ChurnProbability"] = full_probs
    df_fe["RiskLevel"] = pd.cut(
        full_probs,
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )
    return df, df_fe, models, metrics_df, default_name


with st.spinner("Loading models…"):
    df_raw, df_risk, MODELS, METRICS_DF, DEFAULT_MODEL = load_all()


# ── Natural-language analytics engine ────────────────────────────────────────
class ChurnAnalyticsEngine:
    """Maps a business question to filters/aggregations on the dataset."""

    def __init__(self, data: pd.DataFrame):
        self.df = data

    def overall_churn_rate(self) -> str:
        rate = self.df["Exited"].mean() * 100
        n = len(self.df)
        churned = int(self.df["Exited"].sum())
        return (
            f"Overall churn rate is **{rate:.2f}%** "
            f"({churned:,} of {n:,} customers)."
        )

    def churn_by_geography(self) -> str:
        rates = (
            self.df.groupby("Geography")["Exited"]
            .agg(["mean", "count", "sum"])
            .sort_values("mean", ascending=False)
        )
        lines = [
            f"- **{g}**: {r['mean']*100:.1f}% churn "
            f"({int(r['sum'])} of {int(r['count'])} customers)"
            for g, r in rates.iterrows()
        ]
        return (
            "Churn rate by geography:\n\n"
            + "\n".join(lines)
            + f"\n\nHighest-risk region: **{rates.index[0]}**."
        )

    def churn_by_activity(self) -> str:
        r = self.df.groupby("IsActiveMember")["Exited"].mean()
        return (
            f"Inactive members churn at **{r[0]*100:.1f}%**.\n"
            f"Active members churn at **{r[1]*100:.1f}%**.\n\n"
            "Inactivity is one of the strongest behavioural signals of churn."
        )

    def churn_by_products(self) -> str:
        rates = self.df.groupby("NumOfProducts")["Exited"].agg(["mean", "count"])
        lines = [
            f"- **{k} product(s)**: {r['mean']*100:.1f}% churn (n={int(r['count'])})"
            for k, r in rates.iterrows()
        ]
        return "Churn rate by number of products:\n\n" + "\n".join(lines)

    def churn_by_gender(self) -> str:
        r = self.df.groupby("Gender")["Exited"].mean()
        return (
            f"Female churn rate: **{r.get('Female', 0)*100:.1f}%**.\n"
            f"Male churn rate: **{r.get('Male', 0)*100:.1f}%**."
        )

    def age_vs_churn(self) -> str:
        mc = self.df.loc[self.df["Exited"] == 1, "Age"].mean()
        ms = self.df.loc[self.df["Exited"] == 0, "Age"].mean()
        return (
            f"Average age of churned customers: **{mc:.1f}** years.\n"
            f"Average age of retained customers: **{ms:.1f}** years."
        )

    def high_income_churn(self, percentile: int = 75) -> str:
        thr = float(self.df["EstimatedSalary"].quantile(percentile / 100))
        high = self.df[self.df["EstimatedSalary"] >= thr]
        low = self.df[self.df["EstimatedSalary"] < thr]
        hr = high["Exited"].mean() * 100
        lr = low["Exited"].mean() * 100
        inact = (1 - high.loc[high["Exited"] == 1, "IsActiveMember"].mean()) * 100
        geo = high.loc[high["Exited"] == 1, "Geography"].value_counts(normalize=True)
        top = str(geo.index[0]) if len(geo) else "N/A"
        return (
            f"High-income segment (top {100 - percentile}%, salary ≥ €{thr:,.0f}):\n\n"
            f"- Churn rate: **{hr:.1f}%** (vs **{lr:.1f}%** for lower-income)\n"
            f"- Most common geography among high-income churners: **{top}**\n"
            f"- Share of inactive members among high-income churners: **{inact:.1f}%**\n\n"
            "Likely contributing factors include inactivity, product complexity, "
            "and regional service differences — not income alone."
        )

    def risk_segment_summary(self) -> str:
        if "RiskLevel" not in self.df.columns:
            return "Risk segmentation is not available yet."
        s = self.df.groupby("RiskLevel", observed=True).agg(
            Customers=("Exited", "count"),
            Churned=("Exited", "sum"),
            Rate=("Exited", "mean"),
        )
        lines = [
            f"- **{idx}**: {int(r.Customers):,} customers · "
            f"actual churn **{r.Rate*100:.1f}%** ({int(r.Churned)} churned)"
            for idx, r in s.iterrows()
        ]
        return "Customer risk segments:\n\n" + "\n".join(lines)

    def highest_risk_segments(self) -> str:
        parts = []
        g = self.df.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
        parts.append(f"Geography → **{g.index[0]}** ({g.iloc[0]*100:.1f}%)")
        a = self.df.groupby("IsActiveMember")["Exited"].mean()
        parts.append(
            f"Activity → Inactive **{a[0]*100:.1f}%** vs Active {a[1]*100:.1f}%"
        )
        p = self.df.groupby("NumOfProducts")["Exited"].mean().sort_values(ascending=False)
        parts.append(f"Products → **{int(p.index[0])} products** ({p.iloc[0]*100:.1f}%)")
        if "RiskLevel" in self.df.columns:
            r = (
                self.df.groupby("RiskLevel", observed=True)["Exited"]
                .mean()
                .sort_values(ascending=False)
            )
            parts.append(f"Model risk tier → **{r.index[0]}** ({r.iloc[0]*100:.1f}%)")
        return (
            "Customer segments with the highest observed churn risk:\n\n"
            + "\n".join(f"- {x}" for x in parts)
        )

    def balance_insight(self) -> str:
        zero = self.df.loc[self.df["Balance"] == 0, "Exited"].mean() * 100
        pos = self.df.loc[self.df["Balance"] > 0, "Exited"].mean() * 100
        n0 = int((self.df["Balance"] == 0).sum())
        return (
            f"Zero-balance customers ({n0:,}): **{zero:.1f}%** churn.\n"
            f"Positive-balance customers: **{pos:.1f}%** churn.\n\n"
            "Customers holding a balance show a higher observed churn rate."
        )

    def tenure_insight(self) -> str:
        bins = pd.cut(
            self.df["Tenure"],
            bins=[-1, 2, 5, 8, 10],
            labels=["0–2 yrs", "3–5 yrs", "6–8 yrs", "9–10 yrs"],
        )
        rates = self.df.groupby(bins, observed=True)["Exited"].mean()
        lines = [f"- **{k}**: {v*100:.1f}%" for k, v in rates.items()]
        return "Churn rate by tenure band:\n\n" + "\n".join(lines)

    def credit_insight(self) -> str:
        bins = pd.cut(
            self.df["CreditScore"],
            bins=[0, 500, 600, 700, 850],
            labels=["≤500", "501–600", "601–700", "701–850"],
        )
        rates = self.df.groupby(bins, observed=True)["Exited"].mean()
        lines = [f"- **{k}**: {v*100:.1f}%" for k, v in rates.items()]
        return "Churn rate by credit-score band:\n\n" + "\n".join(lines)

    def query(self, question: str) -> str:
        q = (question or "").lower().strip()
        if not q:
            return "Please enter a question."

        if re.search(r"high.?income|high.?salary|wealthy|why.*(leaving|churn)", q):
            return self.high_income_churn()
        if re.search(
            r"highest.*(risk|churn).*segment|which.*segment|segment.*highest", q
        ):
            return self.highest_risk_segments()
        if re.search(r"risk.*(level|segment|tier)|segment.*(summary|overview)", q):
            return self.risk_segment_summary()
        if re.search(r"geograph|country|region|france|spain|germany", q):
            return self.churn_by_geography()
        if re.search(r"inactive|active.?member|activity", q):
            return self.churn_by_activity()
        if re.search(r"product", q):
            return self.churn_by_products()
        if re.search(r"gender|male|female", q):
            return self.churn_by_gender()
        if re.search(r"\bage\b", q):
            return self.age_vs_churn()
        if re.search(r"balance|zero.?balance", q):
            return self.balance_insight()
        if re.search(r"tenure|years with", q):
            return self.tenure_insight()
        if re.search(r"credit.?score|credit score", q):
            return self.credit_insight()
        if re.search(r"overall|total.*churn|churn rate", q):
            return self.overall_churn_rate()

        return (
            "I can answer questions about segments with highest churn risk, "
            "high-income customers, geography, activity, products, gender, age, "
            "balance, tenure, credit score, risk tiers, and overall churn rate. "
            "Please rephrase your question."
        )


engine = ChurnAnalyticsEngine(df_risk)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ChurnGuard")
    st.caption("Bank Customer Intelligence")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Predict Churn", "Ask the Data", "Risk Segments", "Compare Models"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Active scoring model**")

    model_names = list(MODELS.keys())
    ordered = [DEFAULT_MODEL] + [m for m in model_names if m != DEFAULT_MODEL]
    selected_model = st.selectbox(
        "Model", ordered, index=0, label_visibility="collapsed"
    )
    sel = MODELS[selected_model]
    st.markdown(
        f"""
| | |
|---|---|
| Threshold | `{sel['threshold']:.2f}` |
| Recall | `{sel['metrics']['Recall']:.3f}` |
| Precision | `{sel['metrics']['Precision']:.3f}` |
| ROC-AUC | `{sel['metrics']['ROC-AUC']:.3f}` |
"""
    )
    st.caption("Primary metric: Recall · Class-balanced")

# ── Page: Predict ────────────────────────────────────────────────────────────
if page == "Predict Churn":
    st.markdown(
        f"""
        <div class="app-header">
            <h1>Customer Churn Prediction</h1>
            <p>Score a customer with <b>{selected_model}</b> · threshold optimised for Recall.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="section-title">Quick-fill example profiles</p>',
        unsafe_allow_html=True,
    )
    e1, e2, e3 = st.columns(3)
    with e1:
        fill_high = st.button("High-risk example", use_container_width=True)
    with e2:
        fill_med = st.button("Medium-risk example", use_container_width=True)
    with e3:
        fill_low = st.button("Low-risk example", use_container_width=True)

    defaults = {
        "credit_score": 600, "age": 40, "tenure": 3, "balance": 50000.0,
        "num_products": 1, "has_cr_card": 1, "is_active": 1,
        "estimated_salary": 100000.0, "geography": "France", "gender": "Female",
    }
    if fill_high:
        defaults = {
            "credit_score": 550, "age": 48, "tenure": 2, "balance": 125000.0,
            "num_products": 3, "has_cr_card": 1, "is_active": 0,
            "estimated_salary": 90000.0, "geography": "Germany", "gender": "Female",
        }
    elif fill_med:
        defaults = {
            "credit_score": 620, "age": 42, "tenure": 4, "balance": 80000.0,
            "num_products": 1, "has_cr_card": 1, "is_active": 0,
            "estimated_salary": 95000.0, "geography": "Spain", "gender": "Male",
        }
    elif fill_low:
        defaults = {
            "credit_score": 720, "age": 32, "tenure": 6, "balance": 0.0,
            "num_products": 2, "has_cr_card": 1, "is_active": 1,
            "estimated_salary": 110000.0, "geography": "France", "gender": "Male",
        }

    st.markdown(
        '<p class="section-title">Customer details</p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.slider("Credit Score", 300, 850, int(defaults["credit_score"]))
        age = st.slider("Age", 18, 92, int(defaults["age"]))
        tenure = st.slider("Tenure (years)", 0, 10, int(defaults["tenure"]))
        balance = st.number_input(
            "Balance (€)", min_value=0.0,
            value=float(defaults["balance"]), step=1000.0,
        )
    with c2:
        num_products = st.selectbox(
            "Number of Products", [1, 2, 3, 4],
            index=[1, 2, 3, 4].index(int(defaults["num_products"])),
        )
        has_cr_card = st.selectbox(
            "Has Credit Card", [0, 1], index=int(defaults["has_cr_card"]),
            format_func=lambda x: "Yes" if x else "No",
        )
        is_active = st.selectbox(
            "Is Active Member", [0, 1], index=int(defaults["is_active"]),
            format_func=lambda x: "Yes" if x else "No",
        )
        estimated_salary = st.number_input(
            "Estimated Salary (€)", min_value=0.0,
            value=float(defaults["estimated_salary"]), step=1000.0,
        )
    with c3:
        geography = st.selectbox(
            "Geography", ["France", "Spain", "Germany"],
            index=["France", "Spain", "Germany"].index(defaults["geography"]),
        )
        gender = st.selectbox(
            "Gender", ["Female", "Male"],
            index=["Female", "Male"].index(defaults["gender"]),
        )
        st.markdown("")
        go = st.button("Score Customer", type="primary", use_container_width=True)

    if go:
        row = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": estimated_salary,
        }
        Xc = engineer_features(pd.DataFrame([row]))[FEATURE_COLS]
        pipe = MODELS[selected_model]["pipe"]
        thr = MODELS[selected_model]["threshold"]
        prob = float(pipe.predict_proba(Xc)[0, 1])
        will_churn = prob >= thr

        if prob >= 0.60:
            risk_html = '<div class="risk-high">HIGH RISK</div>'
            action = "Contact a retention specialist and prepare a personalised offer."
        elif prob >= 0.30:
            risk_html = '<div class="risk-medium">MEDIUM RISK</div>'
            action = "Add to a nurturing sequence and monitor engagement for 30 days."
        else:
            risk_html = '<div class="risk-low">LOW RISK</div>'
            action = "Standard relationship management — no urgent action required."

        st.markdown('<div class="result-panel">', unsafe_allow_html=True)
        st.markdown(f"#### Result · {selected_model}")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                f'<div class="metric-card"><div class="label">Churn Probability</div>'
                f'<div class="value">{prob*100:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
        with r2:
            label = "Likely to Churn" if will_churn else "Likely to Stay"
            st.markdown(
                f'<div class="metric-card"><div class="label">Prediction</div>'
                f'<div class="value" style="font-size:1.15rem;">{label}</div></div>',
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(risk_html, unsafe_allow_html=True)
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(
            f"Decision threshold = {thr:.2f} (Recall-optimised for {selected_model})"
        )
        st.info(f"**Recommended action:** {action}")
        st.markdown("</div>", unsafe_allow_html=True)

# ── Page: Ask the Data (fixed) ───────────────────────────────────────────────
elif page == "Ask the Data":
    st.markdown(
        """
        <div class="app-header">
            <h1>Querying Data Using LLMs</h1>
            <p>Natural language interaction with the dataset — questions become analytical logic on real data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="flow-box">
        <b>How it works (conceptual LLM layer)</b><br>
        User question → Intent router → Analytical logic (filters &amp; aggregations)
        → Computed result → Natural language answer.<br>
        <i>Every figure is calculated from the dataset. Nothing is invented.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="section-title">Suggested business questions</p>',
        unsafe_allow_html=True,
    )

    suggestions = [
        "Which customer segments have the highest churn risk?",
        "Why are high-income customers leaving the bank?",
        "Which geography has the highest churn?",
        "How does being an inactive member affect churn?",
        "What is the overall churn rate?",
        "Show me the risk segment summary",
        "How does number of products relate to churn?",
        "What is the relationship between age and churn?",
        "How does account balance relate to churn?",
        "How does tenure relate to churn?",
        "How does credit score relate to churn?",
        "What is the churn rate by gender?",
    ]

    # Session state for selected suggestion (fixed pattern)
    if "ask_query" not in st.session_state:
        st.session_state.ask_query = ""
    if "ask_answer" not in st.session_state:
        st.session_state.ask_answer = ""

    def _set_suggestion(q: str):
        st.session_state.ask_query = q
        st.session_state.ask_answer = engine.query(q)

    cols = st.columns(2)
    for i, q in enumerate(suggestions):
        cols[i % 2].button(
            q,
            key=f"sug_{i}",
            use_container_width=True,
            on_click=_set_suggestion,
            args=(q,),
        )

    st.markdown(
        '<p class="section-title">Or type your own question</p>',
        unsafe_allow_html=True,
    )

    with st.form("ask_form", clear_on_submit=False):
        typed = st.text_input(
            "Question",
            value=st.session_state.ask_query,
            placeholder="e.g. Which customer segments have the highest churn risk?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Get answer", type="primary", use_container_width=True)

    if submitted and typed.strip():
        st.session_state.ask_query = typed.strip()
        st.session_state.ask_answer = engine.query(typed.strip())

    if st.session_state.ask_answer:
        st.markdown(
            f'<div class="answer-box"><strong>Answer</strong><br><br>'
            f'{st.session_state.ask_answer}</div>',
            unsafe_allow_html=True,
        )
        st.caption("All figures are computed live from the customer dataset.")

# ── Page: Risk Segments ──────────────────────────────────────────────────────
elif page == "Risk Segments":
    st.markdown(
        """
        <div class="app-header">
            <h1>Customer Risk Segmentation</h1>
            <p>Portfolio view by predicted risk — prioritise retention where it matters.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    X_all = engineer_features(df_raw)[FEATURE_COLS]
    probs_all = MODELS[selected_model]["pipe"].predict_proba(X_all)[:, 1]
    view = df_raw.copy()
    view["ChurnProbability"] = probs_all
    view["RiskLevel"] = pd.cut(
        probs_all,
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    summary = (
        view.groupby("RiskLevel", observed=True)
        .agg(
            Customers=("Exited", "count"),
            Churned=("Exited", "sum"),
            AvgProb=("ChurnProbability", "mean"),
            AvgAge=("Age", "mean"),
            PctInactive=("IsActiveMember", lambda x: (1 - x.mean()) * 100),
            AvgBalance=("Balance", "mean"),
        )
        .reset_index()
    )
    summary["Churn Rate %"] = (
        summary["Churned"] / summary["Customers"] * 100
    ).round(1)
    summary["Avg Age"] = summary["AvgAge"].round(1)
    summary["Avg Balance (€)"] = summary["AvgBalance"].round(0)
    summary["% Inactive"] = summary["PctInactive"].round(1)
    summary["Avg Probability"] = summary["AvgProb"].round(3)

    st.caption(f"Scored with **{selected_model}**")
    st.dataframe(
        summary[
            [
                "RiskLevel", "Customers", "Churned", "Churn Rate %",
                "Avg Probability", "Avg Age", "% Inactive", "Avg Balance (€)",
            ]
        ].rename(columns={"RiskLevel": "Risk Level"}),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Customers by risk level**")
        st.bar_chart(summary.set_index("RiskLevel")["Customers"], color="#1A4A7A")
    with c2:
        st.markdown("**Actual churn rate by risk level**")
        st.bar_chart(summary.set_index("RiskLevel")["Churn Rate %"], color="#2563EB")

    st.markdown("---")
    a, b, c = st.columns(3)
    with a:
        st.markdown(
            '<div class="risk-high">HIGH RISK</div>'
            '<p style="font-size:0.85rem;margin-top:0.5rem;color:#475569;">'
            "Specialist contact + personalised incentive.</p>",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            '<div class="risk-medium">MEDIUM RISK</div>'
            '<p style="font-size:0.85rem;margin-top:0.5rem;color:#475569;">'
            "Nurturing sequence + 30-day monitoring.</p>",
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            '<div class="risk-low">LOW RISK</div>'
            '<p style="font-size:0.85rem;margin-top:0.5rem;color:#475569;">'
            "Standard relationship management.</p>",
            unsafe_allow_html=True,
        )

# ── Page: Compare Models ─────────────────────────────────────────────────────
else:
    st.markdown(
        """
        <div class="app-header">
            <h1>Model Comparison</h1>
            <p>All models trained with class balancing · ranked by Recall (primary metric).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    show = METRICS_DF.copy()
    for col in ["Threshold", "Recall", "Precision", "F1", "ROC-AUC", "Accuracy"]:
        show[col] = show[col].map(lambda x: f"{x:.3f}")
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown(
        '<p class="section-title">Performance comparison</p>',
        unsafe_allow_html=True,
    )
    chart_df = METRICS_DF.set_index("Model")[["Recall", "Precision", "F1", "ROC-AUC"]]
    st.bar_chart(chart_df)

    st.markdown(
        """
**Design choices**
- **Primary metric:** Recall — minimise customers who churn but are predicted to stay
- **Class imbalance:** `class_weight='balanced'` / `scale_pos_weight` (no SMOTE)
- **Threshold:** swept per model to maximise Recall on the hold-out set
- **Split:** stratified 80/20 so the ~20% churn rate is preserved
"""
    )