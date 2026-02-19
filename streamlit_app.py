import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Optional deps
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
except Exception:  # pragma: no cover
    KMeans = None

try:
    from prophet import Prophet  # type: ignore
except Exception:  # pragma: no cover
    Prophet = None


st.set_page_config(
    page_title="Smart Retail Analytics",
    page_icon="🛍️",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Data loading


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    """Load the Online Retail II dataset from /data."""
    DATA_FILENAME = Path(__file__).parent / "data" / "online_retail_II.csv"
    df = pd.read_csv(DATA_FILENAME)

    # Normalize column names that sometimes vary by source
    rename_map = {}
    if "CustomerID" in df.columns and "Customer ID" not in df.columns:
        rename_map["CustomerID"] = "Customer ID"
    if "InvoiceNo" in df.columns and "Invoice" not in df.columns:
        rename_map["InvoiceNo"] = "Invoice"
    if rename_map:
        df = df.rename(columns=rename_map)

    return df


@st.cache_data
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning and feature creation used across the notebook."""
    out = df.copy()

    # Date parsing
    if "InvoiceDate" in out.columns:
        out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="coerce")

    # TotalPrice
    if "TotalPrice" not in out.columns and {"Quantity", "Price"}.issubset(out.columns):
        out["TotalPrice"] = out["Quantity"] * out["Price"]

    # Basic cleaning from notebook
    if {"Quantity", "Price"}.issubset(out.columns):
        out = out[(out["Quantity"] > 0) & (out["Price"] > 0)].copy()

    if "Description" in out.columns:
        out = out.dropna(subset=["Description"])

    return out


@st.cache_data
def build_monthly_sales(df_clean: pd.DataFrame) -> pd.Series:
    s = df_clean.set_index("InvoiceDate").resample("M")["TotalPrice"].sum().sort_index()
    return s


@st.cache_data
def build_rfm(df_clean: pd.DataFrame) -> pd.DataFrame:
    reference_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        df_clean.dropna(subset=["Customer ID"])
        .groupby("Customer ID")
        .agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
    )
    return rfm


def metric_card(label: str, value: str, delta: str | None = None, delta_color: str = "normal") -> None:
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


# -----------------------------------------------------------------------------
# Sidebar controls

st.sidebar.header("Controls")

raw = load_raw_data()
df = preprocess(raw)

if df.empty:
    st.error("Dataset is empty after preprocessing. Check your CSV file.")
    st.stop()

min_date = df["InvoiceDate"].min()
max_date = df["InvoiceDate"].max()

date_range = st.sidebar.date_input(
    "Invoice date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)

countries = sorted([c for c in df["Country"].dropna().unique()])
default_countries = [c for c in ["United Kingdom", "Germany", "France"] if c in countries]
selected_countries = st.sidebar.multiselect(
    "Countries",
    options=countries,
    default=default_countries if default_countries else (countries[:3] if len(countries) >= 3 else countries),
)

top_n = st.sidebar.slider("Top N products/countries", min_value=5, max_value=30, value=10, step=1)

churn_threshold = st.sidebar.slider("Churn threshold in days (RFM)", min_value=30, max_value=180, value=90, step=5)

k_clusters = st.sidebar.slider("RFM clusters (KMeans)", min_value=2, max_value=8, value=4, step=1)

forecast_months = st.sidebar.slider("Forecast horizon (months)", min_value=3, max_value=24, value=6, step=1)

# Apply filters
start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
df_f = df[(df["InvoiceDate"] >= start_d) & (df["InvoiceDate"] < end_d)].copy()
if selected_countries:
    df_f = df_f[df_f["Country"].isin(selected_countries)].copy()

# -----------------------------------------------------------------------------
# Page content

st.title("🛍️ Smart Retail Analytics Dashboard")
st.caption("EDA, RFM segmentation, churn flagging, and sales forecasting for Online Retail II.")

# KPI row
kpi_cols = st.columns(5)
with kpi_cols[0]:
    metric_card("Rows", f"{len(df_f):,}")
with kpi_cols[1]:
    metric_card("Orders (unique invoices)", f"{df_f['Invoice'].nunique():,}")
with kpi_cols[2]:
    metric_card("Customers", f"{df_f['Customer ID'].nunique():,}")
with kpi_cols[3]:
    metric_card("Revenue", f"{df_f['TotalPrice'].sum():,.0f}")
with kpi_cols[4]:
    avg_aov = df_f.groupby("Invoice")["TotalPrice"].sum().mean() if df_f["Invoice"].nunique() else 0
    metric_card("Avg order value", f"{avg_aov:,.0f}")

st.divider()

tab_overview, tab_sales, tab_products, tab_rfm, tab_forecast = st.tabs(["Overview", "Sales", "Products & Countries", "RFM & Churn", "Forecast"])

with tab_overview:
    st.subheader("Dataset preview")
    st.dataframe(df_f.head(50), use_container_width=True)

    st.subheader("Missingness")
    miss = df_f.isna().mean().sort_values(ascending=False).to_frame("missing_rate")
    st.dataframe(miss.head(20), use_container_width=True)

with tab_sales:
    st.subheader("Monthly revenue")
    monthly = build_monthly_sales(df_f)
    st.line_chart(monthly, height=320)

    st.subheader("Revenue distribution (order level)")
    order_rev = df_f.groupby("Invoice", as_index=False)["TotalPrice"].sum()
    cap = order_rev["TotalPrice"].quantile(0.99)
    st.bar_chart(order_rev["TotalPrice"].clip(upper=cap), height=260)

with tab_products:
    left, right = st.columns([1, 1])

    with left:
        st.subheader(f"Top {top_n} products by quantity")
        prod = df_f.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(top_n).reset_index()
        st.dataframe(prod, use_container_width=True, height=380)

    with right:
        st.subheader(f"Top {top_n} countries by revenue")
        country = df_f.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(top_n).reset_index()
        st.dataframe(country, use_container_width=True, height=380)

with tab_rfm:
    st.subheader("RFM features")
    rfm = build_rfm(df_f)

    if KMeans is None:
        st.warning("scikit-learn is not available. Install scikit-learn to enable clustering and churn modeling.")
        st.dataframe(rfm.head(50), use_container_width=True)
    else:
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

        km = KMeans(n_clusters=k_clusters, random_state=42, n_init="auto")
        rfm = rfm.copy()
        rfm["Cluster"] = km.fit_predict(rfm_scaled)

        rfm["Churn"] = (rfm["Recency"] > churn_threshold).astype(int)

        a, b, c = st.columns(3)
        with a:
            metric_card("Customers", f"{len(rfm):,}")
        with b:
            metric_card("Churn rate", f"{rfm['Churn'].mean():.1%}")
        with c:
            metric_card("Clusters", f"{k_clusters}")

        st.caption("Churn here is a simple heuristic: Recency greater than threshold days.")
        st.dataframe(rfm.head(50), use_container_width=True)

        st.subheader("Churn model (Random Forest)")
        X = rfm[["Recency", "Frequency", "Monetary", "Cluster"]]
        y = rfm["Churn"]

        if y.nunique() < 2:
            st.info("Churn label has only one class under current filters. Try adjusting date range or threshold.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).T, use_container_width=True)

            cm = confusion_matrix(y_test, y_pred)
            st.write("Confusion matrix (rows=true, cols=pred):")
            st.dataframe(pd.DataFrame(cm, index=["0", "1"], columns=["0", "1"]), use_container_width=False)

            importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.subheader("Feature importance")
            st.bar_chart(importances, height=260)

with tab_forecast:
    st.subheader("Sales forecasting (Prophet)")
    monthly_all = build_monthly_sales(df_f)

    if Prophet is None:
        st.warning("Prophet is not available. Install prophet to enable forecasting.")
        st.line_chart(monthly_all, height=320)
    else:
        prophet_df = monthly_all.reset_index()
        prophet_df.columns = ["ds", "y"]

        m = Prophet()
        m.fit(prophet_df)

        future = m.make_future_dataframe(periods=forecast_months, freq="M")
        fc = m.predict(future)

        view = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(24 + forecast_months).copy()
        view = view.set_index("ds")

        st.line_chart(view[["yhat", "yhat_lower", "yhat_upper"]], height=320)
        st.caption("yhat is the forecast, and the bands are Prophet uncertainty intervals.")
