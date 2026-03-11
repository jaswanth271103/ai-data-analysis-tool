# ==========================================================
# app.py — ENTERPRISE AI + BI DASHBOARD (FINAL STABLE)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from coding.coding_page import render_coding_page

# ==========================================================
# PAGE STATE INIT
# ==========================================================
if "page" not in st.session_state:
    st.session_state.page = "main"

# ==========================================================
# APP CONFIG
# ==========================================================
st.set_page_config(layout="wide")
st.title("🧠 Enterprise AI Business Intelligence Platform")

# ==========================================================
# FILE LOADING
# ==========================================================
@st.cache_data(show_spinner=False)
def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file, engine="openpyxl")
    else:
        raise ValueError("Unsupported file format")

files = st.file_uploader(
    "Upload CSV / Excel files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

if not files:
    st.stop()

datasets = {f.name: load_file(f) for f in files}

# ==========================================================
# MAIN PAGE
# ==========================================================
if st.session_state.page == "main":

    # ---------------- DATASET SELECTION ----------------
    st.header("📎 Dataset Selection")

    base_name = st.selectbox("Select Dataset", list(datasets.keys()))
    raw_df = datasets[base_name].copy()
    df = raw_df.copy()

    # ✅ Persist dataset IMMEDIATELY (CRITICAL FIX)
    st.session_state.raw_df = raw_df
    st.session_state.df = df
    st.session_state.dataset_name = base_name

    if st.button("as coding"):
        st.session_state.page = "coding"
        st.rerun()

    # ---------------- RAW DATA ----------------
    st.header("0️⃣ Raw Data Preview")
    st.dataframe(raw_df.head(100), use_container_width=True)

    # ---------------- DATA CLEANING ----------------
    df = df.replace([np.inf, -np.inf], np.nan)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if num_cols:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    for col in cat_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

    df = df.drop_duplicates()

    # 🔁 Update cleaned df in session state
    st.session_state.df = df

    st.header("1️⃣ Cleaned Data Preview")
    st.dataframe(df.head(100), use_container_width=True)

    # ---------------- DATA QUALITY ----------------
    st.header("📋 Data Quality Report")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(raw_df))
    c2.metric("Columns", raw_df.shape[1])
    c3.metric("Missing Values", int(raw_df.isna().sum().sum()))
    c4.metric("Duplicates Removed", len(raw_df) - len(df))

    # ---------------- FILTERS ----------------
    st.sidebar.header("🎛️ Filters")
    filtered_df = df.copy()

    if cat_cols:
        fcol = st.sidebar.selectbox("Filter Column", cat_cols)
        fvals = st.sidebar.multiselect(
            "Select Values",
            filtered_df[fcol].unique(),
            default=filtered_df[fcol].unique()
        )
        filtered_df = filtered_df[filtered_df[fcol].isin(fvals)]

    # ---------------- STATISTICS ----------------
    st.header("📐 Statistical Summary")

    stats_cols = filtered_df.select_dtypes(include="number").columns.tolist()
    if not stats_cols:
        st.stop()

    stats_df = filtered_df[stats_cols].describe().T
    stats_df["variance"] = filtered_df[stats_cols].var()
    stats_df["skewness"] = filtered_df[stats_cols].skew()
    stats_df["kurtosis"] = filtered_df[stats_cols].kurtosis()

    st.dataframe(stats_df, use_container_width=True)

    # ---------------- KPI CARDS ----------------
    st.header("📊 KPI Cards")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Average", round(filtered_df[stats_cols].mean().mean(), 2))
    k2.metric("Maximum", round(filtered_df[stats_cols].max().max(), 2))
    k3.metric("Minimum", round(filtered_df[stats_cols].min().min(), 2))
    k4.metric("Std Dev", round(filtered_df[stats_cols].std().mean(), 2))

    # ---------------- MACHINE LEARNING ----------------
    st.header("🤖 Machine Learning")
    target = st.selectbox("Target Metric", stats_cols)

    @st.cache_data(show_spinner=False)
    def compute_ml_score(df, cols, target):
        X, y = df[cols], df[target]
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
        return cross_val_score(pipe, X, y, cv=5, scoring="r2").mean()

    st.metric(
        "Linear Regression CV R²",
        f"{compute_ml_score(filtered_df, stats_cols, target):.3f}"
    )

    # ---------------- PCA (SAFE) ----------------
    @st.cache_data(show_spinner=False)
    def compute_pca_safe(df, cols):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2))
        ])
        return pipe.fit_transform(df[cols])

    pca_vals = compute_pca_safe(filtered_df, stats_cols)
    pca_df = pd.DataFrame(pca_vals, columns=["PC1", "PC2"])

    # ---------------- DASHBOARDS (20+ GRAPHS) ----------------
    st.header("📈 Advanced Analytical Dashboards")

    figs = []

    # Distribution
    figs += [
        px.histogram(filtered_df, x=target),
        px.box(filtered_df, y=target),
        px.violin(filtered_df, y=target),
        px.ecdf(filtered_df, x=target),
        px.histogram(filtered_df, x=target, marginal="rug"),
        px.histogram(filtered_df, x=target, marginal="box"),
    ]

    # Trends
    figs += [
        px.line(filtered_df, y=target),
        px.area(filtered_df, y=target),
        px.scatter(filtered_df, y=target),
        px.scatter(filtered_df, x=filtered_df.index, y=target, trendline="ols"),
    ]

    # Relationships
    if len(stats_cols) >= 2:
        figs.append(px.scatter(filtered_df, x=stats_cols[0], y=stats_cols[1]))

    figs.append(px.scatter_matrix(
        filtered_df[stats_cols].sample(min(300, len(filtered_df)))
    ))

    # Multivariate
    figs.append(px.imshow(filtered_df[stats_cols].corr()))
    figs.append(px.scatter(pca_df, x="PC1", y="PC2"))

    # ---------------- RENDER ----------------
    cols = st.columns(2)
    for i, fig in enumerate(figs):
        fig.update_layout(height=300)
        cols[i % 2].plotly_chart(fig, use_container_width=True)

    # ---------------- EXPORT ----------------
    html = "".join(
        pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
        for fig in figs
    )

    st.download_button(
        "⬇️ Download Full BI Report (HTML)",
        f"<html><body>{html}</body></html>",
        "enterprise_bi_report.html",
        "text/html"
    )

    st.success("✅ ENTERPRISE BI SYSTEM — FULLY STABLE")

# ==========================================================
# CODING PAGE (SAFE)
# ==========================================================
if st.session_state.page == "coding":

    if "df" not in st.session_state:
        st.error("No dataset loaded. Please select a dataset first.")
        st.session_state.page = "main"
        st.stop()

    render_coding_page(st.session_state.df)
