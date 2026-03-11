import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def eda_report(df):
    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) == 0:
        st.warning("No numeric columns for EDA")
        return

    col = st.selectbox("Select column for analysis", numeric_cols)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(df[col], kde=True, ax=ax[0])
    ax[0].set_title("Distribution")

    sns.boxplot(x=df[col], ax=ax[1])
    ax[1].set_title("Outliers")

    st.pyplot(fig)
