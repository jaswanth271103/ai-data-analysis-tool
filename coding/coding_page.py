import streamlit as st

def render_coding_page(df):
    st.header("🧪 Coding Page")
    st.write("Dataset Preview")
    st.dataframe(df.head())
