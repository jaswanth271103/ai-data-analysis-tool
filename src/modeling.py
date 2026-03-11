import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def run_model(df):
    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns")
        return

    target = st.selectbox("Select target column", numeric_cols)
    features = [c for c in numeric_cols if c != target]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    st.write("📌 Model Used: Linear Regression")
    st.write(f"📊 R² Score: **{score:.2f}**")
