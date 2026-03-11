def generate_insights(df):
    insights = []

    rows, cols = df.shape
    insights.append(f"Dataset contains {rows} rows and {cols} columns.")

    missing = df.isnull().sum().sum()
    insights.append("No missing values after cleaning." if missing == 0 else "Missing values still exist.")

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        insights.append(f"Average of {col} is {df[col].mean():.2f}")

    return insights
