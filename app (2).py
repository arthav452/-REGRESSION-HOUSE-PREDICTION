import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="House Price Regression", layout="wide")

FEATURES = [
    "Square_Footage", "Year_Built", "Num_Bedrooms",
    "Num_Bathrooms", "Lot_Size", "Garage_Size", "Neighborhood_Quality"
]
TARGET = "House_Price"

MODEL_DEFS = {
    "Linear Regression": lambda: LinearRegression(),
    "Decision Tree Regressor": lambda: DecisionTreeRegressor(
        max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42
    ),
    "Random Forest Regressor": lambda: RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, random_state=42
    ),
    "Support Vector Machine Regressor": lambda: SVR(
        kernel="rbf", C=100, gamma="scale", epsilon=0.1
    ),
    "K-Nearest Neighbor Regressor": lambda: KNeighborsRegressor(
        n_neighbors=5, weights="uniform", metric="minkowski", p=2
    ),
    "Gradient Boosting Regressor": lambda: GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    ),
}


@st.cache_data
def load_data(file):
    return pd.read_csv(file)


@st.cache_resource
def train_models(df, test_size, random_state):
    x = df[FEATURES]
    y = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    results = {}
    for name, build in MODEL_DEFS.items():
        model = build()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        results[name] = {
            "model": model,
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        }
    return results, (x_train, x_test, y_train, y_test)


def main():
    st.title("🏠 House Price Regression Explorer")
    st.caption(
        "Upload your housing dataset to explore the data, train six regression "
        "models, compare their performance, and predict prices for new houses."
    )

    with st.sidebar:
        st.header("1. Data")
        uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
        st.header("2. Train/Test Split")
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random state", value=42, step=1)

    if uploaded_file is None:
        st.info(
            "👈 Upload a CSV with columns: "
            + ", ".join(FEATURES + [TARGET])
            + " to get started."
        )
        return

    df = load_data(uploaded_file)

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        st.error(f"Dataset is missing required columns: {missing_cols}")
        return

    tab_overview, tab_eda, tab_models, tab_predict = st.tabs(
        ["📋 Overview", "📊 EDA", "🤖 Model Comparison", "🔮 Predict"]
    )

    # ---------------- Overview ----------------
    with tab_overview:
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing values", int(df.isnull().sum().sum()))

        st.subheader("Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Summary statistics")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("Duplicate rows")
        st.write(f"{int(df.duplicated().sum())} duplicate rows found.")

    # ---------------- EDA ----------------
    with tab_eda:
        st.subheader("Outlier detection (IQR method)")
        outlier_rows = []
        for col in FEATURES:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            outlier_rows.append({
                "Feature": col,
                "Outliers": len(outliers),
                "Percentage": f"{(len(outliers) / len(df)) * 100:.2f}%",
            })
        st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)

        st.subheader("Distributions")
        selected_col = st.selectbox("Select feature for boxplot", FEATURES)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(x=df[selected_col], ax=ax)
        st.pyplot(fig)

        st.subheader("Square Footage vs House Price")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x="Square_Footage", y=TARGET, data=df, ax=ax2)
        st.pyplot(fig2)

        st.subheader("Year Built vs House Price (regression line)")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.regplot(x="Year_Built", y=TARGET, data=df, ax=ax3)
        st.pyplot(fig3)

        st.subheader("Correlation heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)

    # ---------------- Models ----------------
    with tab_models:
        with st.spinner("Training models..."):
            results, splits = train_models(df, test_size, random_state)

        st.session_state["results"] = results

        comp_df = pd.DataFrame({
            "Algorithm": list(results.keys()),
            "R2 Score": [r["r2"] for r in results.values()],
            "MAE": [r["mae"] for r in results.values()],
            "RMSE": [r["rmse"] for r in results.values()],
        }).sort_values("R2 Score", ascending=False).reset_index(drop=True)

        st.subheader("Model performance comparison")
        st.dataframe(
            comp_df.style.format({"R2 Score": "{:.4f}", "MAE": "{:,.2f}", "RMSE": "{:,.2f}"}),
            use_container_width=True,
        )

        fig5, ax5 = plt.subplots(figsize=(8, 4))
        sns.barplot(x="R2 Score", y="Algorithm", data=comp_df, ax=ax5, palette="viridis")
        ax5.set_xlim(0, 1)
        st.pyplot(fig5)

        best_model = comp_df.iloc[0]["Algorithm"]
        st.success(f"🏆 Best performing model: **{best_model}** (R² = {comp_df.iloc[0]['R2 Score']:.4f})")

    # ---------------- Predict ----------------
    with tab_predict:
        if "results" not in st.session_state:
            st.warning("Please open the 'Model Comparison' tab first to train the models.")
            return

        results = st.session_state["results"]

        model_choice = st.selectbox("Choose a model for prediction", list(results.keys()))

        st.subheader("Enter house features")
        c1, c2 = st.columns(2)
        with c1:
            sqft = st.number_input("Square Footage", min_value=0.0,
                                    value=float(df["Square_Footage"].median()))
            year_built = st.number_input("Year Built", min_value=1800, max_value=2026,
                                          value=int(df["Year_Built"].median()))
            bedrooms = st.number_input("Num Bedrooms", min_value=0,
                                        value=int(df["Num_Bedrooms"].median()))
            bathrooms = st.number_input("Num Bathrooms", min_value=0,
                                         value=int(df["Num_Bathrooms"].median()))
        with c2:
            lot_size = st.number_input("Lot Size", min_value=0.0,
                                        value=float(df["Lot_Size"].median()))
            garage_size = st.number_input("Garage Size", min_value=0,
                                           value=int(df["Garage_Size"].median()))
            neighborhood_quality = st.number_input("Neighborhood Quality", min_value=0,
                                                     value=int(df["Neighborhood_Quality"].median()))

        if st.button("Predict Price", type="primary"):
            model = results[model_choice]["model"]
            input_df = pd.DataFrame([{
                "Square_Footage": sqft,
                "Year_Built": year_built,
                "Num_Bedrooms": bedrooms,
                "Num_Bathrooms": bathrooms,
                "Lot_Size": lot_size,
                "Garage_Size": garage_size,
                "Neighborhood_Quality": neighborhood_quality,
            }])
            prediction = model.predict(input_df)[0]
            st.metric("Predicted House Price", f"${prediction:,.2f}")

        st.divider()
        st.caption("Predict with all models at once")
        if st.button("Compare all model predictions"):
            input_df = pd.DataFrame([{
                "Square_Footage": sqft,
                "Year_Built": year_built,
                "Num_Bedrooms": bedrooms,
                "Num_Bathrooms": bathrooms,
                "Lot_Size": lot_size,
                "Garage_Size": garage_size,
                "Neighborhood_Quality": neighborhood_quality,
            }])
            preds = {name: r["model"].predict(input_df)[0] for name, r in results.items()}
            pred_df = pd.DataFrame({
                "Algorithm": list(preds.keys()),
                "Predicted Price": [f"${v:,.2f}" for v in preds.values()],
            })
            st.dataframe(pred_df, use_container_width=True)


if __name__ == "__main__":
    main()
