from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

st.set_page_config(
    page_title="FraudLens",
    page_icon="🔎",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(str(Path(__file__).parent / "fraud_model.joblib"))

@st.cache_data
def load_data():
    return pd.read_csv(Path(__file__).parent / "test_data.csv")

model = load_model()
test_df = load_data()
required_columns = list(model.feature_names_in_)

st.title("🔎 FraudLens")
st.write("Credit Card Fraud Detection Dashboard")

st.divider()
st.subheader("Upload Your Own Transactions")

threshold = st.slider(
    "Fraud Detection Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)

        if uploaded_df.empty:
            st.error("Your CSV file is empty.")

        else:
            missing = [
                col for col in required_columns
                if col not in uploaded_df.columns
            ]

            if missing:
                st.error("Required columns are missing:")
                st.write(missing)

            else:
                # Select features in the exact order expected by the model.
                X_uploaded = uploaded_df[required_columns].copy()

                # Check for invalid or missing numeric values.
                for col in required_columns:
                    X_uploaded[col] = pd.to_numeric(
                        X_uploaded[col], errors="coerce"
                    )

                if X_uploaded.isna().any().any():
                    st.error(
                        "Some required features contain missing "
                        "or non-numeric values. Please clean your CSV."
                    )

                elif not X_uploaded.map(
                    lambda value: float("-inf") < value < float("inf")
                ).all().all():
                    st.error(
                        "Your CSV contains infinite or invalid values."
                    )

                else:
                    probabilities = model.predict_proba(
                        X_uploaded
                    )[:, 1]

                    predictions = (
                        probabilities >= threshold
                    ).astype(int)

                    results = uploaded_df.copy()
                    results["Fraud Probability"] = probabilities.round(4)
                    results["Predicted Class"] = predictions
                    results["Prediction"] = pd.Series(
                        predictions
                    ).map({
                        0: "Legitimate",
                        1: "Potential Fraud"
                    }).values

                    st.success("Predictions generated successfully!")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Transactions", len(results))
                    col2.metric(
                        "Potential Fraud Alerts",
                        int(predictions.sum())
                    )
                    col3.metric(
                        "Legitimate Predictions",
                        int((predictions == 0).sum())
                    )

                    st.subheader("Prediction Results")
                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    csv_data = results.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv_data,
                        file_name="fraudlens_predictions.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"Unable to process this CSV: {e}")

st.divider()
st.subheader("Original Test Dataset")

X_test = test_df[required_columns]
y_test = test_df["Class"]

scores = model.predict_proba(X_test)[:, 1]
y_pred = (scores >= threshold).astype(int)

col1, col2, col3 = st.columns(3)
col1.metric("Test Transactions", len(test_df))
col2.metric(
    "Precision",
    f"{precision_score(y_test, y_pred, zero_division=0):.2%}"
)
col3.metric(
    "Recall",
    f"{recall_score(y_test, y_pred, zero_division=0):.2%}"
)

matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

st.write("Confusion Matrix")
st.dataframe(
    pd.DataFrame(
        matrix,
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"]
    ),
    use_container_width=True
)

st.caption(
    "Educational prototype only. Predictions are not verified "
    "banking decisions or guarantees of fraud."
)

