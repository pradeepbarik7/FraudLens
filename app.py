
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="FraudLens | Fraud Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PREMIUM DARK FINTECH THEME
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0B1120;
    --panel: #111A2E;
    --panel-light: #162238;
    --border: #26344B;
    --text: #F1F5F9;
    --muted: #94A3B8;
    --green: #10B981;
    --red: #F87171;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(11, 17, 32, 0.9);
}

[data-testid="stSidebar"] {
    background: #0E1729;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1, h2, h3 {
    color: var(--text) !important;
    letter-spacing: -0.5px;
}

p, label, .stMarkdown {
    color: #CBD5E1;
}

.hero {
    background: linear-gradient(120deg, #13243A, #102C2B);
    border: 1px solid #284A4A;
    border-radius: 20px;
    padding: 30px 32px;
    margin: 6px 0 26px 0;
}

.hero-title {
    font-size: 34px;
    line-height: 1.2;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 10px;
}

.hero-desc {
    color: #A8BACB;
    font-size: 15px;
    line-height: 1.7;
}

.eyebrow {
    color: #34D399;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

.section-label {
    color: #94A3B8;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
}

div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 19px 20px;
    border-radius: 16px;
    min-height: 125px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 13px;
}

div[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
    font-weight: 800;
}

div[data-testid="stMetricDelta"] {
    font-size: 12px;
}

div[data-testid="stFileUploader"] {
    background: #111A2E;
    border: 1px dashed #3A526C;
    padding: 16px;
    border-radius: 16px;
}

.stButton button,
.stDownloadButton button {
    background: #10B981;
    color: #04130F !important;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 42px;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background: #34D399;
    border: none;
    color: #04130F !important;
}

.stSlider [data-baseweb="slider"] {
    margin-top: 10px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

div[data-testid="stTabs"] button {
    color: #A8B7CA;
}

hr {
    border-color: var(--border);
}

.small-note {
    color: #94A3B8;
    font-size: 12px;
    line-height: 1.7;
}

.status-pill {
    display: inline-block;
    padding: 6px 11px;
    background: #123B31;
    color: #6EE7B7;
    border: 1px solid #24634F;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# EXISTING MODEL AND DATA LOADING — UNCHANGED
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("/content/fraud_model.joblib")


@st.cache_data
def load_data():
    return pd.read_csv("/content/test_data.csv")


model = load_model()
test_df = load_data()
required_columns = list(model.feature_names_in_)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 25px 0;">
        <div style="font-size: 30px;">🔎</div>
        <div style="font-size: 25px; font-weight: 800; color: #F8FAFC;">
            Fraud<span style="color:#10B981;">Lens</span>
        </div>
        <div style="font-size: 11px; color: #94A3B8; letter-spacing: 1px;">
            FRAUD INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Navigation")
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Transaction Analysis",
            "Model Performance"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Detection Settings")

    threshold = st.slider(
        "Fraud Detection Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="A lower threshold flags more transactions as potential fraud."
    )

    st.caption(f"Current threshold: {threshold:.0%}")

    st.markdown("---")

    st.markdown("""
    <div class="small-note">
        <b style="color:#10B981;">● System information</b><br><br>
        Model: Logistic Regression<br>
        Task: Credit Card Fraud Detection<br>
        Mode: Local ML inference
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">FRAUD DETECTION PLATFORM</div>
    <div class="hero-title">Fraud intelligence,<br>made actionable.</div>
    <div class="hero-desc">
        Analyze transaction data, identify suspicious activity, and
        review model performance in one workspace.
    </div>
    <br>
    <span class="status-pill">● ML MODEL LOADED</span>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SHARED PREDICTION FUNCTION
# Existing prediction logic is retained.
# --------------------------------------------------
def predict_transactions(uploaded_df):
    missing = [
        col for col in required_columns
        if col not in uploaded_df.columns
    ]

    if missing:
        st.error("Required columns are missing:")
        st.write(missing)
        return

    # Select features in the exact order expected by the model.
    X_uploaded = uploaded_df[required_columns].copy()

    # Check for invalid or missing numeric values.
    for col in required_columns:
        X_uploaded[col] = pd.to_numeric(
            X_uploaded[col], errors="coerce"
        )

    if X_uploaded.isna().any().any():
        st.error(
            "Some required features contain missing or non-numeric "
            "values. Please clean your CSV."
        )
        return

    if not X_uploaded.map(
        lambda value: float("-inf") < value < float("inf")
    ).all().all():
        st.error(
            "Your CSV contains infinite or invalid values."
        )
        return

    # EXISTING MODEL INFERENCE — UNCHANGED
    probabilities = model.predict_proba(X_uploaded)[:, 1]

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

    total = len(results)
    fraud_count = int(predictions.sum())
    legitimate_count = int((predictions == 0).sum())

    c1, c2, c3 = st.columns(3)

    c1.metric("Transactions Analyzed", f"{total:,}")
    c2.metric("Potential Fraud Alerts", f"{fraud_count:,}")
    c3.metric("Legitimate Predictions", f"{legitimate_count:,}")

    if fraud_count:
        st.warning(
            f"{fraud_count:,} transaction(s) were flagged for review. "
            "This is a model prediction, not a confirmed fraud finding."
        )
    else:
        st.info("No transactions were flagged at the current threshold.")

    st.markdown("### Prediction Results")

    tab1, tab2 = st.tabs(["All Results", "Flagged Transactions"])

    with tab1:
        st.dataframe(
            results,
            use_container_width=True,
            height=380
        )

    with tab2:
        flagged = results[
            results["Predicted Class"] == 1
        ]
        if flagged.empty:
            st.info("No potential fraud transactions at this threshold.")
        else:
            st.dataframe(
                flagged,
                use_container_width=True,
                height=320
            )

    # EXISTING CSV DOWNLOAD — RETAINED
    csv_data = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Predictions as CSV",
        data=csv_data,
        file_name="fraudlens_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )


# --------------------------------------------------
# TRANSACTION ANALYSIS
# --------------------------------------------------
if page == "Transaction Analysis":
    st.markdown("## Transaction Analysis")
    st.markdown(
        "Upload a transaction CSV to generate predictions using your "
        "existing trained model."
    )

    st.markdown("### Upload dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Your file must contain all model input features."
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            if uploaded_df.empty:
                st.error("Your CSV file is empty.")
            else:
                st.markdown("#### File preview")
                st.caption(
                    f"{len(uploaded_df):,} rows · "
                    f"{len(uploaded_df.columns)} columns"
                )

                st.dataframe(
                    uploaded_df.head(8),
                    use_container_width=True
                )

                predict_transactions(uploaded_df)

        except Exception as e:
            st.error(f"Unable to process this CSV: {e}")


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------
elif page == "Overview":
    st.markdown("## Dashboard Overview")
    st.markdown(
        "Your workspace for transaction risk analysis and model insights."
    )

    # Keep the original test dataset evaluation.
    X_test = test_df[required_columns]
    y_test = test_df["Class"]

    scores = model.predict_proba(X_test)[:, 1]
    y_pred = (scores >= threshold).astype(int)

    fraud_count = int(y_pred.sum())
    legitimate_count = int((y_pred == 0).sum())

    st.markdown("### Test Dataset Snapshot")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Test Transactions",
        f"{len(test_df):,}"
    )
    c2.metric(
        "Potential Fraud Alerts",
        f"{fraud_count:,}"
    )
    c3.metric(
        "Legitimate Predictions",
        f"{legitimate_count:,}"
    )
    c4.metric(
        "Detection Threshold",
        f"{threshold:.0%}"
    )

    st.markdown("---")
    st.markdown("### Prediction Distribution")

    chart_col, info_col = st.columns([1.4, 1])

    with chart_col:
        distribution = pd.DataFrame({
            "Prediction": ["Legitimate", "Potential Fraud"],
            "Transactions": [legitimate_count, fraud_count]
        }).set_index("Prediction")

        st.bar_chart(
            distribution,
            color="#10B981"
        )

    with info_col:
        st.markdown("#### Current configuration")

        st.markdown(f"""
        <div style="
            background:#111A2E;
            border:1px solid #26344B;
            border-radius:15px;
            padding:20px;
            line-height:2.1;
        ">
            <span style="color:#94A3B8;">Model</span><br>
            <b>Logistic Regression</b><br>
            <span style="color:#94A3B8;">Threshold</span><br>
            <b>{threshold:.0%}</b><br>
            <span style="color:#94A3B8;">Evaluation rows</span><br>
            <b>{len(test_df):,}</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Quick Actions")

    action_col1, action_col2 = st.columns(2)

    with action_col1:
        st.info(
            "**Analyze transactions**\n\n"
            "Open Transaction Analysis from the sidebar to upload a CSV."
        )

    with action_col2:
        st.info(
            "**Review model performance**\n\n"
            "Open Model Performance to inspect evaluation metrics."
        )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
elif page == "Model Performance":
    st.markdown("## Model Performance")
    st.markdown(
        "Evaluation on the existing test dataset using the current threshold."
    )

    X_test = test_df[required_columns]
    y_test = test_df["Class"]

    scores = model.predict_proba(X_test)[:, 1]
    y_pred = (scores >= threshold).astype(int)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Precision", f"{precision:.2%}")
    c2.metric("Recall", f"{recall:.2%}")
    c3.metric("F1 Score", f"{f1:.2%}")

    st.markdown("---")
    st.markdown("### Confusion Matrix")

    matrix = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"]
    )

    st.dataframe(
        matrix_df,
        use_container_width=True
    )

    tn, fp, fn, tp = matrix.ravel()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("True Negatives", f"{tn:,}")
    c2.metric("False Positives", f"{fp:,}")
    c3.metric("False Negatives", f"{fn:,}")
    c4.metric("True Positives", f"{tp:,}")

    st.markdown("---")
    st.markdown("### Metric interpretation")

    st.markdown("""
    - **Precision:** Of the transactions flagged as potential fraud,
      how many were actually fraud in the labeled test dataset?
    - **Recall:** Of the actual fraud cases in the labeled test dataset,
      how many did the model flag?
    - **F1 Score:** A combined measure of precision and recall.
    - **Confusion Matrix:** Shows correct and incorrect classifications.
    """)

    st.caption(
        "These metrics are calculated from the existing labeled test dataset. "
        "They are not verified banking decisions or guarantees of fraud."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:12px 0;">
    <div style="font-size:15px; font-weight:800; color:#10B981;">
        🔎 FraudLens
    </div>
    <div class="small-note">
        AI-powered credit card fraud detection · Educational prototype
    </div>
    <div class="small-note">
        Predictions require human review and are not guaranteed to be correct.
    </div>
</div>
""", unsafe_allow_html=True)