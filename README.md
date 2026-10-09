# 🔎 FraudLens — Credit Card Fraud Detection

FraudLens is a machine learning web application that identifies potentially fraudulent credit card transactions using a trained classification model. It provides an interactive interface to upload transaction data, generate predictions, and evaluate model performance.

## 🚀 Live Demo

**Try FraudLens:** [Open Live App] https://thefraudlens.streamlit.app/

## ✨ Features

* **CSV Upload:** Upload transaction data for fraud detection.
* **Fraud Prediction:** Classify transactions as potentially fraudulent or legitimate.
* **Adjustable Threshold:** Change the prediction threshold to explore different fraud detection outcomes.
* **Prediction Results:** View transaction-level predictions and fraud alerts.
* **Download Results:** Export predictions for further analysis.
* **Model Evaluation:** Review precision, recall, F1-score, and confusion matrix metrics using the available test dataset.
* **Interactive Dashboard:** Use the Streamlit interface to explore results.

## 🧠 Machine Learning Model

FraudLens uses a trained **Logistic Regression** classification model with **StandardScaler** preprocessing.

The model was trained using an anonymized credit card transaction dataset. Since fraudulent transactions are much less common than legitimate transactions, class balancing was used during training.

## 🛠️ Tech Stack

* **Language:** Python
* **Web App:** Streamlit
* **Data Processing:** Pandas
* **Machine Learning:** Scikit-learn
* **Model Persistence:** Joblib

## 📂 Project Structure

```text
FraudLens/
├── app.py
├── fraud_model.joblib
├── test_data.csv
├── requirements.txt
└── README.md
```

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/pradeepbarik7/FraudLens.git
cd FraudLens
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 📊 Evaluation

The application provides evaluation metrics such as precision, recall, F1-score, and a confusion matrix on the available test dataset.

Fraud detection involves a trade-off between detecting fraudulent transactions and incorrectly flagging legitimate ones. Results can vary depending on the prediction threshold and dataset.

## ⚠️ Limitations

* This project is intended for educational and demonstration purposes.
* Predictions depend on the training data and the quality of the input CSV.
* Model performance on the available dataset does not guarantee performance on real-world transactions.
* The application should not be used as the sole basis for real financial decisions.

## 🔮 Future Improvements

* Evaluate the model on additional datasets.
* Compare Logistic Regression with other classification algorithms.
* Improve validation and error handling for uploaded CSV files.
* Add more detailed model evaluation and monitoring.

## 👨‍💻 Author

**Pradeep Barik**

* GitHub: [@pradeepbarik7](https://github.com/pradeepbarik7)
* Project Repository: [FraudLens](https://github.com/pradeepbarik7/FraudLens)

---

If you find this project useful, consider giving the repository a ⭐.
