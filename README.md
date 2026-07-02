                ATM Transaction Dataset
                         │
                         ▼
                 Data Preprocessing
     (Clean data, Sort by Time, Select ATM)
                         │
                         ▼
                 Feature Scaling
                  (MinMaxScaler)
                         │
                         ▼
            Prepare Last 24 Hours Sequence
                         │
                         ▼
                Train LSTM Model
                         │
                         ▼
             Predict Next 24 Hours
           (Recursive Forecasting)
                         │
                         ▼
             Forecast Results Generated
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Forecast Table  Forecast Chart  Forecast Summary















https://colab.research.google.com/drive/14i4T_8AwdDG0cdnon14CIqZhpOdTu242

https://sugumarai-smart-atm-cash-demand-forecast.hf.space


# 🏧 Smart ATM Cash Demand Forecasting Using LSTM

## 📌 Project Overview

Smart ATM Cash Demand Forecasting Using LSTM is a deep learning-based application that predicts the **next 24 hours of ATM cash withdrawal demand** using historical transaction data. The application helps banks estimate future cash requirements, enabling timely cash replenishment and reducing the chances of ATMs running out of cash.

The model is built using a Long Short-Term Memory (LSTM) neural network, which is well suited for time-series forecasting. Users can either use the built-in dataset or upload their own ATM transaction dataset to generate predictions.

---

# 🚨 Problem Statement

Banks replenish ATM cash using fixed schedules or manual estimation. This approach often results in:

- ATMs running out of cash during peak demand.
- Excess cash remaining in low-demand ATMs.
- Increased cash transportation costs.
- Poor customer experience due to cash unavailability.
- Difficulty predicting future cash withdrawal patterns.

An intelligent forecasting system is required to accurately estimate future ATM cash demand based on historical withdrawal data.

---

# 💡 Proposed Solution

This project uses a Long Short-Term Memory (LSTM) deep learning model to analyze historical ATM transaction data and forecast the next 24 hours of cash withdrawal demand.

The application automatically loads the latest 24 hours of transaction data and recursively predicts future withdrawal amounts. It also allows users to upload custom ATM datasets for forecasting.

This helps banks:

- Improve ATM cash availability
- Reduce operational costs
- Optimize cash replenishment schedules
- Improve customer satisfaction
- Support data-driven decision making

---

# Features

- ✅ Built-in ATM Dataset
- ✅ Upload Custom CSV Dataset
- ✅ ATM-wise Prediction
- ✅ Automatic Latest 24 Hours Selection
- ✅ Next 24 Hours Forecast
- ✅ Interactive Forecast Table
- ✅ Forecast Line Chart
- ✅ Forecast Summary
- ✅ Hugging Face Deployment
- ✅ Deep Learning Based Prediction

---

# 🧠 Deep Learning Model

- Model : LSTM (Long Short-Term Memory)
- Framework : TensorFlow / Keras
- Forecast Horizon : Next 24 Hours
- Input Sequence : Last 24 Hours
- Output : Next Hour Prediction (Recursive Forecasting)
- Target Variable : **totalOutcome**

---

# 📂 Dataset Information

Dataset contains historical ATM transaction records.

### Important Columns

| Column | Description |
|---------|-------------|
| transactionTime | Date and Time |
| atmId | ATM Identifier |
| atmName | ATM Name |
| atmCity | City |
| totalBalance | Current ATM Balance |
| totalOutcome | Cash Withdrawn |
| totalNumberTransaction | Total Transactions |

---

# ⚙️ Workflow

```
ATM Transaction Dataset
            │
            ▼
Data Preprocessing
            │
            ▼
Latest 24 Hours Selection
            │
            ▼
Feature Scaling
            │
            ▼
LSTM Model
            │
            ▼
Recursive Prediction
            │
            ▼
Next 24 Hours Forecast
            │
            ▼
Forecast Table & Graph
```

---

# 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- Gradio
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Hugging Face Spaces

---

# 📊 Application Interface

The application provides:

- Select ATM
- Upload CSV Dataset
- Predict Next 24 Hours
- ATM Information
- Forecast Table
- Forecast Chart
- Forecast Summary

---

# 📈 Forecast Output

The application predicts:

- Hour 1 Cash Withdrawal
- Hour 2 Cash Withdrawal
- ...
- Hour 24 Cash Withdrawal

It also displays:

- Total Expected Withdrawal
- Average Hourly Withdrawal
- Peak Demand Hour
- Lowest Demand Hour


# 📁 Project Structure

```
Smart_ATM_Cash_Demand_Forecasting/

│── app.py
│── requirements.txt
│── atm_cash_lstm_model.keras
│── cash_scaler.pkl
│── Smart_ATM_Cash_Demand_Dataset.csv
│── README.md
```

---

# 📷 Output

The application displays:

- ATM Information
- Forecast Table
- Forecast Graph
- Forecast Summary

---

# 🎯 Advantages

- Predicts future ATM cash demand
- Reduces ATM cash shortages
- Optimizes cash replenishment
- Saves operational costs
- Improves customer satisfaction
- Supports intelligent banking operations


# 👨‍💻 Prepared By

**R. Sugumar, M.B.A.,**

