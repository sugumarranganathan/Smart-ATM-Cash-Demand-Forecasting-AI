import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

# ==========================================
# Load Trained Model & Scaler
# ==========================================

MODEL_PATH = "atm_cash_lstm_model.keras"
SCALER_PATH = "cash_scaler.pkl"
DEFAULT_DATASET = "Smart_ATM_Cash_Demand_Dataset.csv"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ==========================================
# Load Default Dataset
# ==========================================

default_df = pd.read_csv(DEFAULT_DATASET)

# Convert transaction time to datetime
default_df["transactionTime"] = pd.to_datetime(
    default_df["transactionTime"]
)

# Sort data
default_df = default_df.sort_values(
    ["atmId", "transactionTime"]
).reset_index(drop=True)

# ==========================================
# ATM List
# ==========================================

ATM_LIST = sorted(default_df["atmId"].unique().tolist())

# ==========================================
# Dataset Loader
# ==========================================

def load_dataset(uploaded_file):

    if uploaded_file is None:
        df = default_df.copy()
    else:
        df = pd.read_csv(uploaded_file)

        df["transactionTime"] = pd.to_datetime(
            df["transactionTime"]
        )

        df = df.sort_values(
            ["atmId", "transactionTime"]
        ).reset_index(drop=True)

    return df


# ==========================================
# Latest 24 Records
# ==========================================

def latest_24_hours(df, atm_id):

    atm_df = df[df["atmId"] == atm_id].copy()

    atm_df = atm_df.sort_values(
        "transactionTime"
    )

    if len(atm_df) < 24:
        return None

    return atm_df.tail(24)


# ==========================================
# Prepare Input Sequence
# ==========================================

def prepare_sequence(atm_df):

    values = atm_df["totalOutcome"].values.reshape(-1,1)

    scaled = scaler.transform(values)

    return scaled.reshape(1,24,1)


# ==========================================
# Recursive Forecast
# ==========================================

def recursive_forecast(sequence, hours=24):

    predictions = []

    current = sequence.copy()

    for i in range(hours):

        pred_scaled = model.predict(
            current,
            verbose=0
        )[0][0]

        pred = scaler.inverse_transform(
            [[pred_scaled]]
        )[0][0]

        predictions.append(float(pred))

        current = np.append(
            current[:,1:,:],
            [[[pred_scaled]]],
            axis=1
        )

    return predictions


# ==========================================
# Forecast Summary
# ==========================================

def forecast_summary(predictions):

    total = np.sum(predictions)

    peak_hour = np.argmax(predictions)+1

    low_hour = np.argmin(predictions)+1

    avg = np.mean(predictions)

    return total, peak_hour, low_hour, avg


# ==========================================
# Create Forecast DataFrame
# ==========================================

def create_forecast_dataframe(predictions):

    forecast_df = pd.DataFrame({
        "Hour": [f"Hour {i}" for i in range(1, 25)],
        "Predicted Cash Withdrawal": np.round(predictions, 2)
    })

    return forecast_df


# ==========================================
# Forecast Chart
# ==========================================

def create_chart(predictions):

    plt.figure(figsize=(10, 4))

    plt.plot(
        range(1, 25),
        predictions,
        marker="o",
        linewidth=2
    )

    plt.title("Next 24 Hour ATM Cash Withdrawal Forecast")

    plt.xlabel("Forecast Hour")

    plt.ylabel("Cash Withdrawal")

    plt.xticks(range(1, 25))

    plt.grid(True)

    plt.tight_layout()

    return plt.gcf()


# ==========================================
# Current ATM Status
# ==========================================

def current_status(atm_df):

    latest = atm_df.iloc[-1]

    status = f"""
ATM ID : {latest['atmId']}

ATM Name : {latest['atmName']}

City : {latest['atmCity']}

Current Balance : ₹ {latest['totalBalance']:,.2f}

Latest Hourly Withdrawal : ₹ {latest['totalOutcome']:,.2f}

Transactions : {int(latest['totalNumberTransaction'])}

Last Updated : {latest['transactionTime']}
"""

    return status


# ==========================================
# Prediction Function
# ==========================================

def predict(atm_id, uploaded_file):

    try:

        df = load_dataset(uploaded_file)

        atm_df = latest_24_hours(df, atm_id)

        if atm_df is None:

            return (
                "❌ Selected ATM has fewer than 24 records.",
                None,
                None,
                "",
                "",
                "",
                ""
            )

        sequence = prepare_sequence(atm_df)

        predictions = recursive_forecast(
            sequence,
            hours=24
        )

        forecast_df = create_forecast_dataframe(
            predictions
        )

        chart = create_chart(
            predictions
        )

        total, peak, low, avg = forecast_summary(
            predictions
        )

        status = current_status(
            atm_df
        )

        total_text = f"₹ {total:,.2f}"

        peak_text = f"Hour {peak}"

        low_text = f"Hour {low}"

        avg_text = f"₹ {avg:,.2f}"

        return (
            status,
            forecast_df,
            chart,
            total_text,
            peak_text,
            low_text,
            avg_text
        )

    except Exception as e:

        return (
            f"❌ Error : {str(e)}",
            None,
            None,
            "",
            "",
            "",
            ""
        )


# ==========================================
# Gradio User Interface
# ==========================================

with gr.Blocks(
    title="Smart ATM Cash Demand Forecasting",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
# 🏧 Smart ATM Cash Demand Forecasting Using LSTM

Predict the next **24 hours ATM cash withdrawal** using a trained LSTM model.

### Features
- ✅ Built-in Dataset
- ✅ Upload Your Own CSV
- ✅ ATM-wise Prediction
- ✅ Automatic Latest 24 Hours Selection
- ✅ Next 24 Hours Forecast
- ✅ Forecast Graph
"""
    )

    with gr.Row():

        atm_dropdown = gr.Dropdown(
            choices=ATM_LIST,
            value=ATM_LIST[0],
            label="Select ATM ID"
        )

        upload_csv = gr.File(
            label="Upload CSV (Optional)",
            file_types=[".csv"],
            type="filepath"
        )

    predict_btn = gr.Button(
        "🚀 Predict Next 24 Hours",
        variant="primary"
    )

    gr.Markdown("---")

    status_box = gr.Textbox(
        label="ATM Status",
        lines=8,
        interactive=False
    )

    forecast_table = gr.Dataframe(
        headers=[
            "Hour",
            "Predicted Cash Withdrawal"
        ],
        label="Forecast Table",
        interactive=False
    )

    forecast_chart = gr.Plot(
        label="Forecast Chart"
    )

    gr.Markdown("## 📊 Forecast Summary")

    with gr.Row():

        total_box = gr.Textbox(
            label="Total Expected Withdrawal",
            interactive=False
        )

        avg_box = gr.Textbox(
            label="Average Hourly Withdrawal",
            interactive=False
        )

    with gr.Row():

        peak_box = gr.Textbox(
            label="Peak Demand Hour",
            interactive=False
        )

        low_box = gr.Textbox(
            label="Lowest Demand Hour",
            interactive=False
        )

    predict_btn.click(
        fn=predict,
        inputs=[
            atm_dropdown,
            upload_csv
        ],
        outputs=[
            status_box,
            forecast_table,
            forecast_chart,
            total_box,
            peak_box,
            low_box,
            avg_box
        ]
    )

    gr.Markdown(
        """
---
### 📌 Model Information

- Model : LSTM
- Input Sequence : Last 24 Hours
- Forecast Horizon : Next 24 Hours
- Target Variable : **totalOutcome**
- Framework : TensorFlow / Keras
- Interface : Gradio
"""
    )

# ==========================================
# Launch App
# ==========================================

demo.launch()
