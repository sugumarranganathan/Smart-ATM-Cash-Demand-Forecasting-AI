import gradio as gr
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

# ==========================
# Load Model & Dataset
# ==========================
MODEL_PATH = "atm_lstm_model.h5"
DATASET_PATH = "atm_transactions.csv"

model = load_model(MODEL_PATH)

df = pd.read_csv(DATASET_PATH)

# Load scaler if saved
try:
    scaler = joblib.load("scaler.pkl")
except:
    scaler = MinMaxScaler()
    scaler.fit(df[["Cash_Withdrawal"]])

# ==========================
# Prepare ATM List
# ==========================
atm_list = sorted(df["ATM_ID"].unique().tolist())


# ==========================
# Prediction Function
# ==========================
def predict(atm_id):

    atm_df = (
        df[df["ATM_ID"] == atm_id]
        .sort_values("Datetime")
        .tail(24)
        .copy()
    )

    if len(atm_df) < 24:
        return (
            "Not enough records for this ATM.",
            pd.DataFrame(),
            "",
            "",
            ""
        )

    current_cash = atm_df.iloc[-1]["Cash_Available"]

    values = atm_df["Cash_Withdrawal"].values.reshape(-1, 1)

    scaled = scaler.transform(values)

    X = scaled.reshape(1, 24, 1)

    pred_scaled = model.predict(X, verbose=0)

    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    result = pd.DataFrame({
        "Hour": [f"Hour {i}" for i in range(1, 25)],
        "Predicted Withdrawal": np.round(pred, 2)
    })

    total = f"₹ {pred.sum():,.2f}"
    peak = f"Hour {np.argmax(pred)+1}"
    low = f"Hour {np.argmin(pred)+1}"

    info = (
        f"ATM ID : {atm_id}\n"
        f"Current Cash : ₹ {current_cash:,.2f}\n"
        f"Latest Records Used : 24 Hours"
    )

    return info, result, total, peak, low


# ==========================
# Gradio UI
# ==========================
with gr.Blocks(title="Smart ATM Cash Demand Forecasting") as demo:

    gr.Markdown(
        """
# 🏧 Smart ATM Cash Demand Forecasting Using LSTM
Automatically loads the latest 24 hours of withdrawal data for the selected ATM and predicts the next 24 hours.
"""
    )

    atm = gr.Dropdown(
        choices=atm_list,
        label="Select ATM",
        value=atm_list[0]
    )

    predict_btn = gr.Button("Predict Next 24 Hours", variant="primary")

    gr.Markdown("## Current Status")

    status = gr.Textbox(label="", interactive=False)

    gr.Markdown("## Next 24 Hour Forecast")

    forecast = gr.Dataframe(
        headers=["Hour", "Predicted Withdrawal"],
        interactive=False
    )

    gr.Markdown("## Forecast Summary")

    total = gr.Textbox(label="Total Expected Withdrawal")

    peak = gr.Textbox(label="Peak Demand Hour")

    low = gr.Textbox(label="Lowest Demand Hour")

    predict_btn.click(
        fn=predict,
        inputs=atm,
        outputs=[
            status,
            forecast,
            total,
            peak,
            low
        ]
    )

demo.launch()
