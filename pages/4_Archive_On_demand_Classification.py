import streamlit as st
import pandas as pd
import os
import requests
import time
from openai import AzureOpenAI
from streamlit_echarts import st_echarts

st.set_page_config(page_title="On-Demand IFRS13 Classification", layout="wide")

# --- Unified Light Theme Styling ---
st.markdown("""
<style>
    html, body, [class*="css"] {
        background-color: #f5f6fa;
        color: #212529;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: #f5f6fa;
        color: #212529;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #212529 !important;
    }
    button[kind="primary"] {
        background-color: #2563eb !important;
        color: white !important;
        font-weight: 600;
    }
    .center-content > div {
        max-width: 900px;
        margin: auto;
    }
</style>
<div class="center-content">
""", unsafe_allow_html=True)

st.title("🔎 On-Demand IFRS13 Fair Value Classification")

def get_secret(key, default=""):
    return os.getenv(key, st.secrets.get(key, default))

single_tab, batch_tab, rationale_tab = st.tabs(["  Single Trade Inference", "  Batch Inference", "  Analytical Review"])

with single_tab:
    st.subheader("🔹 Single Trade Inference")
    with st.sidebar:
        st.markdown("### 🧾 Single Trade Details")
        product_type = st.selectbox("Product Type", ["IR Swaption", "Bond", "CapFloor", "IRSwap"], index=0)
        currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"])
        option_type = st.selectbox("Option Type", ["Receiver", "Payer"])
        notional = st.number_input("Notional", min_value=1_000_000, step=1_000_000, value=10_000_000)
        strike = st.slider("Strike (%)", 0.0, 10.0, 2.5, 0.1)
        st.caption("💡 Adjust the strike rate for the trade scenario.")
        expiry_tenor = st.selectbox("Expiry Tenor (Y)", [2, 3, 5, 10])
        maturity_tenor = st.selectbox("Maturity Tenor (Y)", [5, 10, 15, 20, 30])

    input_data = pd.DataFrame([{
        "product_type": product_type,
        "currency": currency,
        "option_type": option_type,
        "notional": notional,
        "strike": strike,
        "expiry_tenor": expiry_tenor,
        "maturity_tenor": maturity_tenor
    }])

    if st.button("▶ Run Single Trade Inference"):
        payload = {
            "input_data": {
                "columns": input_data.columns.tolist(),
                "index": [0],
                "data": input_data.values.tolist()
            }
        }
        with st.spinner("Calling ML Model..."):
            try:
                start_time = time.time()
                response = requests.post(
                    url=get_secret("AZURE_ML_ENDPOINT"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {get_secret('AZURE_ML_API_KEY')}"
                    },
                    json=payload
                )
                end_time = time.time()
                result = response.json()
                st.session_state["model_pred"] = result[0]
                st.session_state["ML_Model_elapsed_time"] = round(end_time - start_time, 2)
                st.success(f"✅ Predicted IFRS13 Level: {result[0]}")

                with st.expander("📘 Model Details and Input", expanded=False):
                    st.markdown(f"⏱️ Model run completed in {st.session_state['ML_Model_elapsed_time']} seconds")
                    st.subheader("📦 Input JSON Payload")
                    st.json(payload)
                    st.markdown("""
                    <div style='text-align: left; padding: 10px; background-color: #eeeeee; border-radius: 8px; color: #000000; font-family: monospace; font-size: 14px;'>
                    <strong>Model:</strong> Gradient Boosting (AutoML)<br>
                    <strong>Version:</strong> Gradient Boosting (AutoML)<br>
                    <strong>Trained on:</strong> Synthetic IR Swaption Trades<br>
                    <strong>Features:</strong> product_type, currency, option_type, notional, strike, expiry_tenor, maturity_tenor<br>
                    <strong>Accuracy:</strong> 86.2%<br>
                    <strong>AUC:</strong> 0.74<br>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Model call failed: {e}")

with batch_tab:
    st.subheader("🔸 Batch Inference")
    st.markdown("Upload a CSV file with trade inputs to run inference on multiple trades.")

    uploaded_file = st.file_uploader("Upload CSV", type="csv", key="batch")
    if uploaded_file:
        df_infer = pd.read_csv(uploaded_file)
        required_cols = ["product_type", "currency", "option_type", "notional", "strike", "expiry_tenor", "maturity_tenor"]
        if all(col in df_infer.columns for col in required_cols):
            payload = {
                "input_data": {
                    "columns": required_cols,
                    "index": list(range(len(df_infer))),
                    "data": df_infer[required_cols].values.tolist()
                }
            }
            with st.spinner("Running batch inference..."):
                try:
                    response = requests.post(
                        url=get_secret("AZURE_ML_ENDPOINT"),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {get_secret('AZURE_ML_API_KEY')}"
                        },
                        json=payload
                    )
                    result = response.json()
                    df_infer["Predicted IFRS13 Level"] = result
                    st.success("✅ Inference completed!")
                    st.dataframe(df_infer.head(11))

                    # --- Development-only Visualization ---
                    if "trading_desk" in df_infer.columns:
                        heatmap_data = df_infer.groupby(["trading_desk", "Predicted IFRS13 Level"]).size().reset_index(name="count")
                        rows = heatmap_data["trading_desk"].unique().tolist()
                        cols = heatmap_data["Predicted IFRS13 Level"].unique().tolist()

                        row_map = {v: i for i, v in enumerate(rows)}
                        col_map = {v: i for i, v in enumerate(cols)}

                        data = [[col_map[c], row_map[r], int(v)] for r, c, v in heatmap_data.values]

                        option = {
                            "tooltip": {"position": "top"},
                            "grid": {"height": "50%", "top": "10%"},
                            "xAxis": {"type": "category", "data": cols, "splitArea": {"show": True}},
                            "yAxis": {"type": "category", "data": rows, "splitArea": {"show": True}},
                            "visualMap": {
                                "min": 0,
                                "max": max(heatmap_data["count"]),
                                "calculable": True,
                                "orient": "horizontal",
                                "left": "center",
                                "bottom": "15%",
                            },
                            "series": [
                                {
                                    "name": "Trade Count",
                                    "type": "heatmap",
                                    "data": data,
                                    "label": {"show": True},
                                    "emphasis": {
                                        "itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0, 0, 0, 0.5)"}
                                    },
                                }
                            ],
                        }

                        st.subheader("Heatmap: IFRS13 Level by Trading Desk")
                        st_echarts(option, height="400px")

                    st.download_button("📅 Download Results", data=df_infer.to_csv(index=False), file_name="predicted_results.csv")
                except Exception as e:
                    st.error(f"❌ Model call failed: {e}")
        else:
            st.warning(f"CSV must include columns: {', '.join(required_cols)}")

with rationale_tab:
    st.subheader("🧫 Rationale Explanation")
    if st.button("▶ Run GPT-4o Rationale"):
        if all(k in st.session_state for k in ["ir_summary", "vol_summary", "model_pred"]):
            st.session_state["rat_done"] = True
            client = AzureOpenAI(
                api_key=get_secret("AZURE_OPENAI_API_KEY"),
                api_version="2024-02-01",
                azure_endpoint=get_secret("AZURE_OPENAI_ENDPOINT")
            )
            messages = [
                {"role": "system", "content": "You're a financial analyst..."},
                {"role": "user", "content": (
                    f"IR Delta Summary:\n{st.session_state['ir_summary']}\n\n"
                    f"Vol Summary:\n{st.session_state['vol_summary']}\n\n"
                    f"Model Prediction: {st.session_state['model_pred']}\n"
                    "Explain and confirm IFRS13 classification with confidence score."
                )}
            ]
            response = client.chat.completions.create(
                model=get_secret("AZURE_OPENAI_MODEL"),
                messages=messages,
                temperature=0.5
            )
            st.session_state["rationale_text"] = response.choices[0].message.content
            st.rerun()
        else:
            st.warning("⚠️ Ensure both model inference and risk summaries are completed.")

    if "rationale_text" in st.session_state:
        st.success("✅ Rationale Generated")
        st.markdown(f"**Explanation:**\n\n{st.session_state['rationale_text']}")

# --- Close center content div ---
st.markdown("</div>", unsafe_allow_html=True)
