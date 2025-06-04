import streamlit as st
import pandas as pd
import os
import requests
import time
from openai import AzureOpenAI
from streamlit_echarts import st_echarts

def predict_ir_swaption(input_df):
    with st.spinner("Calling ML Model..."):
        try:
            payload = {
                "input_data": {
                    "columns": input_df.columns.tolist(),
                    "index": [0],
                    "data": input_df.values.tolist()
                }
            }

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
            elapsed_time = round(end_time - start_time, 2)

            response.raise_for_status()
            result = response.json()
            return result[0] if isinstance(result, list) else result

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Model call failed: {e}")
            return {"error": str(e)}


# --- Mock model predictors per product ---
def predict_bond(input_data):
    return "Level 1"

def predict_capfloor(input_data):
    return "Level 3"

def predict_irswap(input_data):
    return "Level 2"

def predict_by_product(product_type, input_data):
    if product_type == "IR Swaption":
        return predict_ir_swaption(input_data)
    elif product_type == "Bond":
        return predict_bond(input_data)
    elif product_type == "CapFloor":
        return predict_capfloor(input_data)
    elif product_type == "IRSwap":
        return predict_irswap(input_data)
    else:
        return "Unknown"

def get_secret(key, default=""):
    return os.getenv(key, st.secrets.get(key, default))

# --- Streamlit App Layout ---
st.set_page_config(page_title="Augur - Fair Value Classification Model",layout="centered")

st.title("Augur - Fair Value Classification Model")
st.markdown("""
We leverage advanced machine learning techniques to predict the **fair value classification of financial instruments** in accordance with **IFRS 13** guidelines. 
Model prediction is grounded with observability-based risk assessments and analytical review for robust internal validation and reporting.
> ⚠️ **Responsible AI notice**: This model is intended for **internal reporting and review **. Model predictions should be complemented with expert judgment for external disclosures.
""")
# --- Workflow 
single_tab, batch_tab, rationale_tab = st.tabs(["Trade Inference", "Batch Inference", "Analytical Review"])

with single_tab:
    st.subheader("Predict Fair Value Classification for a Single trade")
    with st.sidebar:
        st.markdown("### Trade Details")
        product_type = st.selectbox("Product Type", ["IR Swaption", "Bond", "CapFloor", "IRSwap"], index=0)
        st.session_state["product_type"] = product_type

        currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"])
        notional = st.number_input("Notional", min_value=1_000_000, step=1_000_000, value=10_000_000)

        if product_type in ["IR Swaption", "CapFloor", "IRSwap"]:
            option_type = st.selectbox("Option Type", ["Receiver", "Payer"])
            strike = st.slider("Strike (%)", 0.0, 10.0, 2.5, 0.1)
            expiry_tenor = st.selectbox("Expiry Tenor (Y)", [2, 3, 5, 10])
            maturity_tenor = st.selectbox("Maturity Tenor (Y)", [5, 10, 15, 20, 30])
        elif product_type == "Bond":
            option_type = "N/A"
            strike = 0.0
            expiry_tenor = 0
            maturity_tenor = st.selectbox("Maturity Tenor (Y)", [1, 2, 3, 5, 10, 30])
            rating = st.selectbox("Credit Rating", ["AAA", "AA", "A", "BBB", "BB", "B"])
            issuer_type = st.selectbox("Issuer Type", ["Government", "Corporate"])

        trade_inputs = {
            "product_type": product_type,
            "currency": currency,
            "notional": notional,
            "option_type": option_type,
            "strike": strike,
            "expiry_tenor": expiry_tenor,
            "maturity_tenor": maturity_tenor,
        }

        if product_type == "Bond":
            trade_inputs.update({
                "rating": rating,
                "issuer_type": issuer_type
            })

    input_data = pd.DataFrame([trade_inputs])

    if st.button("Run Single Trade Inference"):
        try:
            start_time = time.time()
            result = predict_by_product(product_type, input_data)
            end_time = time.time()
            st.session_state["model_pred"] = result
            st.session_state["ML_Model_elapsed_time"] = round(end_time - start_time, 2)
            
            with st.expander("Model Input", expanded=False):
                st.subheader("📦 Input JSON Payload")
                st.json(input_data.to_dict(orient="records")[0])

            with st.expander("Model Details", expanded=False):
                st.markdown(f"⏱️ Model run completed in {st.session_state['ML_Model_elapsed_time']} seconds")
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
                
            st.success(f"✅ Predicted Fair Value Classification: {result}")
            st.markdown("""
                        ⚠️ **Responsible AI Notice:** Fair value level predicted by Model is for internal reporting only and should not be used for external or regulatory disclosure.
                        Please use Augmented Prediction for external reporting.
                        """)
        except Exception as e:
            st.error(f"❌ Mock model failed: {e}")


# --- Batch Inference and Analytical Review ---

with batch_tab:
    st.subheader("Predict Fair Value Classification for a set of trades")
    st.markdown("Upload a CSV file with trade deatils - Supported Products - IR Swaption, Bond, CapFloor, IRSwap")

    uploaded_file = st.file_uploader("Upload CSV", type="csv", key="batch")
    if uploaded_file:
        df_infer = pd.read_csv(uploaded_file)
        required_cols = ["product_type", "currency", "option_type", "notional", "strike", "expiry_tenor", "maturity_tenor"]
        if all(col in df_infer.columns for col in required_cols):
            with st.spinner("Running batch inference..."):
                try:
                    df_infer["Predicted IFRS13 Level"] = df_infer.apply(
                        lambda row: predict_by_product(row["product_type"], row.to_frame().T), axis=1
                    )
                    st.success("✅ Inference completed!")
                    st.dataframe(df_infer.head(11))

                    st.download_button("📅 Download Results", data=df_infer.to_csv(index=False), file_name="predicted_results.csv")
                except Exception as e:
                    st.error(f"❌ Batch mock model failed: {e}")
        else:
            st.warning(f"CSV must include columns: {', '.join(required_cols)}")

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

                        st.subheader("Heatmap: Predicted Fair value Level by Trading Desk")
                        st_echarts(option, height="400px")
        else:
                    st.warning(f"CSV must include columns: {', '.join(required_cols)}")


with rationale_tab:
    st.subheader("Analytical review")
    if st.button("Run GPT-4o Rationale"):
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
