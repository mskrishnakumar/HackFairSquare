import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import date
from openai import AzureOpenAI
from Observability_Stress_Module import (
    run_full_observability_stress_test,
    simulate_greeks,
    ir_delta_stress_test,
    vol_risk_stress_test,
    generate_trade_pv_and_risk_pvs,
    ois_curve_map
)
from workflow_styles import (
    get_workflow_css,
    get_workflow_html_ml,
    get_workflow_html_rf,
    get_workflow_html_rat
)

# --- Initialize Session State ---
for key in ["ml_done", "rf_done", "rat_done"]:
    if key not in st.session_state:
        st.session_state[key] = False

# --- Load Observability Grids ---
ir_grid = pd.read_csv("ir_delta_observability_grid.csv")
ir_grid.columns = ir_grid.columns.str.strip()
ir_grid["Observable Tenor (Years)"] = pd.to_numeric(ir_grid["Observable Tenor (Years)"], errors="coerce")
vol_grid = pd.read_csv("volatility_observability_grid.csv")
vol_grid.columns = vol_grid.columns.str.strip()

# --- Page Config ---
st.set_page_config(page_title="FAIR&SQ - IFRS13 Fair Value Classification Model", layout="wide")

# --- Header Layout ---
# col1, col2 = st.columns([1, 3])

# with col1:
#     st.image("fair_value_logo.svg", width=240)

# with col2:
st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@700&display=swap');

        .title {
            font-family: 'Roboto Slab', serif;
            font-size: 40px;
            color: #007BFF;  /* Bootstrap primary blue */
            margin-bottom: 0;
        }

        .subtitle {
            font-size: 20px;
            color: gray;
            margin-top: 5px;
        }
        </style>

        <div class="title">IFRS13 Fair Value Classification Model</div>
        <div class="subtitle">Fair & Square in every decision.</div>
""", unsafe_allow_html=True)

# --- Workflow CSS ---
st.markdown(get_workflow_css(), unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("Trade Details")
product_type = st.sidebar.selectbox("Product Type", ["IR Swaption", "Bond", "CapFloor", "IRSwap"], index=0)
notional = st.sidebar.number_input("Notional", min_value=1_000_000, step=1_000_000, value=10_000_000)
currency = st.sidebar.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"])
option_type = st.sidebar.selectbox("Option Type", ["Receiver", "Payer"])
strike = st.sidebar.slider("Strike (%)", 0.0, 10.0, 2.5, 0.1)
expiry_tenor = st.sidebar.selectbox("Expiry Tenor (Y)", [2, 3, 5,10])
expiry_date = date.today().replace(year=date.today().year + expiry_tenor)
#st.sidebar.markdown(f"Expiry Date: {expiry_date.strftime('%d-%b-%Y')}") # Don't show date in sidebar for now
maturity_tenor = st.sidebar.selectbox("Maturity Tenor (Y)", [5, 10, 15, 20, 30])
maturity_date = date.today().replace(year=date.today().year + maturity_tenor)
#st.sidebar.markdown(f"Maturity Date: {maturity_date.strftime('%d-%b-%Y')}") # Don't show date in sidebar for now

# --- Trade Dictionary ---
trade = {
    "product_type": product_type,
    "currency": currency,
    "option_type": option_type,
    "expiry_tenor": expiry_tenor,
    "maturity_tenor": maturity_tenor,
    "strike": strike,
    "notional": notional
}

# --- Model Prediction Mock ---
def mock_model_prediction(trade):
    if trade["expiry_tenor"] < 5 and trade["maturity_tenor"] < 15 and trade["strike"] < 3.0:
        return "Level 2"
    return "Level 3"

# --- Azure GPT-4o Client ---
def get_rationale_from_gpt(ir_summary, vol_summary, model_pred):
    client = AzureOpenAI(
        api_key=st.secrets["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-01",
        azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial analyst who explains IFRS 13 classification decisions "
                "clearly and concisely based on input data. Do not include background information "
                "about IFRS 13. Focus strictly on justifying the classification result."
            )
        },
        {
            "role": "user",
            "content": (
                f"IR Delta Observability Summary: {ir_summary}\n\n"
                f"Volatility Observability Summary: {vol_summary}\n\n"
                f"Model Predicted Level: {model_pred}\n\n"
                "Analyze Model predicted IFRS 13 level and risk factor observability results (using 10% threshold on trade PV) and decide on the IFRS 13 level for the trade. "
                "Display the chosen level in bold and provide a brief, direct justification for the predicted IFRS 13 level. "
                "Conclude with a single line that summarizes confidence level High, Medium or Low and the reasoning or notes confidence in the classification."
            )
        }
    ]
    response = client.chat.completions.create(
        model=st.secrets["AZURE_OPENAI_MODEL"],
        messages=messages,
        temperature=0.5
    )
    return response.choices[0].message.content

import requests

def call_azure_ml_model(trade):
    # Format the input in Azure expected tabular format
    payload = {
        "input_data": {
            "columns": [
                "product_type",
                "currency",
                "option_type",
                "notional",
                "strike",
                "expiry_tenor",
                "maturity_tenor"
            ],
            "index": [0],
            "data": [[
                trade["product_type"],
                trade["currency"],
                trade["option_type"],
                trade["notional"],
                trade["strike"],
                trade["expiry_tenor"],
                trade["maturity_tenor"]
            ]]
        }
    }

    endpoint = os.getenv("AZURE_ML_ENDPOINT", st.secrets.get("AZURE_ML_ENDPOINT", ""))
    api_key =  os.getenv("AZURE_ML_API_KEY", st.secrets.get("AZURE_ML_API_KEY", ""))

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(endpoint, headers=headers, json=payload)
    response.raise_for_status()  # Raise an error if bad response

    result = response.json()
    return result[0]  # e.g., "Level 3"

# --- Section: Machine Learning Model Prediction ---
with st.container(border=True):
    st.subheader("1. Machine Learning Model Prediction")

    # Stepper
    step = 3 if st.session_state.get("ml_done") else 0
    st.markdown(get_workflow_html_ml(step), unsafe_allow_html=True)

    if st.button("\u25B6 Run Model Inference", key="run_ml"):
        # Build Payload
        payload = {
            "input_data": {
                "columns": [
                    "product_type", "currency", "option_type",
                    "notional", "strike", "expiry_tenor", "maturity_tenor"
                ],
                "index": [0],
                "data": [[
                    trade["product_type"],
                    trade["currency"],
                    trade["option_type"],
                    trade["notional"],
                    trade["strike"],
                    trade["expiry_tenor"],
                    trade["maturity_tenor"]
                ]]
            }
        }

        # Store payload in session for reuse
        st.session_state["model_payload"] = payload
        with st.spinner("Running model..."):
            start_time = time.time()  # Start timer
        # 🔁 Call endpoint
        try:
            endpoint = st.secrets["AZURE_ML_ENDPOINT"]
            api_key = st.secrets["AZURE_ML_API_KEY"]
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            end_time = time.time()  # End timer
            elapsed = round(end_time - start_time, 4)  # Time in seconds
            st.session_state["model_output"] = result
            st.session_state["model_pred"] = result[0]
            st.session_state["ifrs13_level"] = result[0]
            st.session_state["ml_done"] = True
            st.session_state["ML_Model_elapsed_time"] = elapsed
            st.rerun()

        except Exception as e:
            st.error(f"🚨 Model call failed: {e}")

    # === Subsections ===
    if st.session_state.get("ml_done"):

        with st.expander(" Model Input Payload", expanded=False):
            st.code(json.dumps(st.session_state["model_payload"], indent=2), language="json")

        
        with st.expander("Machine Learning Model Details", expanded=False):
            st.markdown(
                """
                <div style='font-weight: bold; font-size: 18px; margin-bottom: 10px;'>📘 Machine Learning Model Details</div>
                <div style='text-align: left; overflow-y: auto; max-height: 250px;
                            padding: 10px; background-color: #111111;
                            border-radius: 8px; color: #f0f0f0;
                            font-family: monospace; font-size: 14px;'>
                    <strong>Model:</strong> Gradient Boosting (AutoML)<br>
                    <strong>Version:</strong> Gradient Boosting (AutoML)<br>
                    <strong>Trained on:</strong> Synthetic IR Swaption Trades<br>
                    <strong>Features:</strong> product_type, currency, option_type, notional, strike, expiry_tenor, maturity_tenor<br>
                    <strong>Accuracy:</strong> 86.2%<br>
                    <strong>AUC:</strong> 0.74<br>
                    <strong>Last Trained:</strong> 01-Jun-2025<br><br>
                    This model predicts IFRS13 Level based on risk and trade characteristics.<br>
                    AutoML handled class imbalance and feature engineering.
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.expander(" Model Inference Result", expanded=False):
            st.markdown(f"🕒 Model run completed in {st.session_state['ML_Model_elapsed_time']} seconds.")
            st.code(json.dumps(st.session_state["model_output"], indent=2), language="json")
            st.success(f"✅ Predicted IFRS13 Level: {st.session_state['model_pred']}")


if "model_pred" in st.session_state:
    st.sidebar.markdown(f"""
    <div style='background-color:#d4edda;color:#155724;padding:10px;
    border-left:5px solid #28a745;border-radius:5px;margin-top:10px;
    font-weight:bold'>Model Predicted IFRS13 Level:<br>{st.session_state['model_pred']}</div>
    """, unsafe_allow_html=True)

# --- Section: Risk Factor-based Inference ---
with st.container(border=True):
    st.subheader("2. Risk Factor Observability testing")
    step = 4 if st.session_state.rf_done else 0
    st.markdown(get_workflow_html_rf(step), unsafe_allow_html=True)

    if st.button("\u25B6 Run Risk Factor Inference Workflow"):
        greeks = simulate_greeks(trade)
        trade["trade_pv"], generated_pvs = generate_trade_pv_and_risk_pvs(greeks)
        greeks.update(generated_pvs)

        # ✅ Save to session state for persistent view
        st.session_state["greeks"] = greeks
        st.session_state["generated_pvs"] = generated_pvs

        st.success("✅ Risk factors and PV contributions simulated")

        # Run stress tests
        st.success("✅ IR Delta Observability Test Completed")
        ir_stressed, ir_report, ir_stress_pv, ir_msgs = ir_delta_stress_test(trade, greeks)
        st.session_state["ir_summary"] = ir_msgs
        st.dataframe(pd.DataFrame(ir_report).T)

        st.success("✅ Volatility Observability Test Completed")
        vol_stressed, vol_report, vol_stress_pv, vol_msgs = vol_risk_stress_test(trade, greeks)
        st.session_state["vol_summary"] = vol_msgs
        st.dataframe(pd.DataFrame(vol_report).T)

        total_stress_pv = ir_stress_pv + vol_stress_pv
        final_level = "Level 3" if total_stress_pv > 0.1 * trade["trade_pv"] else "Level 2"
        st.metric("Total Stress PV", total_stress_pv)

        st.session_state["ir_report_df"] = pd.DataFrame(ir_report).T
        st.session_state["vol_report_df"] = pd.DataFrame(vol_report).T
        st.session_state["trade_pv"] = trade["trade_pv"]
        st.session_state["ir_stress_pv"] = ir_stress_pv
        st.session_state["vol_stress_pv"] = vol_stress_pv


        if final_level == "Level 3":
            st.error("🔴 Unobservable risk exceeds 10% of total PV  → Level 3")
        else:
            st.success("🟢 Unobservable risk within threshold → Level 2")

        st.session_state["final_level"] = final_level
        st.session_state.rf_done = True
        st.rerun()

    # ✅ Always show stored greeks and PV breakdown
    if "greeks" in st.session_state and "generated_pvs" in st.session_state:
        with st.expander("Simulated Risk Factors", expanded=False):
            df_greeks = pd.DataFrame.from_dict(
                {k: str(v) for k, v in st.session_state["greeks"].items()},
                orient="index",
                columns=["Value"]
            )
            st.dataframe(df_greeks)

        with st.expander("PV Contribution by Risk Factors", expanded=False):
            pv_df = pd.DataFrame.from_dict(
                {k: str(v) for k, v in st.session_state["generated_pvs"].items()},
                orient="index",
                columns=["PV"]
            )
            st.dataframe(pv_df)
    if "ir_report_df" in st.session_state:
        with st.expander(" IR Delta Observability Test Results", expanded=False):
            st.dataframe(st.session_state["ir_report_df"])
            
    if "vol_report_df" in st.session_state:
        with st.expander(" Volatility Observability Test Results", expanded=False):
            st.dataframe(st.session_state["vol_report_df"])
            
    
    # --- PV and Stress Test Summary Box ---
        with st.container():
            st.subheader("Observability Summary")
        
        # Validate keys exist
        required_keys = ["trade_pv", "ir_stress_pv", "vol_stress_pv", "final_level"]
        if all(k in st.session_state for k in required_keys):
            trade_pv = st.session_state["trade_pv"]
            ir_stress_pv = st.session_state["ir_stress_pv"]
            vol_stress_pv = st.session_state["vol_stress_pv"]
            total_stress_pv = ir_stress_pv + vol_stress_pv
            stress_pct = (total_stress_pv / trade_pv) * 100 if trade_pv else 0

            col1, col2, col3 = st.columns(3)
            col1.metric(" Trade PV", f"{trade_pv:,.2f}")
            col2.metric(" Unobservable PV component", f"{total_stress_pv:,.2f}")
            col3.metric(" Unobservable % of Total PV", f"{stress_pct:.2f}%")

            col2.metric(" IR Stress PV", f"{ir_stress_pv:,.2f}")
            col2.metric(" Volatility Stress PV", f"{vol_stress_pv:,.2f}")
            st.metric(" Observability Level", st.session_state["final_level"])
        # else:
        #     st.warning("Observability stress results not available.")

    
# Update sidebar with both predictions

if "final_level" in st.session_state:
    st.sidebar.markdown(f"""
    <div style='background-color:#f8d7da;color:#721c24;padding:10px;
    border-left:5px solid #dc3545;border-radius:5px;margin-top:10px;
    font-weight:bold'>Risk Factor based IFRS13 Level:<br>{st.session_state['final_level']}</div>
    """, unsafe_allow_html=True)
# --- Section: Rationale Explanation ---
with st.container(border=True):
    st.subheader("3. Analytical Review and Rationale Generation")
    step = 3 if st.session_state.get("rat_done") else 0
    st.markdown(get_workflow_html_rat(step), unsafe_allow_html=True)

    
    if st.button("\u25B6 Run Rationale Generation Workflow"):
      if all(k in st.session_state for k in ["ir_summary", "vol_summary", "model_pred"]):
        st.session_state["rat_done"] = True  # ✅ Mark early
        rationale = get_rationale_from_gpt(
            ir_summary="\n".join(st.session_state["ir_summary"]),
            vol_summary="\n".join(st.session_state["vol_summary"]),
            model_pred=st.session_state["model_pred"]
        )
        st.session_state["rationale_text"] = rationale
        st.rerun()  # ✅ Trigger UI update
      else:
          st.warning("Run both ML and Risk Factor workflows before generating rationale.")
    if "rationale_text" in st.session_state:
            st.markdown(
                f"""<div style='background-color:#f1f1f1;color:#000000;padding:10px;
                border-left:5px solid #0078D7;border-radius:5px'>
                <b>Explanation:</b><br>{st.session_state["rationale_text"]}</div>""",
                unsafe_allow_html=True
            )