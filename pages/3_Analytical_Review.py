import streamlit as st
from openai import AzureOpenAI
import os
import pandas as pd

st.set_page_config(page_title="Analytical Review - Augur", layout="centered")
st.title("Analytical Review")

st.subheader("Rationale Explanation Based on Model Prediction & Risk Factors")

if st.button("Explain Rationale using GPT model"):
    if all(k in st.session_state for k in ["ir_summary", "vol_summary", "model_pred"]):
        st.session_state["rat_done"] = True

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY", st.secrets.get("AZURE_OPENAI_API_KEY")),
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", st.secrets.get("AZURE_OPENAI_ENDPOINT"))
        )

        messages = [
            {"role": "system", "content": "You are a financial analyst. Provide a brief, clear explanation of the model's IFRS13 classification based on the following inputs."},
            {"role": "user", "content": (
                f"IR Delta Summary:\n{st.session_state['ir_summary']}\n\n"
                f"Vol Summary:\n{st.session_state['vol_summary']}\n\n"
                f"Model Prediction: {st.session_state['model_pred']}\n"
                "Provide a short rationale (2-3 lines) confirming or questioning the classification. Include confidence level."
            )}
        ]

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL", st.secrets.get("AZURE_OPENAI_MODEL")),
            messages=messages,
            temperature=0.3
        )

        st.session_state["rationale_text"] = response.choices[0].message.content
        st.rerun()
    else:
        st.warning("⚠️ Ensure both model inference and risk summaries are completed.")

if "rationale_text" in st.session_state:
    st.success("✅ Rationale Generated")
    st.markdown(f"**Explanation:**\n\n{st.session_state['rationale_text']}\n")

# --- Movement Over Time Section ---
st.markdown("---------------------------------------------------------------------------------------------------")
st.header("📈 Movement Over Time (MoT) Analysis")

def perform_mot_analysis(df_dec, df_mar):
    df_compare = pd.merge(df_dec, df_mar, on="trade_id", suffixes=("_dec", "_mar"))

    df_compare["Movement Summary"] = df_compare.apply(
        lambda row: f"No change (Level {row['Predicted IFRS13 Level_mar']})"
        if row['Predicted IFRS13 Level_dec'] == row['Predicted IFRS13 Level_mar']
        else f"Moved from {row['Predicted IFRS13 Level_dec']} to {row['Predicted IFRS13 Level_mar']}", axis=1)

    def get_mot_reason(old_level, new_level):
        if old_level == new_level:
            return "No change in classification — inputs remain consistent."
        transition = (old_level, new_level)
        reasons = {
            ("Level 3", "Level 2"): "Trade maturity has now entered the observable range.",
            ("Level 3", "Level 1"): "Market data inputs have become fully observable.",
            ("Level 2", "Level 1"): "Trade is now based entirely on quoted prices in active markets.",
            ("Level 1", "Level 2"): "Partial loss of observability in market inputs.",
            ("Level 2", "Level 3"): "Key valuation inputs have become unobservable.",
            ("Level 1", "Level 3"): "Significant deterioration in input observability."
        }
        return reasons.get(transition, "IFRS13 level changed due to re-evaluation of input observability.")

    df_compare["MOT Commentary"] = df_compare.apply(
        lambda row: get_mot_reason(row["Predicted IFRS13 Level_dec"], row["Predicted IFRS13 Level_mar"]),
        axis=1
    )

    return df_compare

uploaded_dec = st.file_uploader("Upload Trade Dataset 1 (with trade_id)", type=["csv"], key="dec")
uploaded_mar = st.file_uploader("Upload Trade Dataset (with trade_id)", type=["csv"], key="mar")

if uploaded_dec and uploaded_mar:
    df_dec = pd.read_csv(uploaded_dec)
    df_mar = pd.read_csv(uploaded_mar)

    if "trade_id" in df_dec.columns and "trade_id" in df_mar.columns and \
       "Predicted IFRS13 Level" in df_dec.columns and "Predicted IFRS13 Level" in df_mar.columns:
        result_df = perform_mot_analysis(df_dec, df_mar)
        st.success("✅ Movement Over Time Analysis Completed")

        # Show annotation
        st.markdown("---")
        st.markdown("### Trade-Level Commentary on Fair Value Classification Movements")

        # Show table
        st.dataframe(result_df)

        # Download button
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download MoT Analysis CSV", data=csv, file_name="mot_analysis_results.csv", mime="text/csv")
    else:
        st.error("❌ Both files must contain 'trade_id' and 'Predicted IFRS13 Level' columns.")
