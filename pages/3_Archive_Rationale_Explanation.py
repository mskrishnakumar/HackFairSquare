import streamlit as st
import os
from openai import AzureOpenAI
from workflow_styles import get_workflow_css, get_workflow_html_rat

st.set_page_config(page_title="Rationale Generation", layout="wide")
st.markdown(get_workflow_css(), unsafe_allow_html=True)
st.title("3️⃣ Rationale Explanation via GPT")

# --- Secret loader ---
def get_secret(key, default=""):
    return os.getenv(key, st.secrets.get(key, default))

st.markdown(get_workflow_html_rat(3 if st.session_state.get("rat_done") else 0), unsafe_allow_html=True)

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
        st.warning("⚠️ Run both prior steps first!")

if "rationale_text" in st.session_state:
    st.success("✅ Rationale Generated")
    st.markdown(f"**Explanation:**\n\n{st.session_state['rationale_text']}")
