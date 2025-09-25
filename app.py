import streamlit as st
from transformers import pipeline
import pandas as pd

st.set_page_config(page_title="Hate Speech Detector", layout="centered")

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification", 
        model="./saved_model", 
        tokenizer="./saved_model", 
        return_all_scores=True
    )

pipe = load_model()

st.title("🚨 Hate Speech Detector")
st.write("Enter text and the model will classify it as **Hate Speech**, **Offensive Language**, or **No Hate and Offensive**")

user_input = st.text_area("✍️ Type your text here:", height=150)

if st.button("Check"):
    if user_input.strip():
        preds = pipe(user_input)[0]
        top = max(preds, key=lambda x: x['score'])
        st.success(f"**Prediction:** {top['label']}  (Confidence: {top['score']:.3f})")

        df = pd.DataFrame(preds).set_index("label")
        st.subheader("Class probabilities")
        st.bar_chart(df["score"])
    else:
        st.warning("⚠️ Please enter some text.")
