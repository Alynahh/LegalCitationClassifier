import streamlit as st
import joblib

# Load model
model = joblib.load("legal_model.pkl")

# Page settings
st.set_page_config(page_title="Legal Citation Classifier", page_icon="⚖️", layout="centered")

# Custom CSS for background, input box, button, and layout
st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
    }
    .stApp {
        background: linear-gradient(135deg, #f8f9fa, #d6e4f0);
        padding: 2rem;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 42px;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 18px;
        color: #34495e;
        text-align: center;
        margin-bottom: 2rem;
    }
    .footer {
        margin-top: 4rem;
        text-align: center;
        font-size: 13px;
        color: #7f8c8d;
    }

    /* Make text area stand out */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
        border: 1px solid #ced4da !important;
        border-radius: 8px;
        padding: 12px;
        font-size: 15px;
    }

    /* Style the predict button */
    .stButton>button {
        background-color: #2c3e50 !important;
        color: white !important;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        font-size: 15px;
        border: none;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #1a252f !important;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="title">⚖️ Legal Citation Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a legal citation sentence below to predict its citation class using AI.</div>', unsafe_allow_html=True)

# Input box
user_input = st.text_area("📜 Type your legal sentence here:", height=140)

# Predict button
if st.button("🔍 Predict Citation Type"):
    if user_input.strip():
        prediction = model.predict([user_input])[0]
        st.success(f"📌 **Predicted Class:** `{prediction}`")
    else:
        st.warning("Please enter a sentence to get a prediction.")

# Footer with clickable links
st.markdown("""
    <div class="footer">
        Crafted with 🤍 by Aleena Mehmood | AI Undergraduate · 
        <a href="https://github.com/Alynahh" target="_blank">GitHub</a> • 
        <a href="https://www.linkedin.com/in/aleenamehmood/" target="_blank">LinkedIn</a>
    </div>
""", unsafe_allow_html=True)
