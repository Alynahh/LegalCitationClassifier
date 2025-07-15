# ⚖️ Legal Citation Classifier

This project is a machine learning-powered **legal citation classifier**, built to identify how one case refers to another (e.g., *cited*, *applied*, *followed*, *distinguished*, etc.) using Natural Language Processing.

## 💡 Motivation
Legal documents often reference prior judgments. This project aims to automate citation classification using AI — a helpful tool for legal research, summarization, and academic analysis.

## 🧠 Tech Stack
- Python
- Streamlit (for the web app)
- Scikit-learn (Logistic Regression model)
- TF-IDF Vectorizer
- Real-world legal citation dataset (Federal Court of Australia)

## 🚀 Demo
Paste a legal sentence and get its citation type prediction instantly:

> “The High Court applied the doctrine in Mabo v Queensland.”  
> **🔍 Predicted Class:** `applied`

## 🖥️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
