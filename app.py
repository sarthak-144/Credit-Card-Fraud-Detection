import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")

# --- LOAD SAVED MODELS ---
@st.cache_resource
def load_resources():
    # Load the test data and models we saved earlier
    X_test, y_test = joblib.load('data_test.joblib')
    
    models = {
        'Logistic Regression': joblib.load('models/model_lr.joblib'),
        'Random Forest': joblib.load('models/model_rf.joblib'),
        'XGBoost': joblib.load('models/model_xgb.joblib')
    }
    return X_test, y_test, models

try:
    with st.spinner("Loading Models..."):
        X_test, y_test, models = load_resources()
except FileNotFoundError:
    st.error("Model files not found. Please run 'train.py' first.")
    st.stop()

# --- SIDEBAR & LOGIC (Same as before) ---
st.sidebar.header("⚙️ User Controls")
model_name = st.sidebar.selectbox("Choose a Model", list(models.keys()))
selected_model = models[model_name]
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.3, 0.01) # Default to your optimized 0.3

# ... (The rest of the UI code remains exactly the same) ...
# Get probabilities
y_proba = selected_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= threshold).astype(int)

prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("Precision", f"{prec:.2%}")
col2.metric("Recall", f"{rec:.2%}")
col3.metric("F1 Score", f"{f1:.2%}")

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)