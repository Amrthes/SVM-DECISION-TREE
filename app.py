import streamlit as st
import pandas as pd
import pickle

# Load models
svm_model = pickle.load(open("svm_model.pkl", "rb"))
dt_model = pickle.load(open("dt_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("📊 Customer Churn Prediction App")

st.write("Enter customer details below to predict churn:")

# User input
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=55.0)
tenure_months = st.number_input("Tenure (months)", min_value=0, max_value=100, value=24)
contract_type = st.selectbox("Contract Type", options=[0, 1, 2])  # 0=Month-to-Month, 1=One year, 2=Two year
support_calls = st.number_input("Support Calls", min_value=0, max_value=20, value=1)

# Create DataFrame
new_customer = pd.DataFrame({
    'monthly_charges': [monthly_charges],
    'tenure_months': [tenure_months],
    'contract_type': [contract_type],
    'support_calls': [support_calls]
})

# Predict button
if st.button("Predict"):
    # SVM needs scaled data
    new_customer_scaled = scaler.transform(new_customer)

    # Individual model predictions
    svm_pred = svm_model.predict(new_customer_scaled)[0]
    dt_pred = dt_model.predict(new_customer)[0]

    # Probabilities for ensemble
    svm_prob = svm_model.predict_proba(new_customer_scaled)[0][1]
    dt_prob = dt_model.predict_proba(new_customer)[0][1]
    avg_prob = (svm_prob + dt_prob) / 2
    ensemble_pred = 1 if avg_prob >= 0.5 else 0

    # Display results
    st.subheader("🔮 Predictions")
    st.write(f"**SVM Model:** {'Churn' if svm_pred==1 else 'No Churn'} (Prob = {svm_prob:.2f})")
    st.write(f"**Decision Tree Model:** {'Churn' if dt_pred==1 else 'No Churn'} (Prob = {dt_prob:.2f})")
    st.write(f"✅ **Ensemble Final Prediction:** {'Churn' if ensemble_pred==1 else 'No Churn'} (Avg Prob = {avg_prob:.2f})")
