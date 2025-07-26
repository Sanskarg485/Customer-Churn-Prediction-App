import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load encoders and scaler
scaler = joblib.load("C:\\Users\\Sanskar Gupta\\OneDrive\\Desktop\\ML Projects\\Customer Churn\\scaler.pkl")
le_geo = joblib.load("C:\\Users\\Sanskar Gupta\\OneDrive\\Desktop\\ML Projects\\Customer Churn\\le_geo.pkl")
le_gender = joblib.load("C:\\Users\\Sanskar Gupta\\OneDrive\\Desktop\\ML Projects\\Customer Churn\\le_gender.pkl")

# Load models
models = {
    "Logistic Regression": joblib.load("C:\\Users\\Sanskar Gupta\\OneDrive\\Desktop\\ML Projects\\Customer Churn\\logistic_regression_model.pkl"),
    "Random Forest": joblib.load("C:\\Users\\Sanskar Gupta\\OneDrive\\Desktop\\ML Projects\\Customer Churn\\random_forest_model.pkl"),
    "Gradient Boosting": joblib.load("C:\\Users\\Sanskar Gupta\\OneDrive\\Desktop\\ML Projects\\Customer Churn\\gradient_boosting_model.pkl")
}

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("🔍 Customer Churn Prediction App")

# Sidebar - Model Selection
model_name = st.sidebar.selectbox("Choose a Model", list(models.keys()))
model = models[model_name]

# User Inputs
st.markdown("### Enter Customer Information:")
credit_score = st.number_input("Credit Score", 300, 900, 600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0, step=100.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x else "No")
is_active_member = st.radio("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x else "No")
estimated_salary = st.number_input("Estimated Salary", 0.0, 300000.0, 50000.0, step=100.0)

if st.button("Predict Churn"):
    # Preprocess input
    input_df = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': le_geo.transform([geography]),
        'Gender': le_gender.transform([gender]),
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction == 1:
        st.error(f"⚠️ Churn Risk: HIGH\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Churn Risk: LOW\n\nProbability: {probability:.2f}")
