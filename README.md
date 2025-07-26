# 💼 Customer Churn Prediction Web App

A Streamlit-based interactive web application that predicts the likelihood of a customer leaving a bank, using multiple trained machine learning models.

---

## 🚀 Project Overview

Customer churn is one of the most critical metrics in industries like banking, telecom, and retail. This project implements a predictive system using:

- ✅ Logistic Regression
- 🌲 Random Forest Classifier
- 🔥 Gradient Boosting Classifier

Users can enter customer details and select a model to get real-time churn prediction with risk probability.

---

## 🧠 Features

- Interactive Streamlit UI
- Model selection (Logistic, Random Forest, Gradient Boosting)
- Label encoding and feature scaling for preprocessing
- Real-time churn prediction with probability score
- Visual feedback: success or warning messages

---

## 🛠️ Tech Stack

- Python 3.x
- Streamlit
- Pandas & NumPy
- Scikit-learn
- Joblib (for model persistence)

---

## 🗂️ Files & Structure

```bash
.
├── app.py                      # Streamlit app script
├── logistic_regression_model.pkl
├── random_forest_model.pkl
├── gradient_boosting_model.pkl
├── scaler.pkl                 # StandardScaler
├── le_geo.pkl                 # LabelEncoder for Geography
├── le_gender.pkl              # LabelEncoder for Gender
├── requirements.txt
└── README.md

📈 Input Fields
Credit Score

Geography

Gender

Age

Tenure

Account Balance

Number of Products

Has Credit Card (Yes/No)

Active Membership (Yes/No)

Estimated Salary

📦 Trained Models
The models were trained on the Churn_Modelling.csv dataset with feature scaling and label encoding. They were exported using joblib for deployment.

🔐 Use Case
This solution helps businesses proactively identify high-risk customers and improve customer retention through timely interventions.

