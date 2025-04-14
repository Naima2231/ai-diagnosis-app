import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("xgboost_model.pkl")

# Title and description
st.title("Autoimmune Disease Diagnosis AI")
st.write("Enter lab test results and symptoms to predict RA, SLE, or Healthy.")

# Lab test inputs (numeric)
CRP = st.number_input("CRP (C-Reactive Protein)", min_value=0.0, max_value=100.0, value=10.0)
ESR = st.number_input("ESR (Erythrocyte Sedimentation Rate)", min_value=0.0, max_value=100.0, value=20.0)
RF = st.number_input("Rheumatoid Factor (RF)", min_value=0.0, max_value=500.0, value=14.0)
ANA = st.number_input("ANA (Antinuclear Antibodies)", min_value=0.0, max_value=5.0, value=1.0)
Anti_CCP = st.number_input("Anti-CCP", min_value=0.0, max_value=300.0, value=15.0)

# Symptom inputs (categorical: Yes/No → 1/0)
Joint_Pain = st.selectbox("Joint Pain", ["Yes", "No"])
Fatigue = st.selectbox("Fatigue", ["Yes", "No"])
Butterfly_Rash = st.selectbox("Butterfly Rash", ["Yes", "No"])
Fever = st.selectbox("Fever", ["Yes", "No"])
Morning_Stiffness = st.selectbox("Morning Stiffness", ["Yes", "No"])
Photosensitivity = st.selectbox("Photosensitivity", ["Yes", "No"])

# Map inputs to features in correct format
features = {
    "CRP": CRP,
    "ESR": ESR,
    "RF": RF,
    "ANA": ANA,
    "Anti_CCP": Anti_CCP,
    "Joint_Pain": 1 if Joint_Pain == "Yes" else 0,
    "Fatigue": 1 if Fatigue == "Yes" else 0,
    "Butterfly_Rash": 1 if Butterfly_Rash == "Yes" else 0,
    "Fever": 1 if Fever == "Yes" else 0,
    "Morning_Stiffness": 1 if Morning_Stiffness == "Yes" else 0,
    "Photosensitivity": 1 if Photosensitivity == "Yes" else 0,
}

# Create input dataframe with the correct column order
input_df = pd.DataFrame([features])
input_df = input_df[['CRP', 'ESR', 'RF', 'ANA', 'Anti_CCP',
                     'Joint_Pain', 'Fatigue', 'Butterfly_Rash',
                     'Fever', 'Morning_Stiffness', 'Photosensitivity']]

# Predict button
if st.button("Predict Diagnosis"):
    prediction = model.predict(input_df)[0]
    st.subheader("Prediction Result:")
    if prediction == 0:
        st.success("✅ Diagnosis: Healthy")
    elif prediction == 1:
        st.warning("⚠️ Diagnosis: Rheumatoid Arthritis (RA)")
    elif prediction == 2:
        st.error("🚨 Diagnosis: Systemic Lupus Erythematosus (SLE)")
    else:
        st.info("Unknown prediction.")
