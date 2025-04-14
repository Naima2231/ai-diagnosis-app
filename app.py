import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("xgboost_model.pkl")

st.title("Autoimmune Disease Diagnosis AI")

st.markdown("This tool predicts autoimmune diseases like **RA**, **SLE**, or **Healthy** based on lab test results or symptoms.")

# ----------------------
# 🔬 SECTION 1: TEST RESULTS
# ----------------------
st.header("🔬 Lab Test-Based Diagnosis")

CRP = st.number_input("CRP (C-Reactive Protein)", min_value=0.0, max_value=100.0, value=10.0)
ESR = st.number_input("ESR (Erythrocyte Sedimentation Rate)", min_value=0.0, max_value=100.0, value=20.0)
RF = st.number_input("Rheumatoid Factor (RF)", min_value=0.0, max_value=500.0, value=14.0)
ANA = st.number_input("ANA (Antinuclear Antibodies)", min_value=0.0, max_value=5.0, value=1.0)
Anti_CCP = st.number_input("Anti-CCP", min_value=0.0, max_value=300.0, value=15.0)

if st.button("Predict from Lab Tests"):
    test_features = pd.DataFrame([{
        "CRP": CRP,
        "ESR": ESR,
        "RF": RF,
        "ANA": ANA,
        "Anti_CCP": Anti_CCP,
        "Joint_Pain": 0,
        "Fatigue": 0,
        "Butterfly_Rash": 0,
        "Fever": 0,
        "Morning_Stiffness": 0,
        "Photosensitivity": 0
    }])

    prediction = model.predict(test_features)[0]
    diagnosis_map = {
        0: "Healthy",
        1: "Rheumatoid Arthritis (RA)",
        2: "Systemic Lupus Erythematosus (SLE)"
    }
    result = diagnosis_map.get(prediction, "Unknown")
    st.success(f"🧪 Diagnosis from Lab Tests: **{result}**")


# ----------------------
# 🩺 SECTION 2: SYMPTOM CHECKER
# ----------------------
st.header("🩺 Symptom-Based Diagnosis")

Joint_Pain = st.selectbox("Joint Pain", ["Yes", "No"])
Fatigue = st.selectbox("Fatigue", ["Yes", "No"])
Butterfly_Rash = st.selectbox("Butterfly Rash", ["Yes", "No"])
Fever = st.selectbox("Fever", ["Yes", "No"])
Morning_Stiffness = st.selectbox("Morning Stiffness", ["Yes", "No"])
Photosensitivity = st.selectbox("Photosensitivity", ["Yes", "No"])

if st.button("Predict from Symptoms"):
    symptom_features = pd.DataFrame([{
        "CRP": 0,
        "ESR": 0,
        "RF": 0,
        "ANA": 0,
        "Anti_CCP": 0,
        "Joint_Pain": 1 if Joint_Pain == "Yes" else 0,
        "Fatigue": 1 if Fatigue == "Yes" else 0,
        "Butterfly_Rash": 1 if Butterfly_Rash == "Yes" else 0,
        "Fever": 1 if Fever == "Yes" else 0,
        "Morning_Stiffness": 1 if Morning_Stiffness == "Yes" else 0,
        "Photosensitivity": 1 if Photosensitivity == "Yes" else 0,
    }])

    prediction = model.predict(symptom_features)[0]
    diagnosis_map = {
        0: "Healthy",
        1: "Rheumatoid Arthritis (RA)",
        2: "Systemic Lupus Erythematosus (SLE)"
    }
    result = diagnosis_map.get(prediction, "Unknown")
    st.success(f"🩺 Diagnosis from Symptoms: **{result}**")
