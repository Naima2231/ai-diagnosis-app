import streamlit as st
import pandas as pd
import joblib

# Title
st.title("Autoimmune Disease Diagnosis AI")

# Description
st.write("Enter symptoms and lab info to predict if the patient has RA, SLE, or is Healthy.")

# Load the trained model
model = joblib.load("xgboost_model.pkl")

# User input
age = st.slider("Age", 10, 90, 30)
joint_pain = st.selectbox("Joint Pain", ["Yes", "No"])
fatigue = st.selectbox("Fatigue", ["Yes", "No"])
fever = st.selectbox("Fever", ["Yes", "No"])
rash = st.selectbox("Rash", ["Yes", "No"])
photosensitivity = st.selectbox("Photosensitivity", ["Yes", "No"])

# Match the features exactly as trained
features = {
    'age': age,
    'joint_pain': 1 if joint_pain == "Yes" else 0,
    'fatigue': 1 if fatigue == "Yes" else 0,
    'fever': 1 if fever == "Yes" else 0,
    'rash': 1 if rash == "Yes" else 0,
    'photosensitivity': 1 if photosensitivity == "Yes" else 0,
}

# Create input DataFrame with correct column order
input_df = pd.DataFrame([features])
input_df = input_df[['age', 'joint_pain', 'fatigue', 'fever', 'rash', 'photosensitivity']]

# Predict when button clicked
if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    # Show result
    st.subheader("Prediction Result:")
    if prediction == 0:
        st.success("✅ Diagnosis: Healthy")
    elif prediction == 1:
        st.warning("⚠️ Diagnosis: Rheumatoid Arthritis (RA)")
    elif prediction == 2:
        st.error("🚨 Diagnosis: Systemic Lupus Erythematosus (SLE)")
    else:
        st.info("Unknown prediction.")



   
