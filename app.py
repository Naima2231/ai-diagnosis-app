import streamlit as st
import pandas as pd
import joblib

# Load your saved model
model = joblib.load('xgboost_model.pkl')

st.title("Autoimmune Disease Diagnosis AI")
st.write("Enter the patient's symptoms and get a prediction.")

# User input fields
age = st.number_input("Age", min_value=0, max_value=120, step=1)
joint_pain = st.selectbox("Joint Pain", ["No", "Yes"])
fatigue = st.selectbox("Fatigue", ["No", "Yes"])
fever = st.selectbox("Fever", ["No", "Yes"])
rash = st.selectbox("Rash", ["No", "Yes"])
photosensitivity = st.selectbox("Photosensitivity", ["No", "Yes"])

# Convert to model input format
features = {
    "Age": age,
    "Joint Pain": 1 if joint_pain == "Yes" else 0,
    "Fatigue": 1 if fatigue == "Yes" else 0,
    "Fever": 1 if fever == "Yes" else 0,
    "Rash": 1 if rash == "Yes" else 0,
    "Photosensitivity": 1 if photosensitivity == "Yes" else 0,
}

# Predict button
if st.button("Diagnose"):
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]
    if prediction == 0:
        st.success("Diagnosis: Rheumatoid Arthritis (RA)")
    elif prediction == 1:
        st.success("Diagnosis: Systemic Lupus Erythematosus (SLE)")
    else:
        st.success("Diagnosis: Healthy")
