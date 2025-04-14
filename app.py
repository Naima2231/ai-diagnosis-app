import streamlit as st
import joblib

model = joblib.load("xgboost_model.pkl")
st.write("🧠 Feature names in model:")
st.write(model.get_booster().feature_names)
