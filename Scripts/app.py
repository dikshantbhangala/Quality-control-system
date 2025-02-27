import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("models/defect_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("🛠️ Manufacturing Defect Prediction")

# User inputs
temperature = st.number_input("Enter Machine Temperature")
pressure = st.number_input("Enter Machine Pressure")
machine_speed = st.number_input("Enter Machine Speed")
operator_experience = st.number_input("Enter Operator Experience (years)")

# Predict button
if st.button("Predict Defect"):
    input_data = np.array([[temperature, pressure, machine_speed, operator_experience]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    st.success(f"🔍 Predicted Defect Type: {prediction[0]}")
