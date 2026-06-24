import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "heart_attack_rf_model.pkl"

model = joblib.load(MODEL_PATH)


st.title("Heart Attack Prediction")

age = st.number_input("Age", 1, 120)
heart_rate = st.number_input("Heart Rate", 30, 250)
systolic = st.number_input("Systolic Blood Pressure", 50, 250)
diastolic = st.number_input("Diastolic Blood Pressure", 30, 150)
blood_sugar = st.number_input("Blood Sugar", 0, 500)
gender = st.selectbox("Gender", ["male", "female"])
ck_mb = st.number_input("CK-MB")
troponin = st.number_input("Troponin")

gender_encoded = 1 if gender == "male" else 0

if st.button("Predict"):
    data = pd.DataFrame([{
        "Age": age,
        "Gender": gender_encoded,
        "Heart rate": heart_rate,
        "Systolic blood pressure": systolic,
        "Diastolic blood pressure": diastolic,
        "Blood sugar": blood_sugar,
        "CK-MB": ck_mb,
        "Troponin": troponin
    }])

    prediction = model.predict(data)[0]

    st.success(
        "Positive (Heart Attack Risk)"
        if prediction == "positive"
        else "Negative"
    )