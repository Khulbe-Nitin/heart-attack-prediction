import streamlit as st
import joblib
import pandas as pd

model = joblib.load("heart_attack_rf_model.pkl")

st.title("Heart Attack Prediction")

age = st.number_input("Age", 1, 120)
heart_rate = st.number_input("Heart Rate", 30, 250)
systolic = st.number_input("Systolic Blood Pressure", 50, 250)
diastolic = st.number_input("Diastolic Blood Pressure", 30, 150)
blood_sugar = st.number_input("Blood Sugar", 0, 500)
ck_mb = st.number_input("CK-MB")
troponin = st.number_input("Troponin")

if st.button("Predict"):
    data = pd.DataFrame([{
        "Age": age,
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