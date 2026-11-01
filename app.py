import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="💰",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("insurance_model.pkl")

model = load_model()

st.title("💰 Medical Insurance Cost Predictor")
st.write("Enter the details below to estimate medical insurance charges.")
st.caption("This model provides an estimate and is not an actual insurance quotation.")
st.divider()

with st.form("insurance_form"):
    age = st.slider("Age", 18, 100, 30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoking Status", ["Yes", "No"])
    region = st.selectbox("Region", ["Southeast", "Southwest", "Northeast", "Northwest"])
    submitted = st.form_submit_button("Predict Insurance Cost")

if submitted:
    sex_encoded = 0 if sex == "Male" else 1
    smoker_encoded = 0 if smoker == "Yes" else 1
    region_mapping = {
        "Southeast": 0,
        "Southwest": 1,
        "Northeast": 2,
        "Northwest": 3
    }

    input_data = np.array([[
        age, sex_encoded, bmi, children, smoker_encoded, region_mapping[region]
    ]])

    prediction = model.predict(input_data)[0]
    st.success("Prediction completed!")
    st.metric("Estimated Insurance Cost", f"${prediction:,.2f}")
    st.caption("Made by Priyanshu Singh (2427010573)")
