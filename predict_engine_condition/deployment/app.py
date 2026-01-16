import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Sandhya777/engine_condition_prediction_model", filename="best_engine_condition_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Engine Condition Prediction App")
st.write("""
This application predicts the likelihood of a engine failing based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
Engine_rpm = st.number_input(
    "Engine rpm",
    min_value=0.0,
    max_value=2500.0,
    value=750.0,
    step=10.0
)
Lub_oil_pressure = st.number_input(
    "Lub oil pressure",
    min_value=0.0,
    max_value=8.0,
    value=3.2,
    step=0.1
)
Fuel_pressure = st.number_input(
    "Fuel pressure",
    min_value=0.0,
    max_value=22.0,
    value=6.2,
    step=0.1
)
Coolant_pressure = st.number_input(
    "Coolant pressure",
    min_value=0.0,
    max_value=8.0,
    value=2.2,
    step=0.1
)
lub_oil_temp = st.number_input(
    "Lub oil temperature",
    min_value=60.0,
    max_value=100.0,
    value=77.0,
    step=0.5
)
Coolant_temp = st.number_input(
    "Coolant temperature",
    min_value=60.0,
    max_value=200.0,
    value=78.0,
    step=0.5
)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Engine_rpm': Engine_rpm,
    'Lub_oil_pressure': Lub_oil_pressure,
    'Fuel_pressure': Fuel_pressure,
    'Coolant_pressure': Coolant_pressure,
    'lub_oil_temp': lub_oil_temp,
    'Coolant_temp': Coolant_temp
}])


if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Engine Failure" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
