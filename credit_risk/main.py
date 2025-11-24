import streamlit as st
import pandas as pd
import joblib
import os

# Load model
model = joblib.load('extra_tree_credit_model.pkl')

# Load encoders with proper filenames
encoder_files = {
    "Sex": "Sex_encoder.pkl",
    "Housing": "Housing_encoder.pkl",
    "Saving account": "Saving accounts_encoder.pkl",
    "Checking account": "Checking account_encoder.pkl"
}

encoders = {}
missing_files = []
for col, file in encoder_files.items():
    if os.path.exists(file):
        encoders[col] = joblib.load(file)
    else:
        missing_files.append(file)

if missing_files:
    st.error(f"Missing encoder files: {', '.join(missing_files)}. Please add them.")
    st.stop()

# Streamlit UI
st.title("Credit Risk Prediction App")
st.write("Enter applicant information to predict if the credit risk is Good or Bad")

age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ['male', 'female'])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ['own', 'rent', 'free'])
saving_account = st.selectbox("Saving Account", ['little', 'moderate', 'rich', 'quite rich'])
checking_account = st.selectbox("Checking Account", ['little', 'moderate', 'rich'])
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (month)", min_value=1, value=12)

# Encode categorical inputs
sex_encoded = encoders["Sex"].transform([sex])[0]
housing_encoded = encoders["Housing"].transform([housing])[0]
saving_encoded = encoders["Saving account"].transform([saving_account])[0]
checking_encoded = encoders["Checking account"].transform([checking_account])[0]

# Prepare dataframe for prediction
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [sex_encoded],
    "Job": [job],
    "Housing": [housing_encoded],
    "Saving accounts": [saving_encoded],  # note the plural here
    "Checking account": [checking_encoded],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})


# Prediction on button click
if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.success("The predicted credit risk is: **GOOD**")
    else:
        st.error("The predicted credit risk is: **BAD**")
