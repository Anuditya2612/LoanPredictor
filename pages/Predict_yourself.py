import streamlit as st
import joblib
import numpy as np

# Load model and scaler
scaler = joblib.load("model/scaler.pkl")
model = joblib.load("model/random_forest.pkl")  # or logistic_model.pkl

st.set_page_config(page_title="Predict Loan Outcome", layout="centered")
st.title("🧾 Loan Outcome Prediction Form")

st.markdown("### Fill these customer details below:")

# Collect user input
age = st.number_input("Age", min_value=23, max_value=67, value=30)
experience = st.number_input("Years of Experience", min_value=0, max_value=60, value=5)
income = st.number_input("Annual Income (in k$)", min_value=0, max_value=500, value=50)
family = st.selectbox("Family Size", [1, 2, 3, 4])
ccavg = st.number_input("Credit Card Avg. Spend (in k$)", min_value=0.0, value=1.5)
education = st.selectbox("Education Level", ["Undergraduate", "Graduate", "Advanced/Higher"])
mortgage = st.number_input("Mortgage Amount", min_value=0, value=0)

# Binary fields (Yes/No)
securities_account = st.radio("Has Securities Account?", ["No", "Yes"])
cd_account = st.radio("Has CD (Cert. of deposit) Account?", ["No", "Yes"])
online = st.radio("Accesses Online Banking?", ["No", "Yes"])
creditcard = st.radio("Uses Credit Card?", ["No", "Yes"])

# Mapping categorical inputs
education_map = {"Undergraduate": 1, "Graduate": 2, "Advanced/Professional": 3}
bin_map = {"No": 0, "Yes": 1}

# Prepare feature array
input_data = np.array([[
    age,
    experience,
    income,
    family,
    ccavg,
    education_map[education],
    mortgage,
    bin_map[securities_account],
    bin_map[cd_account],
    bin_map[online],
    bin_map[creditcard]
]])

# Scale the input
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict Loan Status"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.success(f"✅ This customer is likely to ACCEPT the loan. (Probability: {probability:.2%})")
    else:
        st.warning(f"❌ This customer is likely to REJECT the loan offer. (Probability: {probability:.2%})")
