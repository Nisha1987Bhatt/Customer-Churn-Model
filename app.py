import joblib
import streamlit as st
import pandas as pd

model = joblib.load('Churn_model.pkl')
model = joblib.load('scaler.pkl')

st.title("Customer Churn Prediction")
st.write("Enter Customer Details for Predictions")

# Input fields
Age = st.number_input("Age", min_value=18, max_value=100, value=35)
Balance = st.number_input("Balance", min_value=0.0 , value=50000.00)
Is_active = st.selectbox("Is Member Active ?", [0,1])
Is_Zero_Balance = st.selectbox("Is Zero Balance ?", [0,1])
Gender = st.selectbox("Gender", ["Male","Female"]) 

# Predict Button 
if st.button("Predict"):
    input_data = pd.DateFrame({
       'Age' : [Age],
       'Balance':[Balance],
       'IsActiveMember' : [Is_active],
       'IsZeroBalance' : [Is_Zero_Balance],
       'Gender' : [Gender]
  })

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

if prediction == 1:
    st.error(f"Customer can Churn (Proablitiy :{probability:.2%})")
else:
    st.suucess(f"Customer Stay ( Churn Proablitiy :{probability:.2%})") 
     

