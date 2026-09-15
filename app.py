import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
features = joblib.load('features.pkl')

st.title("Mobile Price Range Prediction")


columns = st.columns(len(features))


battery_power = columns[0].number_input("Battery Power", min_value=500, max_value=6000, value=1500)
ram = columns[1].number_input("RAM", min_value=250, max_value=8000, value=2000)
mobile_wt = columns[2].number_input("Mobile Weight", min_value=80, max_value=200, value=150)


df = pd.DataFrame([[battery_power, ram, mobile_wt]], columns=features)

if st.button("predict"):
    
    for name in encoder.keys():
        df[name] = encoder[name].transform(df[name])
        
    prediction = model.predict(df)
    st.write("prediction: ", prediction[0])