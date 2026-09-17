import streamlit as st
import pandas as pd
import joblib


model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
features = joblib.load('features.pkl')

st.title("Mobile Price Range Prediction")


columns = st.columns(len(features))


battery_power = columns[0].number_input("battery_power", min_value=500.0, max_value=6000.0, value=1500.0, step=10.0)
ram = columns[1].number_input("ram", min_value=250.0, max_value=4000.0, value=1000.0, step=10.0)
px_height = columns[2].number_input("px_height", min_value=0.0, max_value=2000.0, value=500.0, step=1.0)
px_width = columns[3].number_input("px_width", min_value=0.0, max_value=2000.0, value=1000.0, step=1.0)
mobile_wt = columns[4].number_input("mobile_wt", min_value=80.0, max_value=250.0, value=150.0, step=1.0)


df = pd.DataFrame([[battery_power, ram, px_height, px_width, mobile_wt]], columns=features)

if st.button("predict"):
    
    for name in encoder.keys():
        if name in df.columns:
            df[name] = encoder[name].transform(df[name])
        
    prediction = model.predict(df)
    st.write("prediction: ", prediction[0])