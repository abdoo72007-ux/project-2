import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
features = joblib.load('features.pkl')

st.title("Mobile Price Range Prediction")

# تقسيم الشاشة بعدد الـ features بالضبط
columns = st.columns(len(features))

# إنشاء dictionary لتخزين المدخلات لكل فِيتشر بشكل ديناميكي صحيح
input_data = {}

for i, feature in enumerate(features):
    # لو العمود رقمي، بنعمله number_input
    input_data[feature] = columns[i].number_input(f"{feature}", value=0.0)

# تحويل القاموس إلى DataFrame بالاعتماد على الـ features كأعمدة
df = pd.DataFrame([input_data])

if st.button("predict"):
    # تطبيق الـ encoder على الأعمدة النصية إن وجدت
    for name in encoder.keys():
        if name in df.columns:
            df[name] = encoder[name].transform(df[name])
        
    prediction = model.predict(df)
    st.write("prediction: ", prediction[0])