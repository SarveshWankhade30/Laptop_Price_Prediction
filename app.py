import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("💻 Laptop Price Prediction")

# User Input
brand = st.selectbox("Select Brand", ['Dell', 'HP', 'Asus', 'Lenovo', 'Acer'])
processor = st.selectbox("Select Processor", ['i3', 'i5', 'i7'])
ram = st.slider("RAM (GB)", 2, 64, 8, step=2)
storage = st.slider("Storage (GB)", 128, 2048, 512, step=128)

input_df = pd.DataFrame({
    "Brand": [brand],
    "Processor": [processor],
    "RAM": [ram],
    "Storage": [storage]
})

if st.button("Predict Price"):
    price = model.predict(input_df)[0]
    st.success(f"💰 Estimated Laptop Price: ₹ {int(price)}")
