import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import model_from_json

# Set Streamlit app title and description
st.title("🔮 LSTM Next Number Predictor")
st.write("Enter any 3 numbers in a sequence to predict the next using an LSTM model.")

# Load model
with open("model.pkl", "rb") as f:
    model_json, model_weights = pickle.load(f)

model = model_from_json(model_json)
model.set_weights(model_weights)
model.compile(optimizer="adam", loss="mse")

# User input
n1 = st.number_input("Enter Number 1", value=0.0)
n2 = st.number_input("Enter Number 2", value=0.0)
n3 = st.number_input("Enter Number 3", value=0.0)

# Predict function
def predict_next(n1, n2, n3):
    input_seq = np.array([n1, n2, n3]).reshape((1, 3, 1))
    prediction = model.predict(input_seq, verbose=0)
    return round(float(prediction[0][0]), 2)

# Predict button
if st.button("Predict Next Number"):
    result = predict_next(n1, n2, n3)
    st.success(f"Predicted Next Number: {result}")
