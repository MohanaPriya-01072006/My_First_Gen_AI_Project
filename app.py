import gradio as gr
import numpy as np
import pickle
from tensorflow.keras.models import model_from_json

# Load model
with open("model.pkl", "rb") as f:
    model_json, model_weights = pickle.load(f)

model = model_from_json(model_json)
model.set_weights(model_weights)
model.compile(optimizer="adam", loss="mse")

# Prediction function
def predict_next(n1, n2, n3):
    input_seq = np.array([n1, n2, n3]).reshape((1, 3, 1))
    prediction = model.predict(input_seq, verbose=0)
    return round(float(prediction[0][0]), 2)

# Gradio Interface
iface = gr.Interface(
    fn=predict_next,
    inputs=[
        gr.Number(label="Number 1"),
        gr.Number(label="Number 2"),
        gr.Number(label="Number 3"),
    ],
    outputs=gr.Number(label="Predicted Next Number"),
    title="🔮 LSTM Next Number Predictor",
    description="Enter any 3 numbers in a sequence to predict the next using an LSTM model."
)

iface.launch()
