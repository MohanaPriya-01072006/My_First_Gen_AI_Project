# 🔮 LSTM Next Number Predictor – Gen AI Mini Project

This is my first Generative AI project – a **sequence prediction app** built using an **LSTM model** in TensorFlow and deployed on **Hugging Face Spaces** using **Gradio**.

## 🌟 What It Does

This app predicts the **next number** in a sequence of 3 input numbers using an LSTM neural network trained on data from 1 to 100.

## 🧠 Model Info

- Framework: TensorFlow (Keras)
- Architecture: LSTM with 50 units and 1 Dense output layer
- Training Data: Sequence [1, 2, 3, ..., 100]
- Window Size: 3

## 📦 Files Included

- `app.py`: Gradio interface to run the model
- `model.pkl`: Pickled LSTM model (architecture + weights)
- `requirements.txt`: Required libraries
- `README.md`: Project overview

## 🚀 Demo (Live on Hugging Face)

👉 [Check out the live demo](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME)  
*(Replace with your actual Hugging Face Space link)*

## 🛠 How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/LSTM-Next-Number-Predictor.git
cd LSTM-Next-Number-Predictor
pip install -r requirements.txt
python app.py
