import pickle
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model (ensure the .pkl file is in the same directory or provide the path)
model = pickle.load(open('model.pkl', 'rb'))

# Function to preprocess the input (same as used during model training)
def preprocess_input(input_text, max_len=1000):
    alphabet = " abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    result = []
    for data in input_text:
        mat = []
        data = data.lower()  # Convert to lowercase
        for ch in data:
            if ch in alphabet:
                mat.append(alphabet.index(ch))
        # Pad sequences to match the model input shape
        result.append(mat)
    
    padded_input = pad_sequences(result, padding='post', truncating='post', maxlen=max_len)
    return padded_input

# Streamlit app layout
st.title('XSS Detection Using ML Model')

# Input text box
input_text = st.text_area("Enter input for XSS detection:")

# Prediction when button is clicked
if st.button("Detect XSS"):
    if input_text:
        # Preprocess input before prediction
        preprocessed_input = preprocess_input([input_text])

        # Make the prediction
        prediction = model.predict([preprocessed_input, preprocessed_input])  # Assuming 2 inputs for your model

        # Show prediction result
        if np.argmax(prediction) == 1:
            st.success("XSS Detected!")
        else:
            st.success("No XSS Detected!")
    else:
        st.warning("Please enter some text for detection.")
