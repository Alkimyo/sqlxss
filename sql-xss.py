import pickle
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model (ensure the .pkl file is in the same directory or provide the path)
model = pickle.load(open('model.pkl', 'rb'))

# Example of how the tokenizer was used during model training
# This should match the tokenizer used to train your model.
tokenizer = Tokenizer(num_words=5000)

# Streamlit app layout
st.title('XSS Detection Using ML Model')

# Input text box
input_text = st.text_area("Enter input for XSS detection:")

# Prediction when button is clicked
if st.button("Detect XSS"):
    if input_text:
        # Preprocess the input text just like you did during training
        input_data = [input_text]  # Wrap input text in a list
        
        # Tokenizing and padding the input text
        tokenizer.fit_on_texts(input_data)  # Fit on input data (or use the same tokenizer fitted on training data)
        sequences = tokenizer.texts_to_sequences(input_data)
        padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as needed

        # Make prediction
        prediction = model.predict(padded_sequences)

        # Show prediction result
        if prediction[0] == 1:
            st.success("XSS Detected!")
        else:
            st.success("No XSS Detected!")
    else:
        st.warning("Please enter some text for detection.")
