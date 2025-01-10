import pickle
import streamlit as st
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# Load the model (ensure the .pkl file is in the same directory or provide the path)
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app layout
st.title('XSS Detection Using ML Model')

# Input text box
input_text = st.text_area("Enter input for XSS detection:")

# Example of numeric feature (this should be based on your model)
numeric_feature = 1  # Replace with actual feature, if applicable

# Function to preprocess the input (same as used during model training)
def preprocess_input(input_text, max_len=1000):
    alphabet = " abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    result = []
    for data in input_text:
        mat = []
        data = data.lower()  # Convert to lowercase
        for ch in data:
            if ch in alphabet:
                mat.append(alphabet.index(ch))  # Ensure index is within range of alphabet length
            else:
                mat.append(0)  # If character is not in alphabet, use index 0 (for unknown characters)
        # Pad sequences to match the model input shape
        result.append(mat)
    
    padded_input = pad_sequences(result, padding='post', truncating='post', maxlen=max_len)
    return np.array(padded_input)

# Prediction when button is clicked
if st.button("Detect XSS"):
    if input_text:
        # Preprocess the input before prediction
        preprocessed_input = preprocess_input([input_text])

        # Prepare the inputs for prediction (model expects 2 inputs)
        # Convert the numeric feature to numpy array and ensure inputs are correctly formatted
        numeric_feature = np.array([numeric_feature])
        inputs = [preprocessed_input, numeric_feature]

        # Make prediction
        try:
            prediction = model.predict(inputs)

            # Show prediction result
            if prediction[0] == 1:
                st.success("XSS Detected!")
            else:
                st.success("No XSS Detected!")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    else:
        st.warning("Please enter some text for detection.")
