import pickle
import streamlit as st
import numpy as np

# Load the model (ensure the .pkl file is in the same directory or provide the path)
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app layout
st.title('XSS Detection Using ML Model')

# Input text box
input_text = st.text_area("Enter input for XSS detection:")

# Prediction when button is clicked
if st.button("Detect XSS"):
    if input_text:
        # Convert input_text to an appropriate format (e.g., a single-element list or numpy array)
        input_data = [input_text]  # This is the correct format (a list of strings)

        # If your model expects a NumPy array, you can convert it:
        # input_data = np.array([input_text])

        prediction = model.predict(input_data)  # Make prediction

        # Show prediction result
        if prediction[0] == 1:
            st.success("XSS Detected!")
        else:
            st.success("No XSS Detected!")
    else:
        st.warning("Please enter some text for detection.")
