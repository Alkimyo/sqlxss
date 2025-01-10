import pickle
import streamlit as st

# Load the model (ensure the .pkl file is in the same directory or provide the path)
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app layout
st.title('XSS Detection Using ML Model')

# Input text box
input_text = st.text_area("Enter input for XSS detection:")

# Prediction when button is clicked
if st.button("Detect XSS"):
    if input_text:
        # Preprocess input if necessary before prediction (you can add custom preprocessing)
        prediction = model.predict([input_text])

        # Show prediction result
        if prediction[0] == 1:
            st.success("XSS Detected!")
        else:
            st.success("No XSS Detected!")
    else:
        st.warning("Please enter some text for detection.")
