import pickle
import streamlit as st

model = pickle.load(open('model.pkl', 'rb'))

st.title('XSS Detection Using ML Model')

input_text = st.text_area("Enter input for XSS detection:")

if st.button("Detect XSS"):
    if input_text:
        prediction = model.predict([input_text])

        if prediction[0] == 1:
            st.success("XSS Detected!")
        else:
            st.success("No XSS Detected!")
    else:
        st.warning("Please enter some text for detection.")
