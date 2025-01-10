import pickle
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Example of how the tokenizer was used during model training
tokenizer = Tokenizer(num_words=5000)

# Streamlit app layout
st.title('XSS Detection Using ML Model')

# Input text box
input_text = st.text_area("Enter input for XSS detection:")
# Let's assume we also need a numerical feature (example: a numeric input)
numeric_feature = st.number_input("Enter a numeric feature:", min_value=0, max_value=100, value=50)

# Prediction when button is clicked
if st.button("Detect XSS"):
    if input_text:
        # Preprocess the input text just like you did during training
        input_data = [input_text]  # Wrap input text in a list
        
        # Tokenizing and padding the input text
        tokenizer.fit_on_texts(input_data)  # Fit on input data (or use the same tokenizer fitted on training data)
        sequences = tokenizer.texts_to_sequences(input_data)
        padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as needed

        # Prepare the second input (numeric feature) as a numpy array
        numeric_feature = np.array([[numeric_feature]])

        # Combine both inputs into a tuple or list
        prediction = model.predict([padded_sequences, numeric_feature])

        # Show prediction result
        if prediction[0] == 1:
            st.success("XSS Detected!")
        else:
            st.success("No XSS Detected!")
    else:
        st.warning("Please enter some text for detection.")
