import streamlit as st
import numpy as np
import tensorflow as tf

with open('model.pkl', 'rb') as f: 
    model = pickle.load(f)

# Kirish ma'lumotlarini tayyorlash funksiyalari
def data2char_index(X, max_len, is_remove_comment=False):
    alphabet = " abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    result = []
    for data in X:
        mat = []
        if is_remove_comment:
            data = remove_comment(data)
        for ch in data:
            ch = ch.lower()
            if ch not in alphabet:
                continue
            mat.append(alphabet.index(ch))
        result.append(mat)
    return tf.keras.preprocessing.sequence.pad_sequences(np.array(result), padding='post', truncating='post', maxlen=max_len)

def data_to_symbol_tag(X, max_len, is_remove_comment=False):
    symbol = " -,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    result = []
    for data in X:
        mat = []
        if is_remove_comment:
            data = remove_comment(data)
        for ch in data:
            ch = ch.lower()
            if ch not in symbol:
                mat.append(0)
            else:
                mat.append(symbol.index(ch))
        result.append(mat)
    return tf.keras.preprocessing.sequence.pad_sequences(np.array(result), padding='post', truncating='post', maxlen=max_len)

# Streamlit ilovasi
st.title('SQL Injection, XSS va Normal Matnni Tahlil Qilish')

# Matn kiritish
input_text = st.text_area("Iltimos, matn kiriting:")

if input_text:
    # Kiruvchi qiymatlarni tayyorlash
    max_len = 1000
    input_text_encoded = data2char_index([input_text], max_len)
    input_symbol_encoded = data_to_symbol_tag([input_text], max_len)
    
    # Modelga yuborish
    prediction = model.predict([input_text_encoded, input_symbol_encoded])
    
    # Natijani chiqarish
    st.write("Modelning prognozi:")
    st.write(f"SQL Injection: {prediction[0][0]}")
    st.write(f"XSS: {prediction[0][1]}")
    st.write(f"Normal: {prediction[0][2]}")
