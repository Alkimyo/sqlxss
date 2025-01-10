import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd

# Modelni yuklash
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
st.set_page_config(page_title="SQL Injection, XSS va Normal Matnni Tahlil Qilish", page_icon=":guardsman:", layout="wide")

# Sahifaning fonini o'zgartirish
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f4f8;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
    }
    .css-1v0mbdj {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 12px 20px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Bosh sahifa dizayni
st.title('SQL Injection, XSS detektor')

# Matn kiritish
input_text = st.text_area("Matn kiriting:", height=100)

# Modelni ishga tushirish tugmasi
if st.button('Tahlil qilish'):
    if input_text:
        # Kiruvchi qiymatlarni tayyorlash
        max_len = 1000
        input_text_encoded = data2char_index([input_text], max_len)
        input_symbol_encoded = data_to_symbol_tag([input_text], max_len)

        # Modelga yuborish
        prediction = model.predict([input_text_encoded, input_symbol_encoded])

        # Chiquvchi natijalarni 10 darajasi -2 gacha yaxlitlash
        prediction_rounded = np.round(prediction, decimals=2)

        # Jadval formatida natijalarni chiqarish
        st.write("Modelning prognozi:")
        result_df = pd.DataFrame({
            'Buyruqlar': ['SQL Injection', 'XSS', 'Normal'],
            'Aloqadorligi': prediction_rounded[0]
        })
        st.table(result_df)

        # Qiziqarli vizual effektlar (masalan, ranglar)
        if prediction_rounded[0][0] > 0.9:
            st.markdown('<p style="color:red;">SQL Injection</p>', unsafe_allow_html=True)
        elif prediction_rounded[0][1] > 0.9:
            st.markdown('<p style="color:blue;">XSS</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:green;">Oddiy buyruq</p>', unsafe_allow_html=True)

    else:
        st.warning("Iltimos, matn kiriting va tugmani bosing.")
