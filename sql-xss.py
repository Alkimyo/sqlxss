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

# Sahifaning fonini va tugma ranglarini sozlash
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;  /* Sahifaning foni */
    }
    .stTextInput>div>div>textarea {
        font-size: 16px;
        background-color: #ffffff;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #007BFF;  /* Tugma rangi */
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0056b3;  /* Tugma hover effekti */
    }
    </style>
    """, unsafe_allow_html=True)

# Bosh sahifa dizayni
st.title('SQL Injection va XSS Detektori')
st.subheader('Matnni tahlil qilish va xavfli buyruqlarni aniqlash uchun tizim')

# Matn kiritish
input_text = st.text_area("Matnni kiriting:", height=150, placeholder="SQL yoki JavaScript kodini kiriting...")

# Modelni ishga tushirish tugmasi
if st.button('Tahlil qilish'):
    if input_text.strip():
        # Kiruvchi qiymatlarni tayyorlash
        max_len = 1000
        input_text_encoded = data2char_index([input_text], max_len)
        input_symbol_encoded = data_to_symbol_tag([input_text], max_len)

        # Modelga yuborish
        prediction = model.predict([input_text_encoded, input_symbol_encoded])

        # Chiquvchi natijalarni yaxlitlash
        prediction_rounded = np.round(prediction, decimals=2)

        # Natijalarni jadval formatida ko'rsatish
        st.write("Tahlil natijalari:")
        result_df = pd.DataFrame({
            'Buyruqlar': ['SQL Injection', 'XSS', 'Oddiy Matn'],
            'Aloqadorligi': prediction_rounded[0]
        })
        st.table(result_df)

        # Natija bo'yicha xabar chiqarish
        if prediction_rounded[0][0] > 0.9:
            st.markdown('<p style="color:red; font-size:20px;">Ehtimol: SQL Injection</p>', unsafe_allow_html=True)
        elif prediction_rounded[0][1] > 0.9:
            st.markdown('<p style="color:blue; font-size:20px;">Ehtimol: XSS</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:green; font-size:20px;">Oddiy matn</p>', unsafe_allow_html=True)
    else:
        st.warning("Iltimos, matnni kiriting va qayta urinib ko'ring.")
