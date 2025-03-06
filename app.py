import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)


# Load pre-trained vectorizer and model
tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# App title and header
st.set_page_config(page_title="SMS Spam Detection", page_icon="📩", layout="centered")

# Add a custom sidebar
st.sidebar.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.title("About the App")
st.sidebar.info(
    """
    This application is a **machine learning-powered SMS Spam Detection** tool. 
    It preprocesses text, applies vectorization, and uses a trained model to predict whether an SMS is spam or not.
    """
)
st.sidebar.write("### Instructions:")
st.sidebar.write("- Enter your SMS text in the input box.")
st.sidebar.write("- Press **Predict** to check if it's spam.")

# Main app design
st.markdown(
    """
    <style>
    h1 {
        font-family: 'Arial', sans-serif;
        font-size: 2.5rem;
        color: #333;
    }
    .stTextInput input {
        border: 2px solid #ccc;
        border-radius: 8px;
        padding: 10px;
        width: 100%;
    }
    .stButton button {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 1rem;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add a header
st.title("📩 SMS Spam Detection")
st.write("#### A Machine Learning App for Spam Classification")
st.write("*Made by Keerthana Jella*")

# Input text box
st.markdown("### Enter your SMS below:")
input_sms = st.text_input("", placeholder="Type your SMS here...")

# Predict button
if st.button("🔍 Predict"):

    # Preprocess the input SMS
    transformed_sms = transform_text(input_sms)
    
    # Vectorize the input
    vector_input = tk.transform([transformed_sms])
    
    # Predict
    result = model.predict(vector_input)[0]
    
    # Display the result with styling
    if result == 1:
        st.error("🚨 **Spam**! This SMS is classified as spam.")
    else:
        st.success("✅ **Not Spam**! This SMS is not classified as spam.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #aaa; font-size: 0.9rem;">
        © 2025 Keerthana Jella . All Rights Reserved.
    </div>
    """,
    unsafe_allow_html=True,
)
