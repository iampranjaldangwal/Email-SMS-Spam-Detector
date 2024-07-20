import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

st.markdown(
    """
    <style>
    .stApp {
        background-color: #e5e5f7;
        background-image: repeating-radial-gradient(circle at 0 0, transparent 0, #e5e5f7 14px), repeating-linear-gradient(#cdcdcd55, #cdcdcd);
        opacity: 1;
    }
    .title-text {
        color: black;
        text-align: center; /* Center align the title */
    }
    .prediction-header {
        text-align: center; /* Center align the prediction result */
        font-size: 24px; /* Adjust font size */
        margin-top: 20px; /* Add some margin to the top */
    }
    .centered-button {
        display: flex;
        justify-content: center;
        margin-top: 20px; /* Adjust margin as needed */
    }
    .message-label {
        color: black;
        text-align: center;
        font-size: 30px;
        margin-bottom: -500px;
        font-family: "Courier New", Courier, monospace;
    }
     .prediction-container {
       text-align: center;
        margin-top: 20px;
    }
    .prediction-text {
        font-size: 24px;
        padding: 5px 10px;
        display: inline-block;
        border-radius: 20px;
        background-color: green;
        color: white;
        margin-top: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Add shadow */
        margin-left: -20px; /* Adjust left margin */
    }
    .spam-text {
        background-color: #8B0000;
    }
    .not-spam-text {
        background-color: #006400; /* Dark green background */
    }
    .text-area-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 10px; /* Adjust margin for text area */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Ensure nltk resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.markdown('<h1 class="title-text">Email/SMS Spam Classifier</h1>', unsafe_allow_html=True)

st.markdown('<div class="message-label">Enter The Message</div>', unsafe_allow_html=True)
input_sms = st.text_area("", height=100)  # Empty label to avoid duplicate "Enter the message" text

# Center align the "Predict" button below the text area
col1, col2, col3 = st.columns([5, 5, 1])
with col2:
     if st.button('Predict'):
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        if 'result' in locals():
         st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
        if result == 1:
            st.markdown('<p class="prediction-text" style="background-color: red; color: white;">Spam</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="prediction-text" style="background-color: green; color: white;">Not Spam</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)