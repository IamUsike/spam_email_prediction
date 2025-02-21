import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle

import os
import streamlit as st


# Create a function to ensure NLTK data is downloaded
@st.cache_resource
def download_nltk_resources():
    import nltk

    # Specify the download directory explicitly
    nltk_data_dir = os.path.expanduser("~/nltk_data")
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)

    # Set the NLTK data path
    nltk.data.path.append(nltk_data_dir)

    # Download required resources
    for resource in ["punkt", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, download_dir=nltk_data_dir)


# Call the function at the start of your app
download_nltk_resources()


# Download required NLTK data
@st.cache_resource  # Cache the NLTK downloads
def download_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("punkt")
        nltk.download("stopwords")


download_nltk_data()


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove special characters
    tokens = [re.sub(r"[^a-zA-Z0-9\s]", "", word) for word in tokens]

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    tokens = [
        word
        for word in tokens
        if word not in stop_words
        and word not in string.punctuation
        and word  # Add word check
    ]

    # Stemming
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]

    # Join tokens back into text
    return " ".join(tokens)


@st.cache_resource  # Cache the model loading
def load_models():
    try:
        with open("tfidf_vectorizer.pkl", "rb") as f:
            tfidf = pickle.load(f)
        with open("svm_model.pkl", "rb") as f:
            model = pickle.load(f)
        return tfidf, model
    except FileNotFoundError:
        st.error(
            "Model files not found. Please ensure tfidf_vectorizer.pkl and svm_model.pkl exist in the directory."
        )
        return None, None


def main():
    st.title("Email Spam Detector")
    st.write("Enter an email message to check if it's spam or not!")

    # Create a text area for email input
    email_text = st.text_area("Email Content", height=200)

    if st.button("Check Spam"):
        if email_text:
            try:
                # Load models
                tfidf, model = load_models()
                if tfidf is None or model is None:
                    return

                # Preprocess the input text
                processed_text = preprocess_text(email_text)

                # Transform the text using loaded vectorizer
                text_vector = tfidf.transform([processed_text]).toarray()

                # Make prediction
                prediction = model.predict(text_vector)[0]

                # Display result with custom styling
                if prediction == 1:
                    st.error("🚨 This email is likely SPAM!")
                    st.write(
                        "This message contains characteristics commonly found in spam emails."
                    )
                else:
                    st.success("✅ This email appears to be legitimate (HAM)!")
                    st.write("This message appears to be a normal, non-spam email.")

                # Display confidence metrics (if available)
                st.write("---")
                st.write("Email Statistics:")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Word Count: {len(email_text.split())}")
                with col2:
                    st.write(f"Character Count: {len(email_text)}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please make sure the model files are properly loaded.")
        else:
            st.warning("Please enter some email text to analyze.")

    # Add information about the model
    with st.expander("About the Spam Detection Model"):
        st.write(
            """
        This spam detection model uses:
        - Support Vector Machine (SVM) classifier
        - TF-IDF vectorization for text processing
        - NLTK for text preprocessing
        - Trained on a dataset of labeled spam and ham emails
        """
        )

    # Add tips for users
    with st.expander("Tips for Identifying Spam"):
        st.write(
            """
        Common characteristics of spam emails:
        - Promises of free products or money
        - Urgency or pressure to act quickly
        - Requests for personal information
        - Poor grammar or spelling
        - Unexpected attachments
        - Suspicious sender addresses
        """
        )


if __name__ == "__main__":
    main()
