import os
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize  # Added sent_tokenize
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle


# Set up NLTK data path and downloads
@st.cache_resource
def setup_nltk():
    try:
        # Create and set nltk data directory
        nltk_data_dir = os.path.expanduser("~/nltk_data")
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)

        # Add the directory to NLTK's data path
        nltk.data.path.append(nltk_data_dir)

        # Download required NLTK data
        for package in ["punkt", "stopwords"]:
            try:
                nltk.data.find(f"tokenizers/{package}")
            except LookupError:
                nltk.download(package, download_dir=nltk_data_dir, quiet=True)

        return True
    except Exception as e:
        st.error(f"Error setting up NLTK: {str(e)}")
        return False


def preprocess_text(text):
    try:
        # Convert to lowercase
        text = text.lower()

        # First split into sentences, then words
        sentences = sent_tokenize(text)
        tokens = []
        for sentence in sentences:
            tokens.extend(word_tokenize(sentence))

        # Rest of preprocessing...
        tokens = [re.sub(r"[^a-zA-Z0-9\s]", "", word) for word in tokens]
        stop_words = set(stopwords.words("english"))
        tokens = [
            word
            for word in tokens
            if word not in stop_words
            and word not in string.punctuation
            and word.strip()
        ]

        ps = PorterStemmer()
        tokens = [ps.stem(word) for word in tokens]

        return " ".join(tokens)
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return ""


# Initialize NLTK at startup
if not setup_nltk():
    st.error("Failed to initialize NLTK. Please check your installation.")


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
                if not processed_text:
                    st.error("Error processing text. Please try again.")
                    return

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

                # Display email statistics
                st.write("---")
                st.write("Email Statistics:")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Word Count: {len(email_text.split())}")
                with col2:
                    st.write(f"Character Count: {len(email_text)}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write(
                    "Please make sure all required files and dependencies are properly loaded."
                )
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
