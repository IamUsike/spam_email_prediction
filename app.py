import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")


# Function to preprocess text
def preprocess_text(text):
    """Preprocess the input text: lowercase, tokenize, remove special chars, stopwords, and stem."""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [re.sub(r"[^a-zA-Z0-9\s]", "", word) for word in tokens]
    stop_words = set(stopwords.words("english"))
    tokens = [
        word
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)


# Function to train and save model or load existing model
@st.cache_resource
def load_or_train_model():
    try:
        # Attempt to load pre-trained model and vectorizer
        svc_classifier = joblib.load("svc_model.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        st.info("Loaded pre-trained model and vectorizer.")
    except FileNotFoundError:
        # If files don't exist, train the model
        st.info("Training model from scratch...")

        # Load and preprocess data
        data = pd.read_csv("spam.csv", encoding="latin1")
        data = data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
        data.rename(columns={"v1": "result", "v2": "emails"}, inplace=True)
        data = data.drop_duplicates(keep="first")

        # Preprocess emails
        data["transform_text"] = data["emails"].apply(preprocess_text)

        # Vectorize and encode labels
        tfidf = TfidfVectorizer(max_features=3000)
        X = tfidf.fit_transform(data["transform_text"]).toarray()
        y = data["result"].map({"ham": 0, "spam": 1})  # 0 for ham, 1 for spam

        # Train SVC model
        svc_classifier = SVC()
        svc_classifier.fit(X, y)

        # Save the model and vectorizer
        joblib.dump(svc_classifier, "svc_model.pkl")
        joblib.dump(tfidf, "tfidf_vectorizer.pkl")
        st.success("Model trained and saved successfully!")

    return svc_classifier, tfidf


# Main Streamlit app
def main():
    st.title("Spam Email Classifier")
    st.markdown(
        """
    This app classifies emails as **Spam** or **Ham** using a Support Vector Classifier (SVC).
    Enter an email text below to see the prediction!
    """
    )

    # Load or train model and vectorizer
    svc_classifier, tfidf = load_or_train_model()

    # Sidebar for additional info
    st.sidebar.header("About")
    st.sidebar.write(
        """
    - Built with Streamlit and Scikit-learn.
    - Uses a pre-trained SVC model trained on the `spam.csv` dataset.
    - Features: TF-IDF vectorization, text preprocessing (stemming, stopword removal).
    """
    )

    # User input section
    st.subheader("Classify Your Email")
    email_input = st.text_area(
        "Enter your email text here:",
        height=150,
        placeholder="e.g., 'Get a free iPhone now!'",
    )

    if st.button("Classify"):
        if email_input:
            # Preprocess and predict
            processed_text = preprocess_text(email_input)
            X_new = tfidf.transform([processed_text]).toarray()
            prediction = svc_classifier.predict(X_new)[0]

            # Display result
            if prediction == 1:
                st.error(f"'{email_input}' is predicted as **Spam**.")
            else:
                st.success(f"'{email_input}' is predicted as **Ham**.")
        else:
            st.warning("Please enter some text to classify.")

    # Example predictions section
    st.subheader("Example Predictions")
    examples = [
        "Get a free iPhone now!",
        "Hey, how's it going?",
        "Congratulations! You've won a prize!",
        "Reminder: Meeting at 2 PM tomorrow.",
    ]
    for example in examples:
        processed_example = preprocess_text(example)
        X_example = tfidf.transform([processed_example]).toarray()
        pred = svc_classifier.predict(X_example)[0]
        label = "Spam" if pred == 1 else "Ham"
        st.write(f"'{example}' → **{label}**")

    # Optional: Display model performance (if trained)
    st.subheader("Model Performance")
    if st.checkbox("Show Model Metrics"):
        data = pd.read_csv("spam.csv", encoding="latin1")
        data = data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
        data.rename(columns={"v1": "result", "v2": "emails"}, inplace=True)
        data = data.drop_duplicates(keep="first")
        data["transform_text"] = data["emails"].apply(preprocess_text)

        X = tfidf.transform(data["transform_text"]).toarray()
        y = data["result"].map({"ham": 0, "spam": 1})

        y_pred = svc_classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)

        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.write(f"**Precision:** {precision:.2f}")
        st.write("**Confusion Matrix:**")
        st.write(conf_matrix)

        # Plot confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)


if __name__ == "__main__":
    main()
