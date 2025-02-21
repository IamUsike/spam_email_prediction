# Email Spam Detection System

A machine learning-based system that classifies emails as spam or legitimate (ham) using Natural Language Processing (NLP) and Support Vector Classification (SVC).

## 📊 Dataset Analysis

The dataset contains email messages labeled as spam or ham. Here are key insights from our analysis:

- **Dataset Size**: 5,572 messages (before cleaning)
- **Distribution**:
  - Ham (legitimate): 87.4%
  - Spam: 12.6%

### Email Characteristics

- **Average Length**:

  - Spam Emails: ~138 characters
  - Ham Emails: ~70 characters

- **Word Count**:

  - Spam Emails: ~28 words on average
  - Ham Emails: ~17 words on average

- **Sentence Structure**:
  - Spam Emails: ~3 sentences on average
  - Ham Emails: ~2 sentences on average

### Most Common Words

**In Spam Emails:**

1. call (321 occurrences)
2. free (191 occurrences)
3. txt (141 occurrences)
4. text (122 occurrences)
5. mobile (114 occurrences)

**In Ham Emails:**

1. u (904 occurrences)
2. go (404 occurrences)
3. get (352 occurrences)
4. come (275 occurrences)
5. ok (251 occurrences)

## 🛠 Technical Implementation

### Preprocessing Steps

1. Text lowercase conversion
2. Tokenization using NLTK
3. Special character removal
4. Stop word and punctuation removal
5. Word stemming using Porter Stemmer

### Model Comparison

We tested three different classifiers:

1. **Support Vector Classification (SVC)**

   - Accuracy: 98%
   - Precision: 100%
   - Confusion Matrix:
     ```
     [[889   0]
      [ 25 120]]
     ```

2. **Random Forest Classifier**

   - Accuracy: 98%
   - Precision: 99.19%
   - Confusion Matrix:
     ```
     [[888   1]
      [ 23 122]]
     ```

3. **Naive Bayes Classifier**
   - Accuracy: 97%
   - Precision: 99.17%
   - Confusion Matrix:
     ```
     [[888   1]
      [ 25 120]]
     ```

Based on these results, **SVC** was chosen as the final model due to its highest precision score and better handling of false positives.

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

### Model Files

The repository includes:

- `tfidf_vectorizer.pkl`: Trained TF-IDF vectorizer
- `svm_model.pkl`: Trained SVM classifier model

## 📱 Usage

1. Launch the Streamlit application
2. Enter the email text in the provided text area
3. Click "Check Spam"
4. View the classification result and confidence metrics

## 📊 Features

- Real-time email classification
- Text preprocessing and analysis
- Word count and character statistics
- User-friendly interface
- Detailed spam detection tips

## 🔍 Model Performance

The chosen SVC model demonstrates:

- High precision in spam detection
- Low false positive rate
- Robust performance across various email lengths
- Effective handling of common spam keywords

## 🛠️ Technologies Used

- Python 3.x
- NLTK for NLP tasks
- Scikit-learn for machine learning
- Streamlit for web interface
- Pandas for data manipulation
- Seaborn & Matplotlib for visualization

## 📝 Future Improvements

- Implement real-time model retraining
- Add support for multiple languages
- Enhance feature engineering
- Integrate email attachment analysis
- Add user feedback loop for continuous improvement

## 🤝 Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss the proposed changes.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
