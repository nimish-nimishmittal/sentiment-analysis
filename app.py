import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Streamlit App
st.title("Sentiment Analysis App")

# Step 1: Upload Dataset
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV file with 'text' and 'sentiment' columns", type=['csv'])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())

    # Step 2: Preprocess Data
    st.header("2. Preprocess Data")
    lemmatization = st.checkbox("Apply Lemmatization")
    
    if lemmatization:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        data['text'] = data['text'].apply(lambda x: ' '.join(
            [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(str(x))]))
        st.write("Data after Lemmatization:")
        st.write(data.head())
    
    # Step 3: Vectorization
    st.header("3. Vectorize Text Data")
    vectorizer_choice = st.radio("Choose Vectorization Method", ("TF-IDF", "Word2Vec"))

    if vectorizer_choice == "TF-IDF":
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(data['text'])
    else:
        st.warning("Word2Vec is not implemented in this example.")
        X = None

    y = data['sentiment']
    
    # Step 4: Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Choose Classification Model
    st.header("4. Choose Classification Model")
    model_choice = st.radio("Choose a Model", ("Naive Bayes", "SVM", "Logistic Regression"))

    if model_choice == "Naive Bayes":
        model = MultinomialNB()
    elif model_choice == "SVM":
        model = SVC()
    else:
        model = LogisticRegression()

    # Step 6: Train and Evaluate
    st.header("5. Train and Evaluate")
    if st.button("Train Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {acc}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
