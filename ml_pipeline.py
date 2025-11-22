import pandas as pd
import re
import nltk
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


vectorizer = None
model = None

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()



def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(cleaned)



def train_model():
    global vectorizer, model

    try:
        df = pd.read_csv("youtoxic_english_1000.csv")
    except FileNotFoundError:
        print("Ошибка: файл youtoxic_english_1000.csv не найден!")
        return

    df["IsToxic"] = df["IsToxic"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0})

    
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["IsToxic"]
    )

    
    train_df["clean"] = train_df["Text"].astype(str).apply(preprocess_text)
    test_df["clean"] = test_df["Text"].astype(str).apply(preprocess_text)

    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df["clean"])
    X_test = vectorizer.transform(test_df["clean"])

    y_train = train_df["IsToxic"]
    y_test = test_df["IsToxic"]

    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"\nМодель обучена! Точность: {acc:.4f}")



def predict_comment(text):
    global model, vectorizer

    if model is None or vectorizer is None:
        print("Сначала обучите модель (пункт 1).")
        return "Нет модели", 0.0

    clean = preprocess_text(text)
    X = vectorizer.transform([clean])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][pred]

    label = "Токсичный" if pred == 1 else "Не токсичный"
    return label, prob
