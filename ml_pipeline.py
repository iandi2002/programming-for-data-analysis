import os
import re
import pickle
from typing import List, Tuple

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

# Download NLTK resources (safe to call multiple times)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize

# ================== CONSTANTS ==================

# Main English dataset
DEFAULT_DATASET = "youtoxic_english_1000.csv"

# Folder for additional CSV datasets (e.g., Russian Kaggle datasets)
DATA_DIR = "data"

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# Multi-label toxicity columns
label_columns = [
    "IsToxic",
    "IsAbusive",
    "IsThreat",
    "IsProvocative",
    "IsObscene",
    "IsHatespeech",
    "IsRacist",
    "IsNationalist",
    "IsSexist",
    "IsHomophobic",
    "IsReligiousHate",
    "IsRadicalism",
]

# Human-readable names for UI / reports
label_map = {
    "IsToxic": "General toxicity",
    "IsAbusive": "Abusive / insulting language",
    "IsThreat": "Threats",
    "IsProvocative": "Provocative language",
    "IsObscene": "Obscene language",
    "IsHatespeech": "Hate speech",
    "IsRacist": "Racism",
    "IsNationalist": "Nationalism",
    "IsSexist": "Sexism",
    "IsHomophobic": "Homophobia",
    "IsReligiousHate": "Religious hate",
    "IsRadicalism": "Radicalism",
}

# Global model objects (loaded once and reused)
model: OneVsRestClassifier | None = None
vectorizer: TfidfVectorizer | None = None

# English + Russian stopwords
stop_words = set(stopwords.words("english")) | set(stopwords.words("russian"))
lemmatizer = WordNetLemmatizer()


# ================== TEXT PREPROCESSING ==================

def preprocess_text(text: str) -> str:
    """
    Multilingual text preprocessing (English + Russian):
    - lowercasing
    - keep only English and Russian letters and spaces
    - tokenization
    - stopword removal
    - lemmatization for English tokens (Russian tokens remain as is)
    """
    text = str(text).lower()
    # Keep only latin + cyrillic letters and spaces
    text = re.sub(r"[^a-zа-яё\s]", " ", text)

    tokens = wordpunct_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    lemmas: List[str] = []
    for t in tokens:
        # Lemmatize only English tokens, leave Russian tokens unchanged
        if re.fullmatch(r"[a-z]+", t):
            lemmas.append(lemmatizer.lemmatize(t))
        else:
            lemmas.append(t)

    return " ".join(lemmas)


# ================== DATA LOADING (MULTI-DATASET, KAGGLE SUPPORT) ==================

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names so that different datasets can be merged:
    - if 'comment' exists and 'Text' does not, rename 'comment' -> 'Text'
    - if 'toxic' exists and 'IsToxic' does not, rename 'toxic' -> 'IsToxic'
    - ensure all label_columns exist (missing are filled with zeros)
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # Kaggle-style: 'comment', 'toxic'
    if "comment" in cols_lower and "Text" not in df.columns:
        df.rename(columns={cols_lower["comment"]: "Text"}, inplace=True)

    if "toxic" in cols_lower and "IsToxic" not in df.columns:
        df.rename(columns={cols_lower["toxic"]: "IsToxic"}, inplace=True)

    # Ensure all label columns exist
    for col in label_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only Text + label_columns to have a consistent schema
    return df[["Text"] + label_columns]


def _load_all_datasets() -> pd.DataFrame:
    """
    Load and merge all available datasets:
    - DEFAULT_DATASET (if exists)
    - all CSV files from DATA_DIR/ (if exists)

    Each dataset must have a 'Text' column after normalization.
    Supports Kaggle format with 'comment' + 'toxic'.
    """
    paths: List[str] = []

    if os.path.exists(DEFAULT_DATASET):
        paths.append(DEFAULT_DATASET)

    if os.path.isdir(DATA_DIR):
        for fname in os.listdir(DATA_DIR):
            if fname.lower().endswith(".csv"):
                paths.append(os.path.join(DATA_DIR, fname))

    if not paths:
        raise FileNotFoundError(
            f"No datasets found. Place '{DEFAULT_DATASET}' in the project root "
            f"or add CSV files to the '{DATA_DIR}/' folder."
        )

    dataframes: List[pd.DataFrame] = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f" Skipping {path}: failed to read CSV ({e})")
            continue

        try:
            df = _normalize_columns(df)
        except KeyError:
            print(f" Skipping {path}: cannot normalize columns to Text + labels.")
            continue

        # Drop duplicates and empty texts
        df = df.dropna(subset=["Text"])
        df = df.drop_duplicates(subset=["Text"])

        print(f" Loaded dataset: {path} (rows: {len(df)})")
        dataframes.append(df)

    if not dataframes:
        raise ValueError("No valid datasets with text and labels were found.")

    merged = pd.concat(dataframes, ignore_index=True)
    print(f"\ Total merged dataset size: {len(merged)} rows")
    return merged


# ================== ARTIFACTS (SAVE / LOAD) ==================

def _save_artifacts():
    """Save trained model and TF-IDF vectorizer to disk."""
    global model, vectorizer

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")


def load_model():
    """
    Try to load a previously trained model and vectorizer.
    Called from main.py, api.py and streamlit_app.py on startup.
    """
    global model, vectorizer

    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        print(" Loaded model and vectorizer from disk.")
    else:
        print(" Trained model not found. Please train it first (menu option 1).")


def _ensure_model_loaded():
    """Lazy-load model if it is not already in memory."""
    global model, vectorizer
    if model is None or vectorizer is None:
        load_model()


# ================== VISUALIZATION HELPERS ==================

def _plot_label_f1_scores(report_dict, save_path: str = "label_f1_scores.png"):
    """
    Plot F1-scores for each toxicity label and save to file.
    """
    labels = []
    f1_scores = []

    for label in label_columns:
        if label in report_dict:
            labels.append(label_map.get(label, label))
            f1_scores.append(report_dict[label]["f1-score"])

    if not labels:
        print(" No labels found in classification report for plotting.")
        return

    plt.figure(figsize=(10, 5))
    plt.bar(labels, f1_scores)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.xlabel("Toxicity types")
    plt.ylabel("F1-score")
    plt.title("Model performance by toxicity type (F1-score)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" Saved F1-score visualization to {save_path}")


def _plot_is_toxic_confusion_matrix(y_true, y_pred, save_path: str = "confusion_is_toxic.png"):
    """
    Plot a confusion matrix only for the 'IsToxic' label.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Not toxic", "Toxic"]

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix: IsToxic")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" Saved confusion matrix to {save_path}")


def _plot_prediction_probabilities(probs, save_path: str = "last_prediction_probs.png"):
    """
    Plot probabilities for each toxicity label for a single comment.
    """
    labels = [label_map.get(lbl, lbl) for lbl in label_columns]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, probs)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.xlabel("Toxicity types")
    plt.ylabel("Predicted probability")
    plt.title("Predicted probabilities for each toxicity type")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" Saved prediction probability visualization to {save_path}")


def _compute_top_toxic_words(
    X_train_vec,
    y_train: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    top_n: int = 50,
    save_path: str = "top_toxic_words.txt",
):
    """
    Compute top toxic words based on average TF-IDF scores
    for rows where IsToxic == 1.
    """
    feature_names = vectorizer.get_feature_names_out()
    toxic_mask = y_train["IsToxic"].values == 1

    if toxic_mask.sum() == 0:
        print(" No toxic samples found in training data for top-word analysis.")
        return

    X_toxic = X_train_vec[toxic_mask]
    mean_scores = np.asarray(X_toxic.mean(axis=0)).ravel()

    top_indices = np.argsort(mean_scores)[::-1][:top_n]
    top_words = [(feature_names[i], float(mean_scores[i])) for i in top_indices]

    with open(save_path, "w", encoding="utf-8") as f:
        for word, score in top_words:
            f.write(f"{word}\t{score:.6f}\n")

    print(f" Saved top {top_n} toxic words to {save_path}")


def _generate_toxic_wordcloud(
    df: pd.DataFrame,
    save_path: str = "wordcloud_toxic.png",
):
    """
    Generate a wordcloud image based on all toxic comments (IsToxic == 1).
    Uses preprocessed text.
    """
    toxic_df = df[df["IsToxic"] == 1]
    if toxic_df.empty:
        print(" No toxic samples found for wordcloud.")
        return

    texts = [preprocess_text(t) for t in toxic_df["Text"].astype(str)]
    full_text = " ".join(texts)

    if not full_text.strip():
        print(" No text available for wordcloud after preprocessing.")
        return

    wc = WordCloud(
        width=1200,
        height=800,
        background_color="white",
    ).generate(full_text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved toxic wordcloud to {save_path}")


# ================== TRAINING ==================

def train_model():
    """
    Train a multi-label toxicity detection model (EN + RU, multi-dataset):
    - input: comment text
    - output: 12 toxicity types (0/1 for each)
    Also:
    - generates F1-score bar chart for each label
    - generates confusion matrix for 'IsToxic'
    - generates top toxic words file
    - generates toxic wordcloud image
    """
    global model, vectorizer

    try:
        df = _load_all_datasets()
    except Exception as e:
        print(f" Failed to load datasets: {e}")
        return

    # Safety: drop remaining NaN texts
    df = df.dropna(subset=["Text"])

    X_raw = df["Text"].astype(str)
    y = df[label_columns].astype(int)

    # Stratify by "IsToxic" to keep similar distribution in train/test
    stratify_col = y["IsToxic"]

    print(" Splitting into train/test...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.3,
        random_state=42,
        stratify=stratify_col,
    )

    print("Preprocessing text (this may take a bit)...")
    X_train = [preprocess_text(t) for t in X_train_raw]
    X_test = [preprocess_text(t) for t in X_test_raw]

    print(" TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=20000,      # more features for richer multilingual data
        ngram_range=(1, 2),      # unigrams + bigrams
        min_df=2                 # ignore ultra-rare tokens
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(" Training multi-label model (One-vs-Rest Logistic Regression)...")
    base_clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",  # helps with label imbalance
        C=2.0                     # slightly weaker regularization than default
    )
    model = OneVsRestClassifier(base_clf)
    model.fit(X_train_vec, y_train)

    print("\n Evaluation on test set:")
    y_pred = model.predict(X_test_vec)

    # Console report
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_columns,
            zero_division=0,
        )
    )

    # Dict report for visualization
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=label_columns,
        zero_division=0,
        output_dict=True,
    )
    _plot_label_f1_scores(report_dict)

    # Confusion matrix for IsToxic only
    is_toxic_index = label_columns.index("IsToxic")
    _plot_is_toxic_confusion_matrix(
        y_test["IsToxic"].values,
        y_pred[:, is_toxic_index],
    )

    # Top toxic words + wordcloud
    _compute_top_toxic_words(X_train_vec, y_train, vectorizer)
    _generate_toxic_wordcloud(df)

    _save_artifacts()
    print("\nTraining finished.")


# ================== PREDICTION ==================

def predict_comment(text: str, visualize: bool = True) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Predict toxicity of a single comment.

    Returns:
    - overall_label: "Toxic" / "Not toxic"
    - overall_prob: overall confidence (0..1)
    - toxic_types: list of (label_code, probability) for labels predicted as 1
    """
    global model, vectorizer

    _ensure_model_loaded()

    if model is None or vectorizer is None:
        print("Model is not available. Please train it first (menu option 1).")
        return "No model", 0.0, []

    clean = preprocess_text(text)
    X_vec = vectorizer.transform([clean])

    # Binary predictions (0/1 for each label) and probabilities of class 1
    preds = model.predict(X_vec)[0]
    probs = model.predict_proba(X_vec)[0]

    toxic_types: List[Tuple[str, float]] = []
    for col, y_hat, p in zip(label_columns, preds, probs):
        if y_hat == 1:
            toxic_types.append((col, float(p)))

    # Overall decision: if any label is positive, we call it toxic
    is_toxic_overall = any(preds)
    if is_toxic_overall:
        overall_label = "Toxic"
        overall_prob = float(max(probs))  # use max toxicity probability
    else:
        overall_label = "Not toxic"
        overall_prob = float(1.0 - max(probs))  # confidence in non-toxicity

    if visualize:
        _plot_prediction_probabilities(probs)

    return overall_label, overall_prob, toxic_types
