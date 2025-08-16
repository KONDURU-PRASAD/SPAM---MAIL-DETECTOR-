# SPAM---MAIL-DETECTOR
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load dataset
# Dataset: SMS Spam Collection (UCI)
# Download: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
data = pd.read_csv("spam.csv", encoding='latin-1')[["v1", "v2"]]
data.columns = ["label", "message"]
# 2. Preprocess text
def clean_text(text):
    text = text.lower()                                     # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)     # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    return text

data["cleaned"] = data["message"].apply(clean_text)

# Encode labels: spam=1, ham=0
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# 3. Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X = vectorizer.fit_transform(data["cleaned"])
y = data["label"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
print(" Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
