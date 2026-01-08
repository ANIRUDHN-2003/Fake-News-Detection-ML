import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("Loading LIAR dataset...")

df = pd.read_csv("liar_dataset/train.tsv", sep="\t", header=None)

# LIAR dataset column mapping
df.columns = [
    "id", "label", "statement", "subject", "speaker",
    "job", "state", "party", "barely_true",
    "false", "half_true", "mostly_true", "pants_on_fire", "context"
]

# Binary classification
df["label"] = df["label"].replace({
    "true": 1,
    "mostly-true": 1,
    "half-true": 1,
    "barely-true": 0,
    "false": 0,
    "pants-fire": 0
})

df = df[["statement", "label"]]
df.dropna(inplace=True)

X = df["statement"]
y = df["label"]

print("Vectorizing text...")
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vec = vectorizer.fit_transform(X)

print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

print("Saving model files...")
pickle.dump(model, open("final_model.sav", "wb"))
pickle.dump(vectorizer, open("tfidf.pkl", "wb"))

print("MODEL CREATED SUCCESSFULLY")
