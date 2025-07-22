import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")
fake['label'] = 0
true['label'] = 1
data = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)

X = data['text']
y = data['label']

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_acc = accuracy_score(y_test, nb_model.predict(X_test))

# Save models
os.makedirs("model", exist_ok=True)
pickle.dump(lr_model, open("model/lr_model.pkl", "wb"))
pickle.dump(nb_model, open("model/nb_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

# Save accuracy info
with open("model/accuracy.txt", "w") as f:
    f.write(f"Logistic Regression Accuracy: {lr_acc:.4f}\n")
    f.write(f"Naive Bayes Accuracy: {nb_acc:.4f}\n")

print(f"✅ Logistic Regression Accuracy: {lr_acc:.4f}")
print(f"✅ Naive Bayes Accuracy: {nb_acc:.4f}")
print("✅ Models and vectorizer saved.")
