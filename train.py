import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# 1. Load Data
print("Loading dataset...")
try:
    df = pd.read_csv("comment_dataset.csv")
    df = df.dropna(subset=['comment_text', 'category'])
    print(f"Loaded {len(df)} comments.")
except FileNotFoundError:
    print("Error: 'comment_dataset.csv' not found.")
    exit()

# 2. Prepare Data
X = df['comment_text']
y = df['category']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build Pipeline (Switched to LogisticRegression for better stability)
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('clf', LogisticRegression(random_state=42, max_iter=1000))
])

# 4. Train Model
print("Training model...")
model_pipeline.fit(X_train, y_train)

# 5. Check Scores
# This is how well it learned the examples you gave it (Should be high!)
train_accuracy = model_pipeline.score(X_train, y_train)
# This is how well it guesses on unseen data
test_accuracy = model_pipeline.score(X_test, y_test)

print("-" * 30)
print(f"✅ Training Accuracy: {train_accuracy:.2%}") 
print(f"📊 Test Set Accuracy: {test_accuracy:.2%}")
print("-" * 30)

# Retrain on ALL data for the final app
model_pipeline.fit(X, y)

# 6. Save Model
print("Saving model to 'comment_model.pkl'...")
joblib.dump(model_pipeline, 'comment_model.pkl')
print("Done! You can now run 'streamlit run app.py'.")