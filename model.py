import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Clean text and encode labels
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

data['cleaned_message'] = data['message'].apply(clean_text)
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['cleaned_message'])
y = data['label']

# Train the model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
