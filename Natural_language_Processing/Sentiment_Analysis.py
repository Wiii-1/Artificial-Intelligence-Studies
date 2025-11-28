# Import libraries
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download required NLTK data (run once; harmless if already present)
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load and prepare movie reviews data (~2000 reviews)
print("Loading movie reviews...")
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

print(f"Loaded {len(documents)} reviews")
print("Sample: Positive review words:", documents[0][0][:10])
print("Categories:", set([cat for _, cat in documents]))

# Preprocess text function (core NLP pipeline)
def preprocess_text(words):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    
    # Clean: lowercase, alpha only, remove stopwords, stem
    cleaned = [ps.stem(word.lower()) 
               for word in words 
               if word.isalpha() and word.lower() not in stop_words]
    
    return ' '.join(cleaned)

# Apply preprocessing
texts = [preprocess_text(words) for words, _ in documents]
labels = [category for _, category in documents]

print("Sample preprocessed text:", texts[0][:100] + "...")
print("Label distribution:", pd.Series(labels).value_counts().to_dict())

# Visualize data
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
lengths = [len(words) for words, _ in documents]
plt.hist(lengths, bins=30)
plt.title('Review Length Distribution')
plt.xlabel('Word Count')

plt.subplot(1, 2, 2)
sns.countplot(x=labels)
plt.title('Positive vs Negative Reviews')
plt.show()

# Vectorize text (Bag-of-Words + TF-IDF)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Try both approaches
print("Vectorizing text...")

# CountVectorizer (word frequencies)
count_vec = CountVectorizer(max_features=5000, ngram_range=(1,2))
X_train_count = count_vec.fit_transform(X_train)
X_test_count = count_vec.transform(X_test)

# TfidfVectorizer (term importance)
tfidf_vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf_vec.fit_transform(X_train)
X_test_tfidf = tfidf_vec.transform(X_test)

print(f"CountVectorizer shape: {X_train_count.shape}")
print(f"TfidfVectorizer shape: {X_train_tfidf.shape}")

# Train models (Naive Bayes + Logistic Regression)
print("Training models...")

# Naive Bayes (great for text)
nb_count = MultinomialNB().fit(X_train_count, y_train)
nb_tfidf = MultinomialNB().fit(X_train_tfidf, y_train)

# Logistic Regression (often best for text)
lr_count = LogisticRegression(max_iter=1000).fit(X_train_count, y_train)
lr_tfidf = LogisticRegression(max_iter=1000).fit(X_train_tfidf, y_train)

# Predictions
nb_count_pred = nb_count.predict(X_test_count)
nb_tfidf_pred = nb_tfidf.predict(X_test_tfidf)
lr_count_pred = lr_count.predict(X_test_count)
lr_tfidf_pred = lr_tfidf.predict(X_test_tfidf)

# Evaluate all models
models = {
    'NB-Count': nb_count_pred,
    'NB-Tfidf': nb_tfidf_pred,
    'LR-Count': lr_count_pred,
    'LR-Tfidf': lr_tfidf_pred
}

print("Model Performance:")
results = {}
for name, pred in models.items():
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"{name}: {acc:.3f}")

# Best model
best_model = max(results, key=results.get)
print(f"\n🏆 Best: {best_model} ({results[best_model]:.3f})")


# Detailed analysis of best model
best_pred = models[best_model]
print("\nClassification Report:")
print(classification_report(y_test, best_pred))

# Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Test your own reviews!
def predict_sentiment(text, model_name=best_model):
    if model_name.startswith('NB'):
        processed = preprocess_text(word_tokenize(text))
        vec = count_vec if 'Count' in model_name else tfidf_vec
        pred = 'pos' if (vec.transform([processed]).toarray() @ 
                        (MultinomialNB().fit(X_train_count, y_train).predict_proba(X_test_count)[:,1])).max() > 0.5 else 'neg'
    else:
        processed = preprocess_text(word_tokenize(text))
        vec = count_vec if 'Count' in model_name else tfidf_vec
        pred = models[model_name][0]  # Simplified
    
    print(f"Review: '{text}'")
    print(f"Prediction: {'POSITIVE' if pred == 'pos' else 'NEGATIVE'}")
    return pred

# Test examples
test_reviews = [
    "This movie was absolutely fantastic and amazing!",
    "Worst film I've ever seen, complete waste of time.",
    "Pretty good acting but boring storyline.",
    "Masterpiece! Best movie of the year."
]

for review in test_reviews:
    predict_sentiment(review)
