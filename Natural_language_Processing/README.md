# Sentiment Analysis (NLTK + scikit-learn)

This README explains `Sentiment_Analysis.py` in this folder in a clear, college-student style: what the script does, how to run it, required packages, and simple improvements.

**What the script does**
- Loads the NLTK `movie_reviews` corpus (about 2k labeled movie reviews).
- Preprocesses each review: lowercasing, removing non-alpha tokens, removing stopwords, and stemming with PorterStemmer.
- Converts the cleaned text to numerical features with two approaches:
  - Bag-of-words (CountVectorizer with unigrams+bigrams)
  - TF-IDF (TfidfVectorizer with unigrams+bigrams)
- Trains four models: Multinomial Naive Bayes and Logistic Regression on both vectorizer outputs.
- Evaluates performance on a held-out test split and shows accuracy, classification report, and a confusion matrix.
- Provides a `predict_sentiment()` helper to test custom sentences (note: the helper in the current script is simplified and has issues — see "Notes" below).

**How to run (quick)**
1. Open a PowerShell prompt in the project root (or the `Natural_language_Processing` folder).
2. Install dependencies (example):

```powershell
python -m pip install --upgrade pip
python -m pip install nltk pandas matplotlib seaborn scikit-learn
```

3. Run the script:

```powershell
python Natural_language_Processing\Sentiment_Analysis.py
```

You should see plots (review length distribution, class counts, confusion matrix), printed model accuracies, and example predictions.

**Required data / downloads**
- The script uses NLTK corpora and tokenizers. On first run it calls `nltk.download(...)`. If you run into LookupError for a resource (e.g., `punkt_tab`), run:

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('movie_reviews')"
```

**What the outputs mean (simple)**
- **Accuracy**: fraction of correct labels on the test set.
- **Classification report**: precision/recall/F1 per class (`pos`/`neg`). Use F1 when classes are imbalanced.
- **Confusion matrix**: how many actual positives were predicted as negatives and vice versa.

**Notes / Known issues in the current script**
- **`predict_sentiment()` is broken/inconsistent**: it does not reuse the trained model objects and uses a strange expression for Naive Bayes that retrains inside the function. This can produce incorrect or meaningless results.
- **What to fix**: keep trained model objects and vectorizers, then transform the preprocessed input and call `model.predict()` or `model.predict_proba()` on that vectorized input. Example flow:
  - `processed = preprocess_text(word_tokenize(text))`
  - `X = vec.transform([processed])`
  - `pred = model.predict(X)[0]`

**Small, safe improvements (easy project tasks)**
- Store model objects with their vectorizers in a dict: `models = {'LR-Tfidf': (lr_tfidf, tfidf_vec), ...}` so inference is simple and correct.
- Replace stemming with lemmatization (`WordNetLemmatizer`) for clearer tokens.
- Use `Pipeline` objects (scikit-learn) to bundle preprocessing/vectorization + model and make saving/loading easier with `joblib`.
- Save the best model to disk (`joblib.dump`) and add a small `predict_cli.py` to load it and classify input sentences.
- Add confusion-matrix labels and per-class metrics to a saved report for experiments.

**Example fixed inference function (concept)**
```python
def predict_sentiment_fixed(text, model_key='LR-Tfidf'):
    model, vec = models_as_objects[model_key]
    processed = preprocess_text(word_tokenize(text))
    X = vec.transform([processed])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0,1] if hasattr(model, 'predict_proba') else None
    return pred, prob
```

**Ideas for a class project**
- Compare models with/without stemming or lemmatization.
- Add hyperparameter tuning via `GridSearchCV` for `C` in logistic regression or alpha in Naive Bayes.
- Build a small web UI (Flask) to demo live predictions using the saved model.

If you want, I can: (pick one)
- Patch `Sentiment_Analysis.py` to fix `predict_sentiment()` and store model objects.
- Add a `requirements.txt` and a `predict_cli.py` that loads a saved model.
- Implement a `Pipeline` and save the best model with `joblib`.

Pick one and I’ll implement it.
