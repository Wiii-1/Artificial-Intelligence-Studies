# Dependencies
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load CSV from the script directory to avoid path issues when running from another cwd
csv_path = os.path.join(os.path.dirname(__file__), 'NVDA.csv')
data = pd.read_csv(csv_path)

print(data.head())
print(data.info())

# Create label: whether the close price tomorrow is higher than today
data['Tomorrow'] = data['Close'].shift(-1)
data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)

# Simple features: today's open-close difference and high-low range
data['Open_Close'] = data['Open'] - data['Close']
data['High_Low'] = data['High'] - data['Low']

data = data.dropna()

# Use the actual column names we created above
prediction_features = ['Open_Close', 'High_Low']

# Simple time-based train/test split: last 100 rows as test
train = data.iloc[:-100]
test = data.iloc[-100:]

X_train = train[prediction_features]
y_train = train['Target']
X_test = test[prediction_features]
y_test = test['Target']

# Slightly increase max_iter to help convergence on small datasets
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, preds))
print('Confusion Matrix:\n', confusion_matrix(y_test, preds))
print('Classification Report:\n', classification_report(y_test, preds))

# Get the probability of the positive class (class=1)
probs_pos = model.predict_proba(X_test)[:, 1]
# Predictions using a confidence threshold (e.g., 0.6)
preds_confident = (probs_pos > 0.6).astype(int)

print(f"Confident predictions (p>0.6): {preds_confident.sum()} / {len(preds_confident)}")

plt.figure()
plt.plot(probs_pos)
plt.title('Nvidia Stock Price Movement Prediction (P(class=1))')
plt.xlabel('Test sample index')
plt.ylabel('Predicted probability')
plt.show()