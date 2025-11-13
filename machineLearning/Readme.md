# SPMP — Simple Price Movement Predictor (NVDA)

This is a short, student-style explanation of the `SPMP.py` script in this folder. The script is a minimal proof-of-concept that uses logistic regression to try and predict whether Nvidia's (NVDA) closing price will be higher on the next trading day.

## What the script does (in plain terms)
- Loads historical price data from `NVDA.csv` (expected columns include `Date,Open,High,Low,Close,Volume`).
- Creates a binary label called `Target`: 1 if the next trading day's close is higher than today's close, else 0. This is done with `data['Close'].shift(-1)`.
- Computes two simple features for each row:
	- `Open_Close` = Open - Close (same day)
	- `High_Low` = High - Low (same day)
- Trains a `LogisticRegression` model using all rows except the last 100 as training data.
- Evaluates the model on the last 100 rows (test set), printing accuracy, confusion matrix, and a classification report.
- Plots the predicted probability that the price will go up (P(class=1)) for the test set.

## Inputs and outputs
- Input: `NVDA.csv` in the same folder as `SPMP.py`.
- Outputs printed to console: first few rows and info, accuracy, confusion matrix, classification report, and a plot of predicted probabilities for the test set.
- Visual output: a matplotlib plot of predicted probabilities for the test set.

## What "tomorrow" actually means here
- The script uses `shift(-1)` to make labels — so "tomorrow" is literally the next row in the CSV (the next trading day). During training this is historical data 
- For live use the prediction can represent the next trading day only if you compute features only from information available when you make the prediction 

## How to run (quick)
1. Make sure `NVDA.csv` is in this folder.
2. Install dependencies (PowerShell example):

```powershell
python -m pip install --upgrade pip
python -m pip install pandas matplotlib scikit-learn
```

3. Run the script from this folder:

```powershell
python SPMP.py
```

You should see console output and a probability plot.

## Final note
This is a compact learning exercise — it’s great for understanding how to label time-series data for supervised learning and for experimenting with simple features and models. If you want, I can (a) change the features to lagged ones so you can predict earlier in the day, or (b) add a walk-forward validation/backtest to simulate production behavior.

Enjoy experimenting — and don't forget to test carefully before trusting any model with real money.

