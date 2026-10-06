import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
import yfinance as yf
from prophet import Prophet

from rl_agent import train_agent, predict_signal
from rl_env import SIGNAL_MAP

from predictor import _finbert_model, _finbert_tokenizer, _load_finbert, _score_headlines, FEATURE_COLS, BUY_THRESHOLD, SELL_THRESHOLD

def evaluate_pipeline(symbol: str):
    # 1. Fetch data
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1y", interval="1d")
    hist.reset_index(inplace=True)
    hist = hist.rename(columns={"Date": "ds", "Close": "y"})
    hist["ds"] = hist["ds"].dt.tz_localize(None)

    # 2. Prophet MAE / RMSE
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(hist[["ds", "y"]])
    forecast = m.predict(hist[["ds"]])
    
    merged_prophet = pd.merge(hist, forecast[["ds", "yhat"]], on="ds")
    
    y_true_p = merged_prophet["y"].values
    y_pred_p = merged_prophet["yhat"].values
    
    mae = mean_absolute_error(y_true_p, y_pred_p)
    rmse = np.sqrt(mean_squared_error(y_true_p, y_pred_p))
    
    # 3. Simulate Classification Metrics
    # To get proper classes, we need features. But we can just use the hist to build basic features
    df = merged_prophet.copy()
    df["prophet_gap"] = (df["y"] - df["yhat"]) / df["yhat"]
    
    # fake sentiment for historical
    np.random.seed(42)
    df["sentiment_1d"] = np.random.uniform(-1, 1, size=len(df))
    df["sentiment_3d_ma"] = df["sentiment_1d"].rolling(3).mean().fillna(0)
    df["sentiment_5d_ma"] = df["sentiment_1d"].rolling(5).mean().fillna(0)
    df["price_momentum_5d"] = df["y"].pct_change(5).fillna(0)
    df["volatility_5d"] = df["y"].rolling(5).std().fillna(0)
    df["forecast_band"] = 0.05 # mock
    df.dropna(inplace=True)

    # Compute target labels 
    df["next_return"] = df["y"].shift(-1) / df["y"] - 1.0
    df.dropna(inplace=True)
    
    def _label(ret):
        if ret > BUY_THRESHOLD: return "BUY"
        if ret < SELL_THRESHOLD: return "SELL"
        return "HOLD"
    
    df["signal"] = df["next_return"].apply(_label)
    
    X = df[FEATURE_COLS].values
    le = LabelEncoder()
    y_class = le.fit_transform(df["signal"])
    
    # Train RL agent
    rl_model = train_agent(df, total_timesteps=20_000, seed=42)

    # Evaluate by running agent on the dataset
    all_preds = []
    all_true = list(y_class)
    for i in range(len(df)):
        row = df.iloc[[i]]
        obs = row[FEATURE_COLS].values.flatten()
        result = predict_signal(rl_model, obs)
        all_preds.append(SIGNAL_MAP.get(result["signal"], 1))
        
    accuracy = accuracy_score(all_true, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_true, all_preds).tolist()
    
    # 4. Sentiment Agreement Rate using IN-FINews Dataset.json
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "IN-FINews Dataset.json")
    agreement_rate = None
    if os.path.exists(dataset_path):
        with open(dataset_path, "r") as f:
            news_data = json.load(f)
            
        headlines = [row["Title"] for row in news_data[:100]] # Limit to 100 for speed
        if headlines:
            # Generate proxy "manual labels" via dumb lexicon matching
            positive_words = {"surge", "jump", "growth", "profit", "gain", "up", "bullish"}
            negative_words = {"fall", "drop", "tumble", "loss", "down", "bearish", "crash"}
            
            proxy_labels = []
            for h in headlines:
                words = set(h.lower().split())
                if words.intersection(positive_words):
                    proxy_labels.append(1) # Pos
                elif words.intersection(negative_words):
                    proxy_labels.append(-1) # Neg
                else:
                    proxy_labels.append(0) # Neutral
            
            finbert_scores = _score_headlines(headlines, batch_size=32)
            
            # Classify finbert scores
            finbert_labels = []
            for s in finbert_scores:
                if s > 0.2: finbert_labels.append(1)
                elif s < -0.2: finbert_labels.append(-1)
                else: finbert_labels.append(0)
                
            agreements = sum(1 for p, f in zip(proxy_labels, finbert_labels) if p == f)
            agreement_rate = agreements / len(headlines)

    return {
        "classification": {
            "classes": le.classes_.tolist(),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm
        },
        "prophet_regression": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4)
        },
        "sentiment_agreement": {
            "description": "Computed against proxy lexicon labels using IN-FINews Dataset",
            "agreement_rate": round(agreement_rate, 4) if agreement_rate is not None else None
        }
    }

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <SYMBOL>")
        print("Example: python evaluator.py ^NSEI")
        sys.exit(1)
        
    symbol = sys.argv[1]
    print(f"Running evaluation for {symbol}...\nThis might take a minute...")
    
    try:
        results = evaluate_pipeline(symbol)
        print("\n--- EVALUATION RESULTS ---")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error during evaluation: {e}")
