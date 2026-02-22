"""
predictor.py
------------
Per-stock prediction pipeline:
    1. Fetch price history (yfinance) and train Prophet
    2. Scrape real-time news (scraper.py) and run FinBERT
    3. Merge, build features, train Random Forest classifier
    4. Return Buy / Hold / Sell signal + SHAP explanation

Results are cached in-memory for CACHE_TTL seconds to avoid
re-running the full pipeline on every request.
"""

import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import shap
import torch
import yfinance as yf
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from transformers import BertForSequenceClassification, BertTokenizer

from scraper import scrape_news, STOCK_QUERIES

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
CACHE_TTL  = 3600   # seconds before a cached result expires (1 hour)
BUY_THRESHOLD  =  0.003
SELL_THRESHOLD = -0.003

FEATURE_COLS = [
    "prophet_gap",
    "forecast_band",
    "sentiment_1d",
    "sentiment_3d_ma",
    "sentiment_5d_ma",
    "price_momentum_5d",
    "volatility_5d",
]

SUPPORTED_STOCKS = {
    "^NSEI":         "NIFTY 50",
    "^BSESN":        "Sensex",
    "RELIANCE.NS":   "Reliance Industries",
    "TCS.NS":        "TCS",
    "HDFCBANK.NS":   "HDFC Bank",
    "INFY.NS":       "Infosys",
    "ICICIBANK.NS":  "ICICI Bank",
    "WIPRO.NS":      "Wipro",
    "SBIN.NS":       "SBI",
    "BHARTIARTL.NS": "Bharti Airtel",
}
# ─────────────────────────────────────────────────────────────────────────────

# Lazy-loaded FinBERT (shared across all stocks, loaded once)
_finbert_tokenizer = None
_finbert_model     = None


def _load_finbert():
    global _finbert_tokenizer, _finbert_model
    if _finbert_tokenizer is None:
        print("[predictor] Loading FinBERT model …")
        _finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        _finbert_model     = BertForSequenceClassification.from_pretrained(
            "yiyanghkust/finbert-tone"
        )
        _finbert_model.eval()
        print("[predictor] FinBERT ready.")


def _score_headlines(headlines: list[str], batch_size: int = 32) -> list[float]:
    """Return pos−neg sentiment score for each headline."""
    _load_finbert()
    scores = []
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        inputs = _finbert_tokenizer(
            batch, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        )
        with torch.no_grad():
            probs = torch.softmax(_finbert_model(**inputs).logits, dim=-1)
        scores.extend((probs[:, 1] - probs[:, 2]).tolist())  # positive - negative
    return scores


# ── In-memory cache ──────────────────────────────────────────────────────────
_cache: dict[str, dict] = {}   # symbol → {result, expires_at}


def _label(ret: float) -> str:
    if ret > BUY_THRESHOLD:  return "BUY"
    if ret < SELL_THRESHOLD: return "SELL"
    return "HOLD"


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ds").reset_index(drop=True).copy()
    df["daily_ret"]         = df["y"].pct_change()
    df["prophet_gap"]       = (df["yhat"] - df["y"]) / df["y"]
    df["forecast_band"]     = (df["yhat_upper"] - df["yhat_lower"]) / df["y"]
    df["sentiment_1d"]      = df["sentiment"]
    df["sentiment_3d_ma"]   = df["sentiment"].rolling(3, min_periods=1).mean()
    df["sentiment_5d_ma"]   = df["sentiment"].rolling(5, min_periods=1).mean()
    df["price_momentum_5d"] = df["y"].pct_change(5)
    df["volatility_5d"]     = df["daily_ret"].rolling(5, min_periods=2).std()
    return df


def predict(symbol: str) -> dict:
    """
    Run the full pipeline for *symbol* and return a prediction dict.
    Results are cached for CACHE_TTL seconds.
    """
    # ── Cache check ──────────────────────────────────────────────────────────
    cached = _cache.get(symbol)
    if cached and time.time() < cached["expires_at"]:
        print(f"[predictor] {symbol}: returning cached result")
        return cached["result"]

    print(f"\n[predictor] === Running pipeline for {symbol} ===")

    # ── 1. Price data ────────────────────────────────────────────────────────
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=365)

    raw = yf.download(
        symbol, start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True, progress=False,
    )
    if raw.empty:
        raise ValueError(f"No price data for {symbol}")

    raw = raw.reset_index()
    prices = raw[["Date", "Close"]].copy()
    prices.columns = ["ds", "y"]
    prices["ds"] = pd.to_datetime(prices["ds"])
    prices["y"]  = prices["y"].astype(float)

    current_price = float(prices["y"].iloc[-1])
    print(f"[predictor] {symbol}: {len(prices)} price days, last close {current_price:.2f}")

    # ── 2. Prophet ───────────────────────────────────────────────────────────
    prophet_df = prices[["ds", "y"]].copy()
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(prophet_df)
    future   = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    print(f"[predictor] {symbol}: Prophet fitted")

    # ── 3. Scrape news ───────────────────────────────────────────────────────
    articles = scrape_news(symbol, max_articles=150)
    if not articles:
        raise ValueError(f"No news articles found for {symbol}")

    news_df = pd.DataFrame(articles)
    news_df["ds"] = pd.to_datetime(news_df["date"])
    news_df = news_df[news_df["ds"] >= prices["ds"].min()]

    # FinBERT
    headlines = news_df["title"].tolist()
    print(f"[predictor] {symbol}: running FinBERT on {len(headlines)} headlines …")
    news_df["sentiment"] = _score_headlines(headlines)

    # ── Attach sentiment scores to the articles list for the API ─────────────
    for art, score in zip(articles, news_df["sentiment"].tolist()):
        art["sentiment"] = round(score, 4)

    daily_sent = (
        news_df.groupby("ds")["sentiment"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment": "sentiment"})
    )

    # ── 4. Merge & feature engineering ──────────────────────────────────────
    merged = pd.merge(prices, forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="inner")
    merged = pd.merge(merged, daily_sent, on="ds", how="inner")
    merged = _build_features(merged)

    merged["next_return"] = merged["y"].shift(-1) / merged["y"] - 1
    merged["signal"]      = merged["next_return"].map(_label)
    merged = merged.iloc[:-1].dropna(subset=FEATURE_COLS + ["signal"])

    if len(merged) < 15:
        raise ValueError(
            f"Not enough overlapping data for {symbol} "
            f"(only {len(merged)} rows after merge). "
            "The scraped news date range may not overlap with the price data."
        )

    print(f"[predictor] {symbol}: {len(merged)} samples, "
          f"classes: {merged['signal'].value_counts().to_dict()}")

    # ── 5. Train classifier ──────────────────────────────────────────────────
    X       = merged[FEATURE_COLS].values
    le      = LabelEncoder()
    y       = le.fit_transform(merged["signal"].values)
    classes = le.classes_

    # CV evaluation (only if enough samples)
    cv_f1 = None
    if len(merged) >= 30:
        tscv = TimeSeriesSplit(n_splits=min(5, len(merged) // 10))
        all_preds, all_true = [], []
        for train_idx, test_idx in tscv.split(X):
            clf = RandomForestClassifier(
                n_estimators=300, max_depth=6,
                class_weight="balanced", random_state=42,
            )
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[test_idx])
            all_preds.extend(preds)
            all_true.extend(y[test_idx])
        cv_f1 = round(f1_score(all_true, all_preds, average="weighted", zero_division=0), 4)

    # Final model on all data
    final_clf = RandomForestClassifier(
        n_estimators=300, max_depth=6,
        class_weight="balanced", random_state=42,
    )
    final_clf.fit(X, y)

    # ── 6. SHAP ──────────────────────────────────────────────────────────────
    feature_df  = pd.DataFrame(X, columns=FEATURE_COLS)
    explainer   = shap.TreeExplainer(final_clf)
    shap_vals   = explainer.shap_values(feature_df)    # (n, p, c)

    # Mean absolute SHAP per feature for the predicted class
    latest_row   = merged.iloc[[-1]]
    X_live       = latest_row[FEATURE_COLS].values
    pred_enc     = final_clf.predict(X_live)[0]
    pred_label   = le.inverse_transform([pred_enc])[0]
    proba        = final_clf.predict_proba(X_live)[0]
    proba_dict   = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

    pred_class_idx = int(pred_enc)
    shap_latest    = shap_vals[-1, :, pred_class_idx]          # shape (n_features,)
    shap_dict      = {f: round(float(v), 6) for f, v in zip(FEATURE_COLS, shap_latest)}

    prophet_yhat  = float(merged.iloc[-1]["yhat"])
    latest_sent   = float(merged.iloc[-1]["sentiment_1d"])
    latest_date   = str(merged.iloc[-1]["ds"].date())

    result = {
        "symbol":        symbol,
        "name":          SUPPORTED_STOCKS.get(symbol, symbol),
        "date":          latest_date,
        "price":         round(current_price, 2),
        "prophet_yhat":  round(prophet_yhat, 2),
        "sentiment":     round(latest_sent, 4),
        "signal":        pred_label,
        "probabilities": proba_dict,
        "shap_values":   shap_dict,
        "cv_weighted_f1": cv_f1,
        "news":          articles[:20],   # top 20 newest for display
    }

    # Store in cache
    _cache[symbol] = {"result": result, "expires_at": time.time() + CACHE_TTL}
    print(f"[predictor] {symbol}: DONE — signal={pred_label}, F1={cv_f1}")
    return result
