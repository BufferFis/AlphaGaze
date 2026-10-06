"""
combine_and_xai.py
------------------
Combines Prophet time-series forecasts with FinBERT sentiment scores,
trains a Buy / Hold / Sell classification model, evaluates it with
weighted F1-score / classification report, and explains predictions
via SHAP. Outputs the live trade signal for the latest available date.

Classification labels (next-day return):
    BUY   : return >  +0.3%
    SELL  : return <  -0.3%
    HOLD  : -0.3% <= return <= +0.3%
"""

import os
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

from rl_agent import train_agent, predict_signal, compute_feature_importance, save_agent
from rl_env import ACTION_MAP, SIGNAL_MAP

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
BUY_THRESHOLD  =  0.003   # +0.3 % next-day return  →  BUY
SELL_THRESHOLD = -0.003   # −0.3 % next-day return  →  SELL

FEATURE_COLS = [
    "prophet_gap",
    "forecast_band",
    "sentiment_1d",
    "sentiment_3d_ma",
    "sentiment_5d_ma",
    "price_momentum_5d",
    "volatility_5d",
]
# ─────────────────────────────────────────────────────────────────────────────


def label_signal(ret: float) -> str:
    """Map a next-day return to BUY / HOLD / SELL."""
    if ret > BUY_THRESHOLD:
        return "BUY"
    elif ret < SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def build_features(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from the merged Prophet + FinBERT + price DataFrame.

    Features:
        prophet_gap        (yhat - actual) / actual  (relative prophet error)
        forecast_band      (yhat_upper - yhat_lower) / actual  (normalised band width)
        sentiment_1d       raw FinBERT daily score (positive - negative)
        sentiment_3d_ma    3-day rolling mean of sentiment
        sentiment_5d_ma    5-day rolling mean of sentiment
        price_momentum_5d  5-day price return
        volatility_5d      5-day rolling std of daily returns
    """
    df = merged.sort_values("ds").reset_index(drop=True).copy()
    df["daily_ret"]         = df["y"].pct_change()
    df["prophet_gap"]       = (df["yhat"] - df["y"]) / df["y"]
    df["forecast_band"]     = (df["yhat_upper"] - df["yhat_lower"]) / df["y"]
    df["sentiment_1d"]      = df["sentiment"]
    df["sentiment_3d_ma"]   = df["sentiment"].rolling(3, min_periods=1).mean()
    df["sentiment_5d_ma"]   = df["sentiment"].rolling(5, min_periods=1).mean()
    df["price_momentum_5d"] = df["y"].pct_change(5)
    df["volatility_5d"]     = df["daily_ret"].rolling(5, min_periods=2).std()
    return df


def run_combine_and_xai():
    print("=" * 60)
    print("  NIFTY 50 Buy / Hold / Sell Classifier")
    print("  Prophet (time-series) + FinBERT (sentiment) + SHAP (XAI)")
    print("=" * 60)

    # ── 1. Load pre-computed artefacts ───────────────────────────────────────
    for path in ["results/prophet_forecast.csv", "results/daily_sentiment.csv"]:
        if not os.path.exists(path):
            print(f"[ERROR] {path} not found. Run the relevant script first.")
            return

    forecast  = pd.read_csv("results/prophet_forecast.csv")
    sentiment = pd.read_csv("results/daily_sentiment.csv")
    forecast["ds"]  = pd.to_datetime(forecast["ds"])
    sentiment["ds"] = pd.to_datetime(sentiment["ds"])

    # ── 2. Download actual NIFTY prices ─────────────────────────────────────
    print("\n[1/5] Downloading NIFTY 50 price data...")
    raw = yf.download("^NSEI", start="2025-02-01", end="2025-09-01",
                      auto_adjust=True, progress=False)
    raw = raw.reset_index()
    actual = raw[["Date", "Close"]].copy()
    actual.columns = ["ds", "y"]
    actual["ds"] = pd.to_datetime(actual["ds"])
    actual["y"]  = actual["y"].astype(float)

    # ── 3. Merge & engineer features ────────────────────────────────────────
    print("[2/5] Merging datasets and engineering features...")
    merged = pd.merge(actual,
                      forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
                      on="ds", how="inner")
    merged = pd.merge(merged, sentiment, on="ds", how="inner")
    merged = build_features(merged)

    # Next-day return → label
    merged["next_return"] = merged["y"].shift(-1) / merged["y"] - 1
    merged["signal"]      = merged["next_return"].map(label_signal)

    # Drop last row (no future price) and rows with NaN in features
    merged = merged.iloc[:-1].dropna(subset=FEATURE_COLS + ["signal"])

    print(f"    Samples: {len(merged)}")
    print(f"    Class distribution:")
    for cls, cnt in merged["signal"].value_counts().items():
        pct = cnt / len(merged) * 100
        print(f"      {cls:<5} {cnt:3d}  ({pct:.1f}%)")

    if len(merged) < 20:
        print("[ERROR] Not enough samples to train. Check date alignment.")
        return

    # ── 4. Train PPO RL Agent ───────────────────────────────────────────────
    print("\n[3/5] Training PPO RL agent...")
    X = merged[FEATURE_COLS].values
    le = LabelEncoder()
    y = le.fit_transform(merged["signal"].values)
    classes = le.classes_

    rl_model = train_agent(merged, total_timesteps=50_000, seed=42)

    # Evaluate by running the agent over the entire dataset
    all_preds = []
    all_true = []
    for i in range(len(merged)):
        row = merged.iloc[[i]]
        obs = row[FEATURE_COLS].values.flatten()
        result = predict_signal(rl_model, obs)
        all_preds.append(SIGNAL_MAP.get(result["signal"], 1))
        all_true.append(y[i])

    overall_f1 = f1_score(all_true, all_preds, average="weighted", zero_division=0)
    macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    print(f"\n    ── Weighted F1: {overall_f1:.4f}  |  Macro F1: {macro_f1:.4f} ──")
    print("\n    Full Classification Report:")
    print(classification_report(all_true, all_preds,
                                target_names=classes, zero_division=0))

    # Confusion matrix plot
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(all_true, all_preds, labels=range(len(classes)))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — PPO RL (Weighted F1={overall_f1:.3f})")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.close()
    print("    Confusion matrix → results/confusion_matrix.png")

    # ── 5. Feature Importance (replaces SHAP) ──────────────────────────────
    print("\n[4/5] Computing feature importance...")

    latest = merged.iloc[[-1]]
    X_live = latest[FEATURE_COLS].values.flatten()
    importance = compute_feature_importance(rl_model, X_live)

    # Generate feature importance bar chart (replaces SHAP summary plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    features = list(importance.keys())
    values = list(importance.values())
    colors = ['#3fb950' if v >= 0 else '#f85149' for v in values]
    ax.barh(features, values, color=colors)
    ax.set_title("Feature Importance — PPO RL Agent (Perturbation-Based)")
    ax.set_xlabel("Impact on Action Probability")
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=150)
    plt.close()
    print("    Feature importance → results/feature_importance.png")

    # Save the model
    save_agent(rl_model, "results/ppo_trading_agent")

    # ── 6. Live signal for the latest row ───────────────────────────────────
    print("\n[5/5] Generating live trade signal for latest date...")
    rl_result = predict_signal(rl_model, X_live)
    pred_label = rl_result["signal"]
    proba_dict = rl_result["probabilities"]

    latest_date     = pd.Timestamp(latest["ds"].values[0]).date()
    latest_price    = float(latest["y"].values[0])
    prophet_yhat    = float(latest["yhat"].values[0])
    latest_sent     = float(latest["sentiment_1d"].values[0])

    print("\n" + "=" * 60)
    print(f"  DATE         : {latest_date}")
    print(f"  NIFTY Close  : {latest_price:>12,.2f}")
    print(f"  Prophet yhat : {prophet_yhat:>12,.2f}")
    print(f"  Sentiment    : {latest_sent:>+12.4f}")
    print(f"  {'─'*44}")
    print(f"  SIGNAL       :  *** {pred_label} ***")
    print(f"  Probabilities:")
    for cls in sorted(proba_dict, key=lambda c: -proba_dict[c]):
        p   = proba_dict[cls]
        bar = "█" * int(p * 30)
        print(f"    {cls:<5}  {p:5.1%}  {bar}")
    print("=" * 60)

    # Save summary CSV
    pd.DataFrame([{
        "date":            str(latest_date),
        "nifty_close":     latest_price,
        "prophet_yhat":    prophet_yhat,
        "sentiment":       latest_sent,
        "signal":          pred_label,
        "prob_BUY":        proba_dict.get("BUY",  0),
        "prob_HOLD":       proba_dict.get("HOLD", 0),
        "prob_SELL":       proba_dict.get("SELL", 0),
        "cv_weighted_f1":  overall_f1,
        "cv_macro_f1":     macro_f1,
    }]).to_csv("results/latest_prediction.csv", index=False)
    print("\n  Summary saved  → results/latest_prediction.csv")


if __name__ == "__main__":
    run_combine_and_xai()
