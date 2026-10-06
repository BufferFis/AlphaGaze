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

import os
import time
import warnings
import re
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
from rl_agent import train_agent, predict_signal, compute_feature_importance, load_agent
from rl_env import ACTION_MAP

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

PRETRAINED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ppo_pretrained")
# ─────────────────────────────────────────────────────────────────────────────

# Lazy-loaded FinBERT (shared across all stocks, loaded once)
_finbert_tokenizer = None
_finbert_model     = None

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]"}


def _load_finbert():
    global _finbert_tokenizer, _finbert_model
    if _finbert_tokenizer is None:
        print("[predictor] Loading FinBERT model …")
        _finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        _finbert_model     = BertForSequenceClassification.from_pretrained(
            "yiyanghkust/finbert-tone",
            attn_implementation="eager",  # required for output_attentions=True
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


def _is_noise_token(token: str) -> bool:
    """Return True for tokens that should not drive attention visualization."""
    if token in SPECIAL_TOKENS:
        return True
    if not token or token.strip() == "":
        return True
    # Drop punctuation-only pieces after subword merge.
    return all(not ch.isalnum() for ch in token)


def _aggregate_attention_keywords(attention_maps: list[dict], top_k: int = 12) -> list[dict]:
    """Build an attention-weighted keyword summary across headlines."""
    scores: dict[str, dict] = {}
    for item in attention_maps:
        for tok, attn in zip(item.get("tokens", []), item.get("attention", [])):
            token = str(tok).lower().strip()
            if _is_noise_token(token):
                continue
            entry = scores.setdefault(token, {"sum": 0.0, "count": 0})
            entry["sum"] += float(attn)
            entry["count"] += 1

    ranked = sorted(
        (
            {
                "token": token,
                "avg_attention": round(values["sum"] / values["count"], 4),
                "mentions": values["count"],
            }
            for token, values in scores.items()
            if values["count"] > 0
        ),
        key=lambda x: (x["avg_attention"], x["mentions"]),
        reverse=True,
    )
    return ranked[:top_k]


def _get_attention_maps(headlines: list[str], max_headlines: int = 5) -> list[dict]:
    """
    Extract attention maps from FinBERT for the given headlines.

    For each headline, returns token-level attention weights averaged
    across all attention heads in the last layer. This shows which
    words BERT 'attended to' most when making sentiment decisions.

    Returns
    -------
    list of {"headline": str, "tokens": list[str],
             "attention": list[float], "sentiment": float,
             "token_count": int, "unk_ratio": float}
    """
    _load_finbert()
    results = []
    for text in headlines[:max_headlines]:
        inputs = _finbert_tokenizer(
            text, return_tensors="pt",
            padding=True, truncation=True, max_length=128,
        )
        with torch.no_grad():
            outputs = _finbert_model(
                **inputs, output_attentions=True
            )

        probs = torch.softmax(outputs.logits, dim=-1)
        sentiment = float((probs[0, 1] - probs[0, 2]).item())

        # outputs.attentions is a tuple of (batch, heads, seq, seq)
        # Take the last layer, average across all heads
        last_layer_attn = outputs.attentions[-1]           # (1, heads, seq, seq)
        # Average over heads → (1, seq, seq)
        avg_attn = last_layer_attn.mean(dim=1).squeeze(0)  # (seq, seq)

        # CLS token's attention to all other tokens (row 0)
        cls_attn = avg_attn[0].cpu().numpy()               # (seq,)

        # Get tokens
        token_ids = inputs["input_ids"][0]
        tokens = _finbert_tokenizer.convert_ids_to_tokens(token_ids)

        # Attention mask to ignore padding
        attn_mask = inputs["attention_mask"][0].cpu().numpy()

        total_nonpad = int(attn_mask.sum())
        unk_count = sum(1 for tok, mask_val in zip(tokens, attn_mask) if mask_val == 1 and tok == "[UNK]")

        # Merge subword tokens and their attention scores.
        # We keep the strongest sub-piece attention per merged token.
        merged_tokens = []
        merged_attn = []
        for tok, attn_val, mask_val in zip(tokens, cls_attn, attn_mask):
            if mask_val == 0:
                continue
            if tok in SPECIAL_TOKENS:
                continue
            if tok.startswith("##") and merged_tokens:
                merged_tokens[-1] += tok[2:]
                merged_attn[-1] = max(merged_attn[-1], float(attn_val))
            else:
                merged_tokens.append(tok)
                merged_attn.append(float(attn_val))

        # Remove punctuation/noise tokens to reduce visual artifacts.
        filtered_tokens = []
        filtered_attn = []
        for tok, attn_val in zip(merged_tokens, merged_attn):
            if _is_noise_token(tok):
                continue
            filtered_tokens.append(tok)
            filtered_attn.append(attn_val)

        # Normalize attention scores to 0-1 range
        if filtered_attn:
            max_a = max(filtered_attn)
            min_a = min(filtered_attn)
            rng = max_a - min_a if max_a != min_a else 1.0
            filtered_attn = [(a - min_a) / rng for a in filtered_attn]

        results.append({
            "headline": text,
            "tokens": filtered_tokens,
            "attention": [round(a, 4) for a in filtered_attn],
            "sentiment": round(sentiment, 4),
            "token_count": len(filtered_tokens),
            "unk_ratio": round((unk_count / total_nonpad) if total_nonpad > 0 else 0.0, 4),
        })

    return results


def _prophet_sensitivity(prices_df: pd.DataFrame, prophet_model,
                          forecast_df: pd.DataFrame,
                          n_segments: int = 10) -> dict:
    """
    GradCAM-like sensitivity analysis for Prophet.

    Divides the historical price series into `n_segments` time windows
    and measures how much the forecast changes when each segment is
    perturbed. Higher sensitivity = that time window had more influence
    on the final forecast.

    Also extracts Prophet changepoints and trend decomposition.

    Returns
    -------
    dict with:
      - segments: list of {start, end, sensitivity}
      - changepoints: list of {date, magnitude}
      - forecast_chart: {dates, actual, yhat, yhat_lower, yhat_upper}
    """
    import copy

    df = prices_df.copy().sort_values("ds").reset_index(drop=True)
    n = len(df)
    seg_size = max(n // n_segments, 1)

    # Baseline: mean of yhat for the last 30 forecast days
    future_mask = forecast_df["ds"] > df["ds"].max()
    future_forecast = forecast_df[future_mask]
    baseline_yhat = future_forecast["yhat"].mean() if len(future_forecast) > 0 else forecast_df["yhat"].iloc[-1]

    segments = []
    for i in range(n_segments):
        start_idx = i * seg_size
        end_idx = min((i + 1) * seg_size, n)
        if start_idx >= n:
            break

        # Perturb this segment: add 2% noise
        perturbed = df.copy()
        noise_std = perturbed["y"].iloc[start_idx:end_idx].std() * 0.1
        if noise_std == 0 or pd.isna(noise_std):
            noise_std = perturbed["y"].mean() * 0.02
        np.random.seed(42 + i)
        noise = np.random.normal(0, noise_std, end_idx - start_idx)
        perturbed.loc[start_idx:end_idx - 1, "y"] = (
            perturbed.loc[start_idx:end_idx - 1, "y"].values + noise
        )

        # Re-fit Prophet on perturbed data
        try:
            m_pert = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
            )
            m_pert.fit(perturbed[["ds", "y"]])
            future_pert = m_pert.make_future_dataframe(periods=30)
            fc_pert = m_pert.predict(future_pert)
            future_pert_mask = fc_pert["ds"] > df["ds"].max()
            perturbed_yhat = fc_pert[future_pert_mask]["yhat"].mean()
            sensitivity = abs(perturbed_yhat - baseline_yhat) / abs(baseline_yhat) if baseline_yhat != 0 else 0
        except Exception:
            sensitivity = 0.0

        segments.append({
            "start": str(df.iloc[start_idx]["ds"].date()),
            "end": str(df.iloc[min(end_idx - 1, n - 1)]["ds"].date()),
            "sensitivity": round(float(sensitivity), 6),
        })

    # Normalize sensitivities to 0-1 range
    max_sens = max((s["sensitivity"] for s in segments), default=1)
    if max_sens > 0:
        for s in segments:
            s["sensitivity"] = round(s["sensitivity"] / max_sens, 4)

    # Extract changepoints
    changepoints_data = []
    if hasattr(prophet_model, "changepoints") and prophet_model.changepoints is not None:
        cps = prophet_model.changepoints
        if hasattr(prophet_model, "params") and "delta" in prophet_model.params:
            deltas = prophet_model.params["delta"].flatten()
            for cp, delta in zip(cps, deltas):
                changepoints_data.append({
                    "date": str(cp.date()),
                    "magnitude": round(float(abs(delta)), 6),
                })
            # Sort by magnitude descending
            changepoints_data.sort(key=lambda x: x["magnitude"], reverse=True)

    # Build forecast chart data
    # Merge actual prices with forecast
    chart_merged = pd.merge(
        df[["ds", "y"]], forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds", how="outer"
    ).sort_values("ds")

    forecast_chart = {
        "dates": [str(d.date()) for d in chart_merged["ds"]],
        "actual": [round(float(v), 2) if pd.notna(v) else None for v in chart_merged["y"]],
        "yhat": [round(float(v), 2) for v in chart_merged["yhat"]],
        "yhat_lower": [round(float(v), 2) for v in chart_merged["yhat_lower"]],
        "yhat_upper": [round(float(v), 2) for v in chart_merged["yhat_upper"]],
    }

    return {
        "segments": segments,
        "changepoints": changepoints_data[:10],  # top 10
        "forecast_chart": forecast_chart,
    }


# ── In-memory cache ──────────────────────────────────────────────────────────
_cache: dict[str, dict] = {}   # symbol → {result, expires_at}


def _label(ret: float) -> str:
    if ret > BUY_THRESHOLD:  return "BUY"
    if ret < SELL_THRESHOLD: return "SELL"
    return "HOLD"


def _compute_uncertainty_summary(proba: dict[str, float], latest_row: pd.DataFrame,
                                 sentiment_score: float) -> dict:
    """Compute confidence/uncertainty and a simple actionability recommendation."""
    probs = np.array(list(proba.values()), dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum()

    n_classes = len(probs)
    entropy = float(-(probs * np.log(probs)).sum())
    max_entropy = float(np.log(n_classes)) if n_classes > 1 else 1.0
    entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0

    sorted_probs = np.sort(probs)[::-1]
    top_prob = float(sorted_probs[0])
    second_prob = float(sorted_probs[1]) if n_classes > 1 else 0.0
    margin = top_prob - second_prob
    margin_uncertainty = 1.0 - margin

    forecast_band = abs(float(latest_row["forecast_band"].iloc[0]))
    # 4% band width is treated as high uncertainty; clipped to [0, 1].
    forecast_uncertainty = float(np.clip(forecast_band / 0.04, 0.0, 1.0))

    # Sentiment near 0 is less informative, while stronger polarity lowers uncertainty.
    sentiment_uncertainty = float(np.clip(1.0 - min(abs(sentiment_score), 1.0), 0.0, 1.0))

    uncertainty = (
        0.45 * entropy_norm
        + 0.25 * margin_uncertainty
        + 0.20 * forecast_uncertainty
        + 0.10 * sentiment_uncertainty
    )
    uncertainty = float(np.clip(uncertainty, 0.0, 1.0))
    confidence = 1.0 - uncertainty

    if uncertainty >= 0.67:
        actionability = "NO_TRADE"
    elif uncertainty >= 0.45:
        actionability = "CAUTION"
    else:
        actionability = "TRADE"

    confidence_level = "HIGH" if confidence >= 0.67 else "MEDIUM" if confidence >= 0.45 else "LOW"

    return {
        "score": round(uncertainty, 4),
        "confidence": round(confidence, 4),
        "level": confidence_level,
        "actionability": actionability,
        "components": {
            "entropy": round(float(entropy_norm), 4),
            "margin": round(float(margin_uncertainty), 4),
            "forecast": round(float(forecast_uncertainty), 4),
            "sentiment": round(float(sentiment_uncertainty), 4),
        },
    }


def _counterfactual_headline_tests(attention_maps: list[dict], max_items: int = 3) -> list[dict]:
    """Run lightweight counterfactual tests by removing the most-attended token."""
    if not attention_maps:
        return []

    selected = []
    for item in attention_maps:
        tokens = item.get("tokens") or []
        attention = item.get("attention") or []
        headline = item.get("headline", "")
        if not headline or not tokens or not attention:
            continue

        top_idx = int(np.argmax(attention))
        focus_token = str(tokens[top_idx]).strip()
        if not focus_token:
            continue

        pattern = re.compile(rf"\b{re.escape(focus_token)}\b", flags=re.IGNORECASE)
        counterfactual = pattern.sub("", headline, count=1)
        counterfactual = re.sub(r"\s+", " ", counterfactual).strip(" ,.-")
        if not counterfactual or counterfactual == headline:
            continue

        selected.append({
            "headline": headline,
            "focus_token": focus_token,
            "counterfactual": counterfactual,
        })
        if len(selected) >= max_items:
            break

    if not selected:
        return []

    original_texts = [x["headline"] for x in selected]
    counterfactual_texts = [x["counterfactual"] for x in selected]
    original_scores = _score_headlines(original_texts)
    counterfactual_scores = _score_headlines(counterfactual_texts)

    tests = []
    for item, s0, s1 in zip(selected, original_scores, counterfactual_scores):
        delta = float(s1 - s0)
        tests.append({
            "headline": item["headline"],
            "focus_token": item["focus_token"],
            "counterfactual": item["counterfactual"],
            "original_sentiment": round(float(s0), 4),
            "counterfactual_sentiment": round(float(s1), 4),
            "delta": round(delta, 4),
            "polarity_flip": bool((s0 > 0 and s1 < 0) or (s0 < 0 and s1 > 0)),
        })

    return tests


def _rolling_backtest(merged_df: pd.DataFrame, n_points: int = 90,
                      min_train: int = 20) -> dict:
    """Run walk-forward backtest using PPO RL agent."""
    if merged_df is None or len(merged_df) < 12:
        return {
            "summary": {"samples": 0, "accuracy": None, "avg_confidence": None},
            "series": [],
            "bucket_hit_rate": [],
        }

    df = merged_df.sort_values("ds").reset_index(drop=True)
    effective_min_train = min(min_train, max(10, len(df) // 2))
    records = []

    pretrained_path = PRETRAINED_MODEL_PATH + ".zip"
    base_model = load_agent(PRETRAINED_MODEL_PATH) if os.path.exists(pretrained_path) else None

    for i in range(effective_min_train, len(df)):
        train_data = df.iloc[:i].copy()
        test_row = df.iloc[[i]]

        if len(train_data) < 5:
            continue

        try:
            # Fast fine-tuning on historical window
            agent = train_agent(train_data, total_timesteps=1000, seed=42, pretrained_model=base_model)

            obs = test_row[FEATURE_COLS].values.flatten()
            result = predict_signal(agent, obs)
            pred = result["signal"]
            confidence = max(result["probabilities"].values())
            actual = str(test_row["signal"].iloc[0])

            records.append({
                "date": str(test_row["ds"].iloc[0].date()),
                "predicted": pred,
                "actual": actual,
                "correct": pred == actual,
                "confidence": round(confidence, 4),
                "next_return": round(float(test_row["next_return"].iloc[0]), 6),
            })
        except Exception:
            continue

    if not records:
        return {
            "summary": {"samples": 0, "accuracy": None, "avg_confidence": None},
            "series": [],
            "bucket_hit_rate": [],
        }

    records = records[-n_points:]
    correct = np.array([1 if r["correct"] else 0 for r in records], dtype=float)
    conf = np.array([r["confidence"] for r in records], dtype=float)

    buckets = {"low": (0.0, 0.50), "mid": (0.50, 0.67), "high": (0.67, 1.01)}
    bucket_rows = []
    for name, (lo, hi) in buckets.items():
        idx = [i for i, c in enumerate(conf) if lo <= c < hi]
        if not idx:
            bucket_rows.append({"bucket": name, "samples": 0, "hit_rate": None})
            continue
        hit_rate = float(np.mean(correct[idx]))
        bucket_rows.append({
            "bucket": name, "samples": len(idx), "hit_rate": round(hit_rate, 4),
        })

    return {
        "summary": {
            "samples": len(records),
            "accuracy": round(float(np.mean(correct)), 4),
            "avg_confidence": round(float(np.mean(conf)), 4),
        },
        "series": records,
        "bucket_hit_rate": bucket_rows,
    }


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
    start_date = end_date - timedelta(days=730)  # 2 years for richer RL training

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

    # ── 5. Train RL agent ────────────────────────────────────────────────────
    pretrained_path = PRETRAINED_MODEL_PATH + ".zip"
    pretrained = load_agent(PRETRAINED_MODEL_PATH) if os.path.exists(pretrained_path) else None

    if pretrained is not None:
        print(f"[predictor] {symbol}: Fine-tuning pre-trained PPO on {len(merged)} samples …")
        rl_model = train_agent(merged, total_timesteps=10_000, seed=42,
                               pretrained_model=pretrained)
    else:
        print(f"[predictor] {symbol}: No pre-trained model. Training PPO from scratch on {len(merged)} samples …")
        rl_model = train_agent(merged, total_timesteps=50_000, seed=42)

    # ── 6. RL Prediction ────────────────────────────────────────────────────
    latest_row = merged.iloc[[-1]]
    X_live = latest_row[FEATURE_COLS].values.flatten()

    rl_result = predict_signal(rl_model, X_live)
    pred_label = rl_result["signal"]
    proba_dict = rl_result["probabilities"]

    # ── 7. Feature Importance (replaces SHAP) ───────────────────────────────
    print(f"[predictor] {symbol}: computing feature importance …")
    shap_dict = compute_feature_importance(rl_model, X_live)

    # CV F1 is no longer applicable for RL
    cv_f1 = None

    prophet_yhat  = float(merged.iloc[-1]["yhat"])
    latest_sent   = float(merged.iloc[-1]["sentiment_1d"])
    latest_date   = str(merged.iloc[-1]["ds"].date())
    uncertainty_summary = _compute_uncertainty_summary(proba_dict, latest_row, latest_sent)

    # ── 8. BERT Attention maps for top headlines ─────────────────────────────
    # Pick the most impactful headlines (highest abs sentiment)
    sent_articles = sorted(articles, key=lambda a: abs(a.get("sentiment", 0)), reverse=True)
    top_headlines = [a["title"] for a in sent_articles[:8]]
    print(f"[predictor] {symbol}: extracting attention maps for {len(top_headlines)} headlines …")
    attention_maps = _get_attention_maps(top_headlines, max_headlines=8)
    attention_keywords = _aggregate_attention_keywords(attention_maps, top_k=12)
    counterfactual_tests = _counterfactual_headline_tests(attention_maps, max_items=3)

    # ── 9. Prophet GradCAM-like sensitivity analysis ─────────────────────────
    print(f"[predictor] {symbol}: running Prophet sensitivity analysis …")
    prophet_sensitivity = _prophet_sensitivity(prices, m, forecast, n_segments=10)

    print(f"[predictor] {symbol}: running rolling backtest …")
    rolling_backtest = _rolling_backtest(merged, n_points=90, min_train=60)

    result = {
        "symbol":        symbol,
        "name":          SUPPORTED_STOCKS.get(symbol, symbol),
        "date":          latest_date,
        "price":         round(current_price, 2),
        "prophet_yhat":  round(prophet_yhat, 2),
        "sentiment":     round(latest_sent, 4),
        "signal":        pred_label,
        "probabilities": proba_dict,
        "uncertainty":  uncertainty_summary,
        "shap_values":   shap_dict,
        "cv_weighted_f1": cv_f1,
        "news":          articles[:20],
        "attention_maps": attention_maps,
        "attention_keywords": attention_keywords,
        "counterfactual_tests": counterfactual_tests,
        "prophet_sensitivity": prophet_sensitivity,
        "rolling_backtest": rolling_backtest,
    }

    # Store in cache
    _cache[symbol] = {"result": result, "expires_at": time.time() + CACHE_TTL}
    print(f"[predictor] {symbol}: DONE — signal={pred_label}, F1={cv_f1}")
    return result
