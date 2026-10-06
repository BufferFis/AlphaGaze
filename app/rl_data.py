"""
rl_data.py
----------
Offline data preparation and PPO pre-training on maximum available yfinance history.

Usage:
    python app/rl_data.py --pretrain
    python app/rl_data.py --pretrain --timesteps 100000
    python app/rl_data.py --pretrain --force-download

This will:
    1. Download max-history price data for all 10 supported stocks via yfinance
    2. Fit Prophet on each stock's full history to compute prophet_gap + forecast_band
    3. Build 7-feature dataframes (sentiment = 0.0 for historical data)
    4. Chain-train PPO across all stocks sequentially (transfer learning)
    5. Save pre-trained model to data/ppo_pretrained.zip
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet

warnings.filterwarnings("ignore")

# All supported stocks (from predictor.py)
ALL_SYMBOLS = [
    "^NSEI", "^BSESN", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS",
    "INFY.NS", "ICICIBANK.NS", "WIPRO.NS", "SBIN.NS", "BHARTIARTL.NS",
]

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "prices")
PRETRAINED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ppo_pretrained")

FEATURE_COLS = [
    "prophet_gap", "forecast_band", "sentiment_1d",
    "sentiment_3d_ma", "sentiment_5d_ma",
    "price_momentum_5d", "volatility_5d",
]


def _safe_symbol_filename(symbol: str) -> str:
    """Convert symbol to safe filename: ^NSEI -> NSEI, RELIANCE.NS -> RELIANCE_NS"""
    return symbol.replace("^", "").replace(".", "_")


def download_and_cache(symbol: str, force: bool = False) -> pd.DataFrame:
    """
    Download max-period price history via yfinance and cache to CSV.

    Returns DataFrame with columns: [ds, y, next_return]
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{_safe_symbol_filename(symbol)}_max.csv")

    if os.path.exists(cache_path) and not force:
        print(f"[rl_data] {symbol}: Loading from cache {cache_path}")
        df = pd.read_csv(cache_path)
        df["ds"] = pd.to_datetime(df["ds"])
        return df

    print(f"[rl_data] {symbol}: Downloading max history from yfinance...")
    raw = yf.download(symbol, period="max", auto_adjust=True, progress=False)
    if raw.empty:
        print(f"[rl_data] WARNING: No data for {symbol}")
        return pd.DataFrame()

    raw = raw.reset_index()
    # Handle multi-level columns if returned by yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    df = raw[["Date", "Close"]].copy()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df["y"] = df["y"].astype(float)
    df = df.sort_values("ds").reset_index(drop=True)
    df["next_return"] = df["y"].shift(-1) / df["y"] - 1

    df.to_csv(cache_path, index=False)
    print(f"[rl_data] {symbol}: {len(df)} days cached to {cache_path}")
    return df


def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for historical pre-training.
    Uses price-only features; sentiment columns are set to 0.0 (neutral).
    """
    df = df.sort_values("ds").reset_index(drop=True).copy()
    df = df.dropna(subset=["y"])

    if len(df) < 30:
        return pd.DataFrame()

    # Fit Prophet on full history
    prophet_df = df[["ds", "y"]].copy()
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(prophet_df)
    forecast = m.predict(prophet_df[["ds"]])[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    merged = pd.merge(df, forecast, on="ds", how="inner")

    # Price-based features
    merged["daily_ret"]         = merged["y"].pct_change()
    merged["prophet_gap"]       = (merged["yhat"] - merged["y"]) / merged["y"]
    merged["forecast_band"]     = (merged["yhat_upper"] - merged["yhat_lower"]) / merged["y"]
    merged["price_momentum_5d"] = merged["y"].pct_change(5)
    merged["volatility_5d"]     = merged["daily_ret"].rolling(5, min_periods=2).std()

    # Sentiment = 0.0 (neutral placeholder for historical data without news)
    merged["sentiment_1d"]    = 0.0
    merged["sentiment_3d_ma"] = 0.0
    merged["sentiment_5d_ma"] = 0.0

    # Drop NaN rows and last row (no next_return)
    merged = merged.dropna(subset=FEATURE_COLS + ["next_return"])
    merged = merged.iloc[:-1]

    return merged.reset_index(drop=True)


def pretrain_on_all_stocks(total_timesteps_per_stock: int = 50_000,
                            force_download: bool = False) -> None:
    """
    Download all stocks, build price features, chain-train PPO on all of them.

    Chained training = sequential transfer learning:
        Train on stock 1 -> [checkpoint A]
        Train on stock 2 starting from A -> [checkpoint B]
        ... etc. Each stock starts where the last left off.

    Saves final model to data/ppo_pretrained.zip
    """
    from rl_agent import train_agent, save_agent

    print(f"\n{'='*60}")
    print("  AlphaGaze PPO Pre-training on Maximum Historical Data")
    print(f"{'='*60}")

    all_dfs = []
    for symbol in ALL_SYMBOLS:
        try:
            raw_df = download_and_cache(symbol, force=force_download)
            if raw_df.empty:
                continue
            feature_df = build_price_features(raw_df)
            if len(feature_df) < 100:
                print(f"[rl_data] {symbol}: Not enough data ({len(feature_df)} rows), skipping")
                continue
            feature_df["symbol"] = symbol
            all_dfs.append(feature_df)
            print(f"[rl_data] {symbol}: {len(feature_df)} usable training rows")
        except Exception as e:
            print(f"[rl_data] {symbol}: Failed ({e}), skipping")

    if not all_dfs:
        print("[rl_data] ERROR: No data collected. Cannot pre-train.")
        return

    total_rows = sum(len(d) for d in all_dfs)
    print(f"\n[rl_data] Total training data: {total_rows:,} rows across {len(all_dfs)} stocks")
    print(f"[rl_data] Training {total_timesteps_per_stock:,} timesteps per stock...")

    # Chain-train: each stock starts from the previous stock's model
    model = None
    for i, df in enumerate(all_dfs):
        symbol = df["symbol"].iloc[0]
        print(f"\n[rl_data] [{i+1}/{len(all_dfs)}] Training on {symbol} ({len(df)} days)...")
        model = train_agent(df, total_timesteps=total_timesteps_per_stock,
                            seed=42, pretrained_model=model)

    save_agent(model, PRETRAINED_MODEL_PATH)
    total_steps = total_timesteps_per_stock * len(all_dfs)
    print(f"\n{'='*60}")
    print(f"  Pre-training complete!")
    print(f"  Model:      {PRETRAINED_MODEL_PATH}.zip")
    print(f"  Stocks:     {len(all_dfs)}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Total steps:{total_steps:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaGaze RL offline pre-training")
    parser.add_argument("--pretrain", action="store_true",
                        help="Run offline pre-training on all stocks")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download price data even if cached")
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="PPO timesteps per stock (default: 50,000)")
    args = parser.parse_args()

    if args.pretrain:
        pretrain_on_all_stocks(
            total_timesteps_per_stock=args.timesteps,
            force_download=args.force_download,
        )
    else:
        print("Usage:")
        print("  python app/rl_data.py --pretrain")
        print("  python app/rl_data.py --pretrain --timesteps 100000")
        print("  python app/rl_data.py --pretrain --force-download")
