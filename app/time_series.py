import yfinance as yf
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_prophet():
    print("Downloading NIFTY 50 index data...")
    # Download NIFTY 50 index data — extend to 2025 to overlap with news dataset
    df = yf.download("^NSEI", start="2018-01-01", end="2025-09-01")

    # Reset index and inspect
    df = df.reset_index()
    
    prophet_df = df[["Date", "Close"]].copy()
    prophet_df.columns = ["ds", "y"]

    # Ensure correct data types
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    prophet_df["y"] = prophet_df["y"].astype(float)

    # Initialize Prophet model
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    # Fit model
    print("Fitting Prophet model...")
    model.fit(prophet_df)

    # Forecast next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # Save forecast to results folder
    os.makedirs("results", exist_ok=True)
    forecast.to_csv("results/prophet_forecast.csv", index=False)
    print("Prophet forecast saved to results/prophet_forecast.csv")
    
    return forecast

if __name__ == "__main__":
    run_prophet()
