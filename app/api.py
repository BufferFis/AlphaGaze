"""
api.py
------
FastAPI backend for the NIFTY/stock prediction system.

Endpoints
---------
GET  /                      → serves frontend/index.html
GET  /api/stocks            → list of supported stocks
GET  /api/predict/{symbol}  → run full pipeline, return prediction
GET  /api/news/{symbol}     → scrape latest headlines only (fast)

Run:
    uvicorn app.api:app --reload --port 8000
  or from project root:
    source .venv/bin/activate && uvicorn app.api:app --reload --port 8000
"""

import sys
import os

# Allow imports from app/ when running as `uvicorn app.api:app`
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from predictor import predict, SUPPORTED_STOCKS
from scraper import scrape_news

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Stock Predictor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serve the single-page frontend."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/api/stocks")
def get_stocks():
    """Return list of supported stocks."""
    return [
        {"symbol": sym, "name": name}
        for sym, name in SUPPORTED_STOCKS.items()
    ]


@app.get("/api/predict/{symbol:path}")
def get_prediction(symbol: str):
    """
    Run the full Prophet + FinBERT + RandomForest + SHAP pipeline
    for the given stock symbol.  Results are cached for 1 hour.
    """
    if symbol not in SUPPORTED_STOCKS:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not supported.")
    try:
        result = predict(symbol)
        return JSONResponse(content=result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")


@app.get("/api/news/{symbol:path}")
def get_news(symbol: str, max_articles: int = 30):
    """Scrape and return latest news headlines for a stock (no ML, fast)."""
    if symbol not in SUPPORTED_STOCKS:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not supported.")
    articles = scrape_news(symbol, max_articles=max_articles)
    return {"symbol": symbol, "articles": articles}
