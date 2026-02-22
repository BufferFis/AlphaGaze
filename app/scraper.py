"""
scraper.py
----------
Fetches real-time financial news headlines for any Indian stock using
Google News RSS — no API key, no rate limits, works for any query.

Returns a list of {title, date, url} dicts ready for FinBERT.
"""

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape

import feedparser
import requests

# How Google encodes the actual article URL inside its redirect link
_REDIRECT_RE = re.compile(r"url=([^&]+)")

# Mapping of yfinance symbol → human query string for better news results
STOCK_QUERIES = {
    "^NSEI":          "NIFTY 50 India stock market",
    "^BSESN":         "Sensex BSE India stock market",
    "RELIANCE.NS":    "Reliance Industries India stock",
    "TCS.NS":         "TCS Tata Consultancy Services India stock",
    "HDFCBANK.NS":    "HDFC Bank India stock",
    "INFY.NS":        "Infosys India stock",
    "ICICIBANK.NS":   "ICICI Bank India stock",
    "WIPRO.NS":       "Wipro India stock",
    "SBIN.NS":        "SBI State Bank of India stock",
    "BHARTIARTL.NS":  "Bharti Airtel India stock",
}


def _clean_title(raw: str) -> str:
    """Strip HTML tags and unescape entities from RSS title."""
    raw = re.sub(r"<[^>]+>", "", raw)
    return unescape(raw).strip()


def _extract_date(entry) -> str:
    """Return 'YYYY-MM-DD' from an RSS entry, fall back to today."""
    for field in ("published", "updated"):
        val = getattr(entry, field, None)
        if val:
            try:
                return parsedate_to_datetime(val).strftime("%Y-%m-%d")
            except Exception:
                pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def scrape_news(symbol: str, max_articles: int = 100) -> list[dict]:
    """
    Scrape recent news headlines for *symbol* via Google News RSS.

    Parameters
    ----------
    symbol       : yfinance ticker, e.g. 'RELIANCE.NS'
    max_articles : cap on returned articles

    Returns
    -------
    list of {"title": str, "date": "YYYY-MM-DD", "url": str}
    sorted newest-first.
    """
    query = STOCK_QUERIES.get(symbol, f"{symbol} India stock finance")

    rss_url = (
        "https://news.google.com/rss/search"
        f"?q={requests.utils.quote(query)}"
        "&hl=en-IN&gl=IN&ceid=IN:en"
    )

    try:
        feed = feedparser.parse(rss_url)
    except Exception as exc:
        print(f"[scraper] feedparser error for {symbol}: {exc}")
        return []

    results: list[dict] = []
    seen: set[str] = set()

    for entry in feed.entries[:max_articles]:
        title = _clean_title(getattr(entry, "title", "") or "")
        if not title or title in seen:
            continue
        seen.add(title)

        # Google News wraps the real URL in a redirect; try to unwrap it
        raw_link = getattr(entry, "link", "") or ""
        match = _REDIRECT_RE.search(raw_link)
        url = requests.utils.unquote(match.group(1)) if match else raw_link

        results.append({
            "title": title,
            "date":  _extract_date(entry),
            "url":   url,
        })

    # Sort newest first
    results.sort(key=lambda r: r["date"], reverse=True)
    print(f"[scraper] {symbol}: fetched {len(results)} articles")
    return results
