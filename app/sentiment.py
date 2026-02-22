import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
from tqdm import tqdm

def get_sentiment_scores_batch(texts, tokenizer, model, batch_size=32):
    scores = []
    model.eval()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Calculating Sentiment"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # For yiyanghkust/finbert-tone:
        # 0: Neutral, 1: Positive, 2: Negative
        # Score = Positive probability - Negative probability
        pos_probs = probs[:, 1].tolist()
        neg_probs = probs[:, 2].tolist()
        
        batch_scores = [p - n for p, n in zip(pos_probs, neg_probs)]
        scores.extend(batch_scores)
        
    return scores

def run_sentiment_analysis():
    print("Loading FinBERT model...")
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

    print("Loading news dataset...")
    # Load JSON dataset
    news_df = pd.read_json("data/IN-FINews Dataset.json")

    # Keep only Date and Title
    news_df = news_df[["Date", "Title"]]

    # Rename for consistency
    news_df = news_df.rename(columns={
        "Date": "ds",
        "Title": "News"
    })

    # Convert to datetime
    news_df["ds"] = pd.to_datetime(news_df["ds"])

    # Drop empty headlines
    news_df = news_df.dropna(subset=["News"])
    news_df = news_df[news_df["News"].str.strip() != ""]
    
    # For demonstration purposes, we can limit the dataset size if it's too large
    # news_df = news_df.head(1000)

    print("Calculating sentiment scores (this may take a while)...")
    texts = news_df["News"].tolist()
    news_df["sentiment"] = get_sentiment_scores_batch(texts, tokenizer, model)

    daily_sentiment = (
        news_df
        .groupby("ds")["sentiment"]
        .mean()
        .reset_index()
    )

    os.makedirs("results", exist_ok=True)
    daily_sentiment.to_csv("results/daily_sentiment.csv", index=False)
    print("Daily sentiment saved to results/daily_sentiment.csv")
    
    return daily_sentiment

if __name__ == "__main__":
    run_sentiment_analysis()
