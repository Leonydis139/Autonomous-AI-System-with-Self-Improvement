import numpy as np
import pandas as pd
import datetime
from typing import Dict, List, Optional
import nltk
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import logging

class EnhancedAnalyticsEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.sia = SentimentIntensityAnalyzer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

    def analyze_stock_trends(self, symbol: str, period: str = "1mo") -> Dict:
        from live_data_provider import EnhancedLiveDataProvider
        data_provider = EnhancedLiveDataProvider()
        try:
            data = data_provider.get_stock_data(symbol, period)
            if data.empty:
                return {"error": "No data available"}
            data['MA_5'] = data['Close'].rolling(window=5).mean()
            data['MA_20'] = data['Close'].rolling(window=20).mean()
            data['RSI'] = self.calculate_rsi(data['Close'])
            data['Volatility'] = data['Close'].rolling(window=20).std()
            recent_price = data['Close'].iloc[-1]
            ma_5 = data['MA_5'].iloc[-1]
            ma_20 = data['MA_20'].iloc[-1]
            trend = "Bullish" if recent_price > ma_5 > ma_20 else "Bearish"
            prediction = self.predict_stock_price(data)
            return {
                "symbol": symbol,
                "current_price": recent_price,
                "trend": trend,
                "rsi": data['RSI'].iloc[-1],
                "volatility": data['Volatility'].iloc[-1],
                "prediction": prediction,
                "data": data
            }
        except Exception as e:
            return {"error": str(e)}

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def predict_stock_price(self, data: pd.DataFrame) -> Dict:
        features = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_20', 'RSI']
        X = data[features].dropna()
        y = data['Close'].iloc[len(data) - len(X):]
        if len(X) < 10:
            return {"error": "Insufficient data for prediction"}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        next_price = model.predict([X.iloc[-1]])[0]
        return {
            "next_price": next_price,
            "mse": mse,
            "r2_score": r2,
            "confidence": "High" if r2 > 0.7 else "Medium" if r2 > 0.4 else "Low"
        }

    def analyze_sentiment(self, texts: List[str]) -> Dict:
        if not texts:
            return {"error": "No text data provided"}
        sentiments = []
        for text in texts:
            try:
                blob = TextBlob(text)
                polarity_tb = blob.sentiment.polarity
                vader_scores = self.sia.polarity_scores(text)
                polarity_vader = vader_scores['compound']
                combined_polarity = (polarity_tb + polarity_vader) / 2.0
                sentiment_label = "Positive" if combined_polarity > 0.1 else "Negative" if combined_polarity < -0.1 else "Neutral"
                sentiments.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'polarity': combined_polarity,
                    'sentiment': sentiment_label
                })
            except Exception:
                continue
        if not sentiments:
            return {"error": "No sentiment data generated"}
        avg_polarity = np.mean([s['polarity'] for s in sentiments])
        sentiment_counts = {
            "Positive": sum(1 for s in sentiments if s['sentiment'] == "Positive"),
            "Neutral": sum(1 for s in sentiments if s['sentiment'] == "Neutral"),
            "Negative": sum(1 for s in sentiments if s['sentiment'] == "Negative")
        }
        overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        return {
            "overall_sentiment": overall_sentiment,
            "average_polarity": avg_polarity,
            "sentiment_distribution": sentiment_counts,
            "individual_sentiments": sentiments
        }

    def generate_word_cloud(self, texts: List[str]) -> Image:
        try:
            full_text = " ".join(texts)
            wordcloud = WordCloud(
                width=800, height=400, background_color='white', stopwords=self.stopwords
            ).generate(full_text)
            return wordcloud.to_image()
        except Exception:
            return Image.new('RGB', (800, 400), color='white')

    def cluster_news(self, news_data: List[Dict]) -> List[Dict]:
        if not news_data:
            return news_data
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        texts = [f"{item['title']} {item['summary']}" for item in news_data]
        X = vectorizer.fit_transform(texts)
        n_clusters = min(5, len(news_data))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        for i, item in enumerate(news_data):
            item['cluster'] = int(kmeans.labels_[i])
        return news_data
