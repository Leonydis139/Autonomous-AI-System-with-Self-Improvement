import streamlit as st
import json
import re
import requests
import os
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
import hashlib
import threading
import logging
import sqlite3
from typing import Dict, Optional, List
import warnings
import random
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
import feedparser
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from wordcloud import WordCloud
import nltk
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import seaborn as sns
from PIL import Image
import io
import base64
import pytz
from dateutil import parser
from bs4 import BeautifulSoup
import json
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

# Load a pre-trained model from the Hugging Face model hub
model = HuggingFaceModel(
    hf_model_id="bert-base-uncased",
    role="your-iam-role",
    transformers_version="4.12.5",
    pytorch_version="1.9.0"
)

# Get the Docker image URI for the LLM
image_uri = get_huggingface_llm_image_uri(
    "bert-base-uncased",
    pytorch_version="1.9.0",
    transformers_version="4.12.5"
)
# Optional imports for features; handle ImportError where used
try:
    from sklearn.ensemble import RandomForestRegressor as _RandomForestRegressor
    from sklearn.metrics import mean_squared_error as _mean_squared_error, r2_score as _r2_score
    from sklearn.model_selection import train_test_split as _train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    _RandomForestRegressor = None
    _mean_squared_error = None
    _r2_score = None
    _train_test_split = None
	
warnings.filterwarnings('ignore')

# ===================== ENHANCED SYSTEM CONFIGURATION =====================
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
MAX_RESEARCH_RESULTS = 10
CODE_EXECUTION_TIMEOUT = 30
SAFE_MODE = True
PERSONAS = ["Researcher", "Teacher", "Analyst", "Engineer", "Scientist", "Assistant", "Consultant", "Creative", "Problem Solver"]
SESSION_FILE = "session_state.json"
USER_DB = "users.db"
TEAM_DB = "teams.json"
WORKFLOW_DB = "workflows.json"
CACHE_DB = "cache.db"
TIMEZONES = pytz.all_timezones

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO if DEBUG_MODE else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== ENHANCED DATABASE MANAGER =====================
class EnhancedDatabaseManager:
    def __init__(self):
        self.pg_pool = None
        self.sqlite_lock = threading.Lock()
        self.init_databases()
        self.cache = {}
        self.cache_lock = threading.Lock()

    def get_connection(self):
        """Get a database connection (PostgreSQL if available, else SQLite)"""
        if self.pg_pool:
            try:
                return self.pg_pool.getconn(), "postgresql"
            except Exception as e:
                logger.warning(f"PostgreSQL connection failed: {e}")
        return sqlite3.connect(CACHE_DB), "sqlite"

    def init_databases(self):
        """Initialize both SQLite and PostgreSQL databases"""
        try:
            self.init_sqlite()
            self.init_postgresql()
        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    def init_sqlite(self):
        """Initialize SQLite database with proper error handling"""
        try:
            if not os.path.exists(CACHE_DB):
                open(CACHE_DB, 'a').close()
            conn = sqlite3.connect(CACHE_DB)
            cursor = conn.cursor()
            cursor.executescript('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    preferences TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    action TEXT,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')
            conn.commit()
            conn.close()
            logger.info("SQLite database initialized successfully")
        except Exception as e:
            logger.error(f"SQLite initialization error: {e}")

    def init_postgresql(self):
        """Initialize PostgreSQL connection if available"""
        database_url = os.environ.get('DATABASE_URL')
        if database_url:
            try:
                from psycopg2 import pool
                self.pg_pool = pool.SimpleConnectionPool(1, 10, database_url)
                logger.info("PostgreSQL connection pool initialized")
            except ImportError:
                logger.warning("psycopg2 not installed, skipping PostgreSQL initialization")

    def return_connection(self, conn, db_type):
        """Return connection to pool"""
        try:
            if db_type == "postgresql" and self.pg_pool:
                self.pg_pool.putconn(conn)
            else:
                conn.close()
        except Exception as e:
            logger.error(f"Error returning connection: {e}")

    def get_cached_result(self, key: str) -> Optional[str]:
        """Get cached result with enhanced error handling"""
        conn, db_type = None, None
        try:
            conn, db_type = self.get_connection()
            cursor = conn.cursor()
            if db_type == "postgresql":
                cursor.execute('''
                    SELECT value FROM cache 
                    WHERE key = %s AND expires_at > NOW()
                ''', (key,))
            else:
                cursor.execute('''
                    SELECT value FROM cache 
                    WHERE key = ? AND expires_at > datetime('now')
                ''', (key,))
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None
        finally:
            if conn:
                self.return_connection(conn, db_type)

    def set_cached_result(self, key: str, value: str, ttl_minutes: int = 60):
        """Set cached result with enhanced error handling"""
        conn, db_type = None, None
        try:
            conn, db_type = self.get_connection()
            cursor = conn.cursor()
            expires_at = datetime.datetime.now() + timedelta(minutes=ttl_minutes)
            if db_type == "postgresql":
                cursor.execute('''
                    INSERT INTO cache (key, value, expires_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (key) DO UPDATE SET 
                    value = EXCLUDED.value, 
                    expires_at = EXCLUDED.expires_at
                ''', (key, value, expires_at))
            else:
                cursor.execute('''
                    INSERT OR REPLACE INTO cache (key, value, expires_at)
                    VALUES (?, ?, ?)
                ''', (key, value, expires_at))
            conn.commit()
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
        finally:
            if conn:
                self.return_connection(conn, db_type)

    def log_analytics(self, user_id: str, action: str, details: str = ""):
        """Log analytics data"""
        conn, db_type = None, None
        try:
            conn, db_type = self.get_connection()
            cursor = conn.cursor()

            if db_type == "postgresql":
                cursor.execute('''
                    INSERT INTO analytics (user_id, action, details)
                    VALUES (%s, %s, %s)
                ''', (user_id, action, details))
            else:
                cursor.execute('''
                    INSERT INTO analytics (user_id, action, details)
                    VALUES (?, ?, ?)
                ''', (user_id, action, details))

            conn.commit()

        except Exception as e:
            logger.error(f"Analytics logging error: {e}")
        finally:
            if conn:
                self.return_connection(conn, db_type)

# ===================== ENHANCED LIVE DATA PROVIDERS =====================
class EnhancedLiveDataProvider:
    """Enhanced live data provider with caching and error handling"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
        self.logger = logging.getLogger(__name__)
        self.news_sources = {
            "technology": [
                "https://feeds.feedburner.com/oreilly/radar",
                "https://techcrunch.com/feed/",
                "https://news.ycombinator.com/rss"
            ],
            "business": [
                "https://www.bloomberg.com/feed/podcasts/etf-report.rss",
                "https://www.cnbc.com/id/100003114/device/rss/rss.html"
            ],
            "general": [
                "https://rss.cnn.com/rss/cnn_topstories.rss",
                "https://feeds.bbci.co.uk/news/rss.xml"
            ],
            "science": [
                "https://www.scientificamerican.com/rss/?section=all",
                "https://www.nasa.gov/rss/dyn/breaking_news.rss"
            ]
        }
        self.crypto_ids = {
            "bitcoin": "bitcoin",
            "ethereum": "ethereum",
            "cardano": "cardano",
            "solana": "solana",
            "dogecoin": "dogecoin"
        }

    @st.cache_data(ttl=300, show_spinner="Fetching stock data...")
    def get_stock_data(symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Fetch stock data with proper caching"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data.reset_index() if not data.empty else pd.DataFrame()
        except Exception as e:
            logging.error(f"Stock data error: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=300, show_spinner="Fetching crypto data...")
    def get_crypto_data(coin_name: str = "bitcoin") -> Dict[str, float]:
        """
        Fetch cryptocurrency data with cache-safe parameters
        Args:
            coin_name: Common name of cryptocurrency
        Returns:
            Dictionary with price data
        """
        try:
            coin_id = self.crypto_ids.get(coin_name.lower(), coin_name.lower())
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if coin_id not in data:
                return {}
                
            return {
                'price': data[coin_id]['usd'],
                'change_24h': data[coin_id]['usd_24h_change'],
                'last_updated': datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Crypto data error: {e}")
            return {}

    def _get_news_from_source(self, source_url: str, max_items: int = 5) -> List[Dict]:
        """Parse news from a single RSS source"""
        try:
            feed = feedparser.parse(source_url)
            articles = []
            for entry in feed.entries[:max_items]:
                try:
                    published = parser.parse(entry.published) if hasattr(entry, 'published') else datetime.datetime.now()
                    
                    # Extract clean text from summary
                    summary = entry.get('summary', '')
                    if summary:
                        soup = BeautifulSoup(summary, 'html.parser')
                        summary = soup.get_text()
                    
                    articles.append({
                        'title': entry.get('title', 'No title'),
                        'link': entry.get('link', '#'),
                        'published': published,
                        'summary': summary[:200] + '...' if len(summary) > 200 else summary,
                        'source': feed.feed.get('title', source_url)
                    })
                except Exception as e:
                    self.logger.warning(f"Error parsing news entry: {e}")
            return articles
        except Exception as e:
            self.logger.warning(f"Error fetching from {source_url}: {e}")
            return []

    @st.cache_data(ttl=600, show_spinner="Fetching news...")
    def get_news_data(_self, topic: str = "technology", max_articles: int = 5) -> List[Dict]:
        """
        Fetch and cache news data with proper hashing
        """
        sources = _self.news_sources.get(topic.lower(), _self.news_sources["general"])
        all_articles = []
        
        for source in sources[:3]:
            articles = _self._get_news_from_source(source, max(3, max_articles//len(sources)))
            all_articles.extend(articles)
            
            if len(all_articles) >= max_articles:
                break
                
        # Sort by published date
        return sorted(
            all_articles,
            key=lambda x: x['published'],
            reverse=True
        )[:max_articles]

    @st.cache_data(ttl=1800, show_spinner="Checking weather...")
    def get_weather_data(self, location: str, api_key: Optional[str] = None) -> Dict:
        """Fetch weather data with location validation"""
        api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        if not api_key:
            return {"error": "API key not configured"}
            
        try:
            # Geocoding to get coordinates
            geo_url = "http://api.openweathermap.org/geo/1.0/direct"
            geo_params = {'q': location, 'limit': 1, 'appid': api_key}
            geo_response = self.session.get(geo_url, params=geo_params)
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            
            if not geo_data:
                return {"error": "Location not found"}
                
            lat, lon = geo_data[0]['lat'], geo_data[0]['lon']
            
            # Get weather data
            weather_url = "https://api.openweathermap.org/data/3.0/onecall"
            weather_params = {
                'lat': lat,
                'lon': lon,
                'exclude': 'minutely,hourly',
                'appid': api_key,
                'units': 'metric'
            }
            
            weather_response = self.session.get(weather_url, params=weather_params)
            weather_response.raise_for_status()
            
            data = weather_response.json()
            data['location'] = geo_data[0]
            return data
            
        except Exception as e:
            self.logger.error(f"Weather data error for {location}: {e}")
            return {"error": str(e)}

    @st.cache_data(ttl=86400, show_spinner="Fetching economic data...")
    def get_economic_indicators(self, api_key: Optional[str] = None) -> Dict:
        """Fetch economic indicators with better data structure"""
        api_key = api_key or os.getenv('FRED_API_KEY')
        if not api_key:
            return {"error": "API key not configured"}
            
        indicators = {
            'GDP': {'series_id': 'GDP', 'name': 'Gross Domestic Product'},
            'Unemployment': {'series_id': 'UNRATE', 'name': 'Unemployment Rate'},
            'Inflation': {'series_id': 'CPIAUCSL', 'name': 'Consumer Price Index'}
        }
        
        results = {}
        for key, config in indicators.items():
            try:
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': config['series_id'],
                    'api_key': api_key,
                    'file_type': 'json',
                    'limit': 12,
                    'sort_order': 'desc'
                }
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                results[key] = {
                    'metadata': config,
                    'data': data.get('observations', []),
                    'as_of': datetime.datetime.now().isoformat()
                }
            except Exception as e:
                self.logger.warning(f"Error fetching {key}: {e}")
                results[key] = {'error': str(e), 'metadata': config}
                
        return results

# ===================== ENHANCED UI COMPONENTS =====================
class EnhancedUIComponents:
    """Enhanced UI components with modern styling"""
    
    @staticmethod
    def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal"):
        """Create a modern metric card"""
        with st.container():
            st.metric(label=title, value=value, delta=delta, delta_color=delta_color)

    @staticmethod
    def create_info_card(title: str, content: str, icon: str = "ℹ️"):
        """Create an information card"""
        st.markdown(f"""
        <div style="
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin: 1rem 0;
        ">
            <h4 style="margin: 0 0 0.5rem 0; color: #1f77b4;">
                {icon} {title}
            </h4>
            <p style="margin: 0; color: #333;">
                {content}
            </p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def create_progress_bar(value: float, max_value: float, title: str):
        """Create a styled progress bar"""
        progress = min(value / max_value, 1.0)
        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-weight: bold;">{title}</span>
                <span>{value:.1f} / {max_value:.1f}</span>
            </div>
            <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px;">
                <div style="
                    background: linear-gradient(90deg, #1f77b4, #17a2b8);
                    width: {progress * 100}%;
                    height: 100%;
                    border-radius: 10px;
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    @staticmethod
    def create_sentiment_badge(sentiment: str):
        """Create a colored badge for sentiment"""
        color_map = {
            "Positive": "#2ca02c",
            "Negative": "#d62728",
            "Neutral": "#7f7f7f"
        }
        return st.markdown(
            f"<span style='background-color:{color_map.get(sentiment, 'gray')};"
            f"color:white;padding:0.2rem 0.5rem;border-radius:0.25rem;font-weight:bold;'>"
            f"{sentiment}</span>",
            unsafe_allow_html=True
        )
        
    @staticmethod
    def create_news_card(title: str, source: str, published: datetime, summary: str, link: str, sentiment: str = None):
        """Create a news card with sentiment badge"""
        time_ago = (datetime.datetime.now() - published).days
        time_text = f"{time_ago} days ago" if time_ago > 0 else "Today"
        
        badge = f"""
        <div style="margin-bottom: 0.5rem;">
            {EnhancedUIComponents.create_sentiment_badge(sentiment)._proxy}
        </div>
        """ if sentiment else ""
        
        st.markdown(f"""
        <div class="news-card" style="
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
            border-left: 4px solid #4CAF50;
            transition: all 0.3s ease;
        ">
            <h4 style="margin: 0 0 0.5rem 0;">{title}</h4>
            <div style="display: flex; justify-content: space-between; color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">
                <span>{source}</span>
                <span>{time_text}</span>
            </div>
            {badge}
            <p style="margin: 0.5rem 0 1rem 0;">{summary}</p>
            <a href="{link}" target="_blank" style="
                display: inline-block;
                background: #1f77b4;
                color: white;
                padding: 0.3rem 0.8rem;
                border-radius: 4px;
                text-decoration: none;
                font-weight: 500;
            ">Read Article</a>
        </div>
        """, unsafe_allow_html=True)

# ===================== ENHANCED ANALYTICS ENGINE =====================
class EnhancedAnalyticsEngine:
    """Enhanced analytics with ML capabilities"""
    
    def __init__(self):
        self.data_provider = EnhancedLiveDataProvider()
        self.scaler = StandardScaler() if 'StandardScaler' in globals() else None
        self.sia = SentimentIntensityAnalyzer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

    def analyze_stock_trends(self, symbol: str, period: str = "1mo") -> Dict:
        """Analyze stock trends with ML predictions"""
        try:
            data = self.data_provider.get_stock_data(symbol, period)
            if data.empty:
                return {"error": "No data available"}

            # Technical indicators
            data['MA_5'] = data['Close'].rolling(window=5).mean()
            data['MA_20'] = data['Close'].rolling(window=20).mean()
            data['RSI'] = self.calculate_rsi(data['Close'])
            data['Volatility'] = data['Close'].rolling(window=20).std()

            # Trend analysis
            recent_price = data['Close'].iloc[-1]
            ma_5 = data['MA_5'].iloc[-1]
            ma_20 = data['MA_20'].iloc[-1]
            
            trend = "Bullish" if recent_price > ma_5 > ma_20 else "Bearish"
            
            # ML prediction if available
            prediction = self.predict_stock_price(data) if _RandomForestRegressor else None

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
            logger.error(f"Stock analysis error: {e}")
            return {"error": str(e)}

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def predict_stock_price(self, data: pd.DataFrame) -> Dict:
        """Predict stock price using ML"""
        try:
            # Prepare features
            features = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_20', 'RSI']
            X = data[features].dropna()
            y = data['Close'].iloc[len(data) - len(X):]

            if len(X) < 10:
                return {"error": "Insufficient data for prediction"}

            # Train model
            X_train, X_test, y_train, y_test = _train_test_split(X, y, test_size=0.2, random_state=42)
            model = _RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Make prediction
            predictions = model.predict(X_test)
            
            # Calculate metrics
            mse = _mean_squared_error(y_test, predictions)
            r2 = _r2_score(y_test, predictions)

            # Predict next price
            next_price = model.predict([X.iloc[-1]])[0]

            return {
                "next_price": next_price,
                "mse": mse,
                "r2_score": r2,
                "confidence": "High" if r2 > 0.7 else "Medium" if r2 > 0.4 else "Low"
            }
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return {"error": str(e)}

    def analyze_sentiment(self, texts: List[str]) -> Dict:
        """Analyze sentiment of text data using both TextBlob and VADER"""
        try:
            if not texts:
                return {"error": "No text data provided"}

            sentiments = []
            for text in texts:
                try:
                    # TextBlob sentiment
                    blob = TextBlob(text)
                    polarity_tb = blob.sentiment.polarity
                    
                    # VADER sentiment
                    vader_scores = self.sia.polarity_scores(text)
                    polarity_vader = vader_scores['compound']
                    
                    # Combine both scores
                    combined_polarity = (polarity_tb + polarity_vader) / 2.0
                    
                    # Determine sentiment label
                    sentiment_label = "Positive" if combined_polarity > 0.1 else "Negative" if combined_polarity < -0.1 else "Neutral"
                    
                    sentiments.append({
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'polarity': combined_polarity,
                        'sentiment': sentiment_label
                    })
                except Exception as e:
                    logger.warning(f"Sentiment analysis error for text: {e}")

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
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"error": str(e)}
    
    def generate_word_cloud(self, texts: List[str]) -> Image:
        """Generate word cloud from text data"""
        try:
            # Combine all texts
            full_text = " ".join(texts)
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                stopwords=self.stopwords
            ).generate(full_text)
            
            return wordcloud.to_image()
        except Exception as e:
            logger.error(f"Word cloud generation error: {e}")
            return Image.new('RGB', (800, 400), color='white')
    
    def cluster_news(self, news_data: List[Dict]) -> List[Dict]:
        """Cluster news articles using K-Means"""
        try:
            if not news_data:
                return news_data
                
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            texts = [f"{item['title']} {item['summary']}" for item in news_data]
            X = vectorizer.fit_transform(texts)
            
            # Cluster using K-Means
            n_clusters = min(5, len(news_data))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X)
            
            # Add cluster labels to news data
            for i, item in enumerate(news_data):
                item['cluster'] = int(kmeans.labels_[i])
                
            return news_data
        except Exception as e:
            logger.error(f"News clustering error: {e}")
            return news_data

# ===================== USER PREFERENCES =====================
class UserPreferences:
    """Manage user preferences and settings"""
    
    def __init__(self):
        self.preferences = {
            "theme": "light",
            "timezone": "UTC",
            "default_stock": "AAPL",
            "default_crypto": "bitcoin",
            "default_news_category": "technology",
            "refresh_interval": 5
        }
        
    def load_preferences(self, user_id: str):
        """Load user preferences from session or database"""
        if 'preferences' in st.session_state:
            self.preferences = st.session_state.preferences
        else:
            # In a real app, load from database
            st.session_state.preferences = self.preferences
            
    def save_preferences(self, user_id: str):
        """Save user preferences to database"""
        # In a real app, save to database
        st.session_state.preferences = self.preferences
        
    def preference_editor(self):
        """UI for editing user preferences"""
        with st.expander("⚙️ User Preferences"):
            self.preferences['theme'] = st.selectbox("Theme", ["light", "dark", "blue"], index=0)
            self.preferences['timezone'] = st.selectbox("Timezone", TIMEZONES, index=TIMEZONES.index("UTC"))
            self.preferences['default_stock'] = st.text_input("Default Stock", "AAPL")
            self.preferences['default_crypto'] = st.selectbox("Default Crypto", list(self.data_provider.crypto_ids.keys()), index=0)
            self.preferences['default_news_category'] = st.selectbox("Default News Category", ["technology", "business", "science"], index=0)
            self.preferences['refresh_interval'] = st.slider("Refresh Interval (min)", 1, 60, 5)
            
            if st.button("Save Preferences"):
                self.save_preferences("current_user")
                st.success("Preferences saved!")

# ===================== MAIN APPLICATION =====================
def main():
    """Main application with enhanced UI"""
    st.set_page_config(
        page_title="Enhanced Data Analytics Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize components
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = EnhancedDatabaseManager()
    if 'data_provider' not in st.session_state:
        st.session_state.data_provider = EnhancedLiveDataProvider()
    if 'analytics_engine' not in st.session_state:
        st.session_state.analytics_engine = EnhancedAnalyticsEngine()
    if 'ui_components' not in st.session_state:
        st.session_state.ui_components = EnhancedUIComponents()
    if 'preferences' not in st.session_state:
        st.session_state.preferences = UserPreferences()
        st.session_state.preferences.load_preferences("current_user")

    # Custom CSS for styling
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #1f77b4, #17a2b8);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            text-align: center;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-card {
            background: white;
            padding: 1.2rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin: 0.8rem 0;
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .sidebar-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1.2rem;
            border-left: 4px solid #1f77b4;
        }
        .tab-content {
            animation: fadeIn 0.5s;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            border-radius: 8px 8px 0 0 !important;
            background-color: #f0f2f6 !important;
            transition: all 0.3s !important;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4 !important;
            color: white !important;
        }
        footer {
            text-align: center;
            padding: 1rem;
            color: #6c757d;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Page header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0;">📊 Enhanced Data Analytics Platform</h1>
        <p style="margin:0; opacity:0.9;">Real-time insights powered by AI • Free Forever</p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h3>🎛️ Control Panel</h3>
        </div>
        """, unsafe_allow_html=True)

        # User persona selection
        persona = st.selectbox(
            "Select Your Role",
            PERSONAS,
            index=0,
            help="Choose your role for personalized experience"
        )

        # Data source selection
        st.markdown("""
        <div class="sidebar-section">
            <h4>📡 Data Sources</h4>
        </div>
        """, unsafe_allow_html=True)
        data_sources = st.multiselect(
            "Select Data Sources",
            ["Stock Market", "Cryptocurrency", "News", "Weather", "Economic Indicators"],
            default=["Stock Market", "News"],
            label_visibility="collapsed"
        )

        # Analysis settings
        st.markdown("""
        <div class="sidebar-section">
            <h4>⚙️ Analysis Settings</h4>
        </div>
        """, unsafe_allow_html=True)
        enable_ml = st.checkbox("Enable ML Predictions", value=True)
        enable_sentiment = st.checkbox("Enable Sentiment Analysis", value=True)
        enable_clustering = st.checkbox("Enable News Clustering", value=True)
        
        # Personalization
        st.markdown("""
        <div class="sidebar-section">
            <h4>🎨 Personalization</h4>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.preferences.preference_editor()

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <p>Built with ❤️ using Streamlit</p>
            <p>Version 3.0 • Free Edition</p>
            <p style="font-size: 0.8rem; margin-top: 1rem;">© 2023 Enhanced Analytics Platform</p>
        </div>
        """, unsafe_allow_html=True)

    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Market Dashboard",
        "📰 News & Sentiment",
        "🔍 Custom Analysis",
        "⚙️ User Preferences"
    ])

    with tab1:
        st.header("📈 Live Market Dashboard")
        
        # Stock analysis section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            stock_symbol = st.text_input("Stock Symbol", 
                                        value=st.session_state.preferences.preferences['default_stock'], 
                                        help="Enter stock symbol (e.g., AAPL, GOOGL)")
            period = st.selectbox("Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2)
            
            if st.button("🔍 Analyze Stock", type="primary", use_container_width=True):
                with st.spinner("Analyzing stock data..."):
                    analysis = st.session_state.analytics_engine.analyze_stock_trends(stock_symbol, period)
                    
                    if "error" not in analysis:
                        # Display metrics
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Current Price", f"${analysis['current_price']:.2f}")
                        with col_b:
                            st.metric("Trend", analysis['trend'])
                        with col_c:
                            st.metric("RSI", f"{analysis['rsi']:.2f}")
                        with col_d:
                            st.metric("Volatility", f"{analysis['volatility']:.2f}")
                        
                        # Create interactive chart
                        data = analysis['data']
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=[f"{stock_symbol} Price Chart", "Volume"],
                            vertical_spacing=0.1,
                            shared_xaxes=True
                        )
                        
                        # Price chart with moving averages
                        fig.add_trace(
                            go.Candlestick(
                                x=data['Date'],
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name="Price"
                            ),
                            row=1, col=1
                        )
                        
                        if 'MA_5' in data.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=data['Date'],
                                    y=data['MA_5'],
                                    mode='lines',
                                    name='MA 5',
                                    line=dict(color='orange', width=2)
                                ),
                                row=1, col=1
                            )
                        
                        if 'MA_20' in data.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=data['Date'],
                                    y=data['MA_20'],
                                    mode='lines',
                                    name='MA 20',
                                    line=dict(color='red', width=2)
                                ),
                                row=1, col=1
                            )
                        
                        # Volume chart
                        fig.add_trace(
                            go.Bar(
                                x=data['Date'],
                                y=data['Volume'],
                                name="Volume",
                                marker_color='#1f77b4'
                            ),
                            row=2, col=1
                        )
                        
                        fig.update_layout(
                            title=f"{stock_symbol} Stock Analysis",
                            height=600,
                            showlegend=True,
                            xaxis_rangeslider_visible=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # ML Prediction section
                        if enable_ml and analysis.get('prediction'):
                            prediction = analysis['prediction']
                            if "error" not in prediction:
                                st.subheader("🤖 ML Prediction")
                                col_pred1, col_pred2, col_pred3 = st.columns(3)
                                
                                with col_pred1:
                                    st.metric("Predicted Price", f"${prediction['next_price']:.2f}")
                                with col_pred2:
                                    st.metric("R² Score", f"{prediction['r2_score']:.3f}")
                                with col_pred3:
                                    st.metric("Confidence", prediction['confidence'])
                                
                                if prediction['confidence'] != "High":
                                    st.warning("⚠️ Prediction confidence is not high. Use with caution.")
                    else:
                        st.error(f"Error: {analysis['error']}")
        
        with col2:
            st.subheader("💰 Cryptocurrency")
            crypto_coins = list(st.session_state.data_provider.crypto_ids.keys())
            
            for coin in crypto_coins[:5]:
                crypto_data = st.session_state.data_provider.get_crypto_data(coin)
                if crypto_data and 'price' in crypto_data:
                    price = crypto_data['price']
                    change = crypto_data.get('change_24h', 0)
                    
                    st.metric(
                        coin.capitalize(),
                        f"${price:,.2f}",
                        f"{change:.2f}%" if change else None,
                        delta_color="inverse"
                    )
                    
            st.markdown("---")
            st.subheader("📊 Economic Indicators")
            eco_data = st.session_state.data_provider.get_economic_indicators()
            for indicator, data in eco_data.items():
                if 'data' in data and len(data['data']) > 0:
                    latest = data['data'][0]
                    st.metric(
                        indicator,
                        latest.get('value', 'N/A'),
                        data['metadata'].get('unit', '')
                    )

    with tab2:
        st.header("📰 News & Sentiment Analysis")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # News categories
            news_topic = st.selectbox(
                "Select News Category",
                ["technology", "business", "science", "general"],
                index=0
            )
            
            max_articles = st.slider("Number of Articles", 5, 30, 10)
            
            if st.button("📰 Fetch Latest News", type="primary", use_container_width=True):
                with st.spinner("Fetching latest news..."):
                    news_data = st.session_state.data_provider.get_news_data(news_topic, max_articles)
                    
                    if news_data:
                        st.success(f"Found {len(news_data)} articles")
                        
                        # Enable clustering if requested
                        if enable_clustering:
                            with st.spinner("Clustering news articles..."):
                                news_data = st.session_state.analytics_engine.cluster_news(news_data)
                        
                        # Sentiment analysis
                        sentiment_results = None
                        if enable_sentiment:
                            with st.spinner("Analyzing sentiment..."):
                                news_texts = [article['title'] + ' ' + article.get('summary', '') for article in news_data]
                                sentiment_results = st.session_state.analytics_engine.analyze_sentiment(news_texts)
                        
                        # Display news cards
                        cluster_colors = {
                            0: "#1f77b4",
                            1: "#ff7f0e",
                            2: "#2ca02c",
                            3: "#d62728",
                            4: "#9467bd"
                        }
                        
                        for i, article in enumerate(news_data):
                            sentiment = sentiment_results['individual_sentiments'][i]['sentiment'] if sentiment_results else None
                            st.session_state.ui_components.create_news_card(
                                title=article['title'],
                                source=article['source'],
                                published=article['published'],
                                summary=article['summary'],
                                link=article['link'],
                                sentiment=sentiment
                            )
                            
                            # Show cluster badge if clustered
                            if 'cluster' in article:
                                cluster_id = article['cluster']
                                st.markdown(f"""
                                <div style="
                                    display: inline-block;
                                    padding: 0.25rem 0.5rem;
                                    border-radius: 12px;
                                    font-size: 0.8rem;
                                    font-weight: bold;
                                    margin-bottom: 1rem;
                                    background-color: {cluster_colors.get(cluster_id, '#999999')};
                                    color: white;
                                ">
                                    Cluster #{cluster_id + 1}
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("No news articles found for this category")
        
        with col2:
            st.subheader("😊 Sentiment Analysis")
            if 'sentiment_results' in locals() and sentiment_results:
                if "error" not in sentiment_results:
                    # Overall sentiment
                    st.metric("Overall Sentiment", sentiment_results['overall_sentiment'])
                    
                    # Sentiment distribution
                    st.markdown("**Sentiment Distribution**")
                    dist = sentiment_results['sentiment_distribution']
                    fig_dist = px.pie(
                        names=list(dist.keys()),
                        values=list(dist.values()),
                        color=list(dist.keys()),
                        color_discrete_map={
                            "Positive": "#2ca02c",
                            "Neutral": "#7f7f7f",
                            "Negative": "#d62728"
                        }
                    )
                    fig_dist.update_layout(showlegend=False)
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Word cloud
                    st.markdown("**Top Keywords**")
                    if news_data:
                        news_texts = [article['title'] + ' ' + article.get('summary', '') for article in news_data]
                        wordcloud_img = st.session_state.analytics_engine.generate_word_cloud(news_texts)
                        st.image(wordcloud_img, caption="Word Cloud of News Content")

    with tab3:
        st.header("🔍 Custom Analysis")
        st.subheader("Upload your own dataset for analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="Upload your dataset for custom analysis"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("Dataset loaded successfully!")
                
                # Show dataset preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Basic stats
                st.subheader("Basic Statistics")
                st.write(df.describe())
                
                # Column selector
                selected_columns = st.multiselect(
                    "Select columns for analysis",
                    df.columns
                )
                
                if selected_columns:
                    # Correlation analysis
                    st.subheader("Correlation Matrix")
                    corr = df[selected_columns].corr()
                    fig_corr = px.imshow(
                        corr,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        zmin=-1,
                        zmax=1
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Distribution plots
                    st.subheader("Distribution Plots")
                    for col in selected_columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            fig_dist = px.histogram(df, x=col, title=f"Distribution of {col}")
                            st.plotly_chart(fig_dist, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            st.info("Please upload a CSV file to get started")
            
    with tab4:
        st.header("⚙️ User Preferences")
        st.session_state.preferences.preference_editor()
        
        st.subheader("Account Settings")
        with st.form("account_form"):
            st.text_input("Full Name", "John Doe")
            st.text_input("Email", "john.doe@example.com")
            timezone = st.selectbox("Timezone", TIMEZONES, index=TIMEZONES.index("UTC"))
            if st.form_submit_button("Save Settings"):
                st.success("Settings updated successfully")
                
        st.subheader("Data Privacy")
        st.info("Your data is stored locally and never shared with third parties")
        if st.button("Clear All Data"):
            st.session_state.clear()
            st.success("All local data has been cleared")

# Run the application
if __name__ == "__main__":
    main()
