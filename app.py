
import streamlit as st
import json
import re
import requests
import subprocess
import tempfile
import time
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

# Download NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# Optional imports for features; handle ImportError where used
try:
    from sklearn.ensemble import RandomForestRegressor as _RandomForestRegressor
    from sklearn.metrics import mean_squared_error as _mean_squared_error, r2_score as _r2_score
    from sklearn.model_selection import train_test_split as _train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
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

# ===================== LIVE DATA PROVIDERS =====================
import streamlit as st
import yfinance as yf
import requests
import feedparser
import pandas as pd
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from functools import lru_cache

import streamlit as st
import yfinance as yf
import requests
import feedparser
import pandas as pd
import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class LiveDataProvider:
    """Enhanced live data provider with proper caching implementation"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
        self.logger = logging.getLogger(__name__)

    # Static methods for better caching
    @staticmethod
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

    @staticmethod
    @st.cache_data(ttl=300, show_spinner="Fetching crypto data...")
    def get_crypto_data(coin_id: str = "bitcoin") -> Dict[str, float]:
        """
        Fetch cryptocurrency data with cache-safe parameters
        Args:
            coin_id: CoinGecko compatible coin ID (lowercase, no symbols)
        Returns:
            Dictionary with price data
        """
        try:
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': coin_id.lower().strip(),
                'vs_currencies': 'usd',
                'include_24hr_change': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Simplify the return structure
            return {
                'price': data[coin_id]['usd'],
                'change_24h': data[coin_id]['usd_24h_change'],
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Crypto data error: {e}")
            return {}

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner="Fetching news...")
    def get_news_feed(source_url: str, max_items: int = 5) -> List[Dict[str, str]]:
        """Fetch news feed with cache-safe parameters"""
        try:
            feed = feedparser.parse(source_url)
            return [{
                'title': entry.get('title', ''),
                'link': entry.get('link', '#'),
                'published': entry.get('published', ''),
                'source': feed.feed.get('title', source_url)
            } for entry in feed.entries[:max_items]]
        except Exception as e:
            logging.error(f"News feed error: {e}")
            return []

    # Instance method for stateful operations
    def get_multiple_crypto_prices(self, coin_ids: List[str]) -> Dict[str, Dict]:
        """Get prices for multiple coins (not cached)"""
        results = {}
        for coin_id in coin_ids:
            results[coin_id] = self.get_crypto_data(coin_id)
        return results
        for source in sources[:3]:  # Limit to top 3 sources
            try:
                feed = feedparser.parse(source)
                for entry in feed.entries[:max(3, max_articles//len(sources))]:
                    article = {
                        'title': entry.get('title', 'No title'),
                        'link': entry.get('link', '#'),
                        'published': entry.get('published', ''),
                        'summary': entry.get('summary', ''),
                        'source': feed.feed.get('title', source)
                    }
                    all_articles.append(article)
                    
                    if len(all_articles) >= max_articles:
                        break
                        
            except Exception as e:
                self.logger.warning(f"Error fetching news from {source}: {e}")
                
        return sorted(all_articles, 
                     key=lambda x: datetime.strptime(x['published'], '%a, %d %b %Y %H:%M:%S %Z') 
                     if x['published'] else datetime.now(), 
                     reverse=True)[:max_articles]

    @st.cache_data(ttl=1800, show_spinner="Checking weather...")
    def get_weather_data(self, location: str, api_key: Optional[str] = None) -> Dict:
        """Fetch weather data with location validation"""
        api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        if not api_key:
            return {"error": "API key not configured"}
            
        try:
            # First try geocoding to get coordinates
            geo_url = "http://api.openweathermap.org/geo/1.0/direct"
            geo_params = {
                'q': location,
                'limit': 1,
                'appid': api_key
            }
            
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

    @st.cache_data(ttl=86400, show_spinner="Fetching economic data...")  # 24 hour cache
    def get_economic_indicators(self, api_key: Optional[str] = None) -> Dict:
        """Fetch economic indicators with better data structure"""
        api_key = api_key or os.getenv('FRED_API_KEY')
        if not api_key:
            return {"error": "API key not configured"}
            
        indicators = {
            'GDP': {
                'series_id': 'GDP',
                'name': 'Gross Domestic Product',
                'unit': 'Billions of Dollars',
                'frequency': 'Quarterly'
            },
            'Unemployment': {
                'series_id': 'UNRATE',
                'name': 'Unemployment Rate',
                'unit': 'Percent',
                'frequency': 'Monthly'
            },
            'Inflation': {
                'series_id': 'CPIAUCSL',
                'name': 'Consumer Price Index',
                'unit': 'Index',
                'frequency': 'Monthly'
            }
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
                    'as_of': datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.warning(f"Error fetching {key}: {e}")
                results[key] = {
                    'error': str(e),
                    'metadata': config
                }
                
        return results

    def clear_cache(self, cache_type: Optional[str] = None):
        """Clear specific or all cached data"""
        if cache_type == 'stocks' or not cache_type:
            st.cache_data.clear()
        if cache_type == 'database' or not cache_type:
            self.cache_manager.clear_all_cache()

# ===================== ENHANCED UI COMPONENTS =====================
class EnhancedUIComponents:
    """Enhanced UI components with modern styling"""
    
    @staticmethod
    def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal"):
        """Create a modern metric card"""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(
                label=title,
                value=value,
                delta=delta,
                delta_color=delta_color
            )

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
                    background-color: #1f77b4;
                    width: {progress * 100}%;
                    height: 100%;
                    border-radius: 10px;
                    transition: width 0.3s ease;
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    @staticmethod
    def create_sentiment_badge(sentiment: str):
        """Create a colored badge for sentiment"""
        color_map = {
            "Positive": "green",
            "Negative": "red",
            "Neutral": "gray"
        }
        return st.markdown(
            f"<span style='background-color:{color_map.get(sentiment, 'gray')};"
            f"color:white;padding:0.2rem 0.5rem;border-radius:0.25rem;'>"
            f"{sentiment}</span>",
            unsafe_allow_html=True
        )

# ===================== ENHANCED ANALYTICS ENGINE =====================
class EnhancedAnalyticsEngine:
    """Enhanced analytics with ML capabilities"""
    
    def __init__(self):
        self.data_provider = LiveDataProvider()
        self.scaler = StandardScaler() if 'StandardScaler' in globals() else None
        self.sia = SentimentIntensityAnalyzer()

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
            prediction = None
            if _RandomForestRegressor and len(data) >= 30:
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
            if not _RandomForestRegressor:
                return {"error": "ML libraries not available"}

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
                        'sentiment': sentiment_label,
                        'vader_neg': vader_scores['neg'],
                        'vader_neu': vader_scores['neu'],
                        'vader_pos': vader_scores['pos']
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
                stopwords=set(nltk.corpus.stopwords.words('english'))
                .generate(full_text)
            )
            # Convert to PIL Image
            return wordcloud.to_image()
        except Exception as e:
            logger.error(f"Word cloud generation error: {e}")
            return None
    
    def cluster_news(self, news_data: List[Dict]) -> List[Dict]:
        """Cluster news articles using K-Means"""
        try:
            if not _RandomForestRegressor or not news_data:
                return news_data
                
            # Create TF-IDF vectors
            from sklearn.feature_extraction.text import TfidfVectorizer
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
        st.session_state.data_provider = LiveDataProvider()
    if 'analytics_engine' not in st.session_state:
        st.session_state.analytics_engine = EnhancedAnalyticsEngine()
    if 'ui_components' not in st.session_state:
        st.session_state.ui_components = EnhancedUIComponents()

    # Enhanced header with styling
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #1f77b4, #17a2b8);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .sidebar-header {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border-left: 4px solid #1f77b4;
        }
        .news-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
            border-left: 4px solid #4CAF50;
            transition: all 0.3s ease;
        }
        .news-card:hover {
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transform: translateX(5px);
        }
        .cluster-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
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
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4 !important;
            color: white !important;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1>📊 Enhanced Data Analytics Platform</h1>
        <p>Real-time data analysis with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
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
        st.subheader("📡 Data Sources")
        data_sources = st.multiselect(
            "Select Data Sources",
            ["Stock Market", "Cryptocurrency", "News", "Weather", "Economic Indicators"],
            default=["Stock Market", "News"]
        )

        # Refresh interval
        refresh_interval = st.slider(
            "Auto-refresh interval (minutes)",
            min_value=1,
            max_value=60,
            value=5,
            help="How often to refresh live data"
        )

        # Analysis settings
        st.subheader("⚙️ Analysis Settings")
        enable_ml = st.checkbox("Enable ML Predictions", value=True)
        enable_sentiment = st.checkbox("Enable Sentiment Analysis", value=True)
        enable_clustering = st.checkbox("Enable News Clustering", value=True)
        
        # Personalization
        st.subheader("🎨 Personalization")
        theme = st.selectbox("Theme", ["Light", "Dark", "Blue"], index=0)
        density = st.selectbox("Data Density", ["Compact", "Normal", "Detailed"], index=1)

        # User info
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center;">
            <p>Built with ❤️ using Streamlit</p>
            <p>Version 2.0 | Free Edition</p>
        </div>
        """, unsafe_allow_html=True)

    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Market Dashboard",
        "📰 News Analysis",
        "🌤️ Weather & Environment",
        "🔍 Custom Analysis",
        "📊 Analytics Dashboard"
    ])

    with tab1:
        st.header("📈 Live Market Dashboard")
        
        # Stock analysis section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            stock_symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter stock symbol (e.g., AAPL, GOOGL)")
            period = st.selectbox("Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2)
            
            if st.button("🔍 Analyze Stock", type="primary"):
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
                            vertical_spacing=0.1
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
                                marker_color='lightblue'
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
            crypto_coins = ["bitcoin", "ethereum", "cardano", "polkadot", "chainlink"]
            
            for coin in crypto_coins:
                crypto_data = st.session_state.data_provider.get_crypto_data(coin)
                if crypto_data and coin in crypto_data:
                    coin_info = crypto_data[coin]
                    price = coin_info.get('usd', 0)
                    change = coin_info.get('usd_24h_change', 0)
                    
                    st.metric(
                        coin.capitalize(),
                        f"${price:,.2f}",
                        f"{change:.2f}%",
                        delta_color="inverse"
                    )

    with tab2:
        st.header("📰 News Analysis Dashboard")
        
        # News categories
        news_topic = st.selectbox(
            "Select News Category",
            ["technology", "business", "science", "politics", "sports"],
            index=0
        )
        
        max_articles = st.slider("Number of Articles", 5, 50, 10)
        
        if st.button("📰 Fetch Latest News", type="primary"):
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
                    
                    # Display in tabs
                    tab_articles, tab_sentiment, tab_wordcloud = st.tabs([
                        "📝 Articles", 
                        "😊 Sentiment Analysis", 
                        "☁️ Word Cloud"
                    ])
                    
                    with tab_articles:
                        # Display articles in cards
                        cluster_colors = {
                            0: "#1f77b4",
                            1: "#ff7f0e",
                            2: "#2ca02c",
                            3: "#d62728",
                            4: "#9467bd"
                        }
                        
                        for i, article in enumerate(news_data):
                            with st.container():
                                st.markdown(f"""
                                <div class="news-card">
                                    <h4>{article['title']}</h4>
                                    <p><strong>Source:</strong> {article['source']} | 
                                    <strong>Published:</strong> {article.get('published', 'N/A')}</p>
                                    <p>{article.get('summary', '')[:200]}...</p>
                                    <a href="{article['link']}" target="_blank">Read full article</a>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show sentiment badge if available
                                if sentiment_results and i < len(sentiment_results['individual_sentiments']):
                                    sentiment = sentiment_results['individual_sentiments'][i]['sentiment']
                                    st.session_state.ui_components.create_sentiment_badge(sentiment)
                                
                                # Show cluster badge if clustered
                                if 'cluster' in article:
                                    cluster_id = article['cluster']
                                    st.markdown(f"""
                                    <div class="cluster-badge" style="background-color: {cluster_colors.get(cluster_id, '#999999')}; color: white;">
                                        Cluster #{cluster_id + 1}
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    with tab_sentiment:
                        if sentiment_results:
                            if "error" not in sentiment_results:
                                # Overall sentiment
                                st.subheader("Overall Sentiment")
                                col_s1, col_s2, col_s3 = st.columns(3)
                                with col_s1:
                                    st.metric("Overall", sentiment_results['overall_sentiment'])
                                with col_s2:
                                    st.metric("Avg. Polarity", f"{sentiment_results['average_polarity']:.2f}")
                                
                                # Sentiment distribution
                                st.subheader("Sentiment Distribution")
                                dist = sentiment_results['sentiment_distribution']
                                fig_dist = px.pie(
                                    names=list(dist.keys()),
                                    values=list(dist.values()),
                                    color=list(dist.keys()),
                                    color_discrete_map={
                                        "Positive": "#2ca02c",
                                        "Neutral": "#1f77b4",
                                        "Negative": "#d62728"
                                    }
                                )
                                st.plotly_chart(fig_dist, use_container_width=True)
                                
                                # Sentiment over time (mock data)
                                st.subheader("Sentiment Trend")
                                dates = pd.date_range(end=datetime.datetime.now(), periods=7, freq='D')
                                sentiment_trend = np.random.uniform(-0.5, 0.5, size=7)
                                df_trend = pd.DataFrame({
                                    'Date': dates,
                                    'Sentiment': sentiment_trend
                                })
                                fig_trend = px.line(
                                    df_trend, 
                                    x='Date', 
                                    y='Sentiment',
                                    title="Sentiment Trend (Last 7 Days)",
                                    markers=True
                                )
                                fig_trend.update_layout(
                                    yaxis_range=[-1, 1],
                                    shapes=[{
                                        'type': 'line',
                                        'y0': 0,
                                        'y1': 0,
                                        'xref': 'paper',
                                        'x0': 0,
                                        'x1': 1,
                                        'line': {
                                            'color': 'gray',
                                            'dash': 'dash'
                                        }
                                    }]
                                )
                                st.plotly_chart(fig_trend, use_container_width=True)
                            else:
                                st.error(f"Sentiment analysis error: {sentiment_results['error']}")
                        else:
                            st.info("Enable sentiment analysis in settings")
                    
                    with tab_wordcloud:
                        if news_data:
                            news_texts = [article['title'] + ' ' + article.get('summary', '') for article in news_data]
                            with st.spinner("Generating word cloud..."):
                                wordcloud_img = st.session_state.analytics_engine.generate_word_cloud(news_texts)
                                if wordcloud_img:
                                    st.image(wordcloud_img, caption="Word Cloud of News Content")
                                else:
                                    st.warning("Could not generate word cloud")
                        else:
                            st.info("No news data available")
                else:
                    st.warning("No news articles found for this category")

    with tab3:
        st.header("🌤️ Weather & Environment Dashboard")
        col1, col2 = st.columns(2)
        
        with col1:
            city = st.text_input("Enter City Name", "New York")
            if st.button("Get Weather", type="primary"):
                with st.spinner("Fetching weather data..."):
                    weather_data = st.session_state.data_provider.get_weather_data(city)
                    if "error" not in weather_data:
                        # Display weather info
                        st.subheader(f"Weather in {city}")
                        col_w1, col_w2 = st.columns(2)
                        with col_w1:
                            st.metric("Temperature", f"{weather_data['main']['temp']}°C")
                            st.metric("Humidity", f"{weather_data['main']['humidity']}%")
                        with col_w2:
                            st.metric("Feels Like", f"{weather_data['main']['feels_like']}°C")
                            st.metric("Pressure", f"{weather_data['main']['pressure']} hPa")
                        
                        # Weather description
                        weather_desc = weather_data['weather'][0]['description'].title()
                        st.markdown(f"**Conditions:** {weather_desc}")
                        
                        # Weather icon
                        icon_code = weather_data['weather'][0]['icon']
                        st.image(f"http://openweathermap.org/img/wn/{icon_code}@2x.png", width=100)
                        
                        # Air quality (mock data)
                        st.subheader("Air Quality Index")
                        aqi = random.randint(20, 120)
                        status = "Good" if aqi <= 50 else "Moderate" if aqi <= 100 else "Unhealthy"
                        color = "green" if aqi <= 50 else "orange" if aqi <= 100 else "red"
                        
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(90deg, #00e400 0%, #ffff00 50%, #ff0000 100%);
                            height: 30px;
                            border-radius: 15px;
                            position: relative;
                            margin: 1rem 0;
                        ">
                            <div style="
                                position: absolute;
                                left: {min(aqi, 150)/1.5}%;
                                top: -35px;
                                font-weight: bold;
                                color: {color};
                            ">
                                ▼ {aqi} ({status})
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Forecast chart (mock data)
                        st.subheader("5-Day Forecast")
                        days = ['Today', 'Tomorrow', 'Day 3', 'Day 4', 'Day 5']
                        temps = [weather_data['main']['temp'] + random.uniform(-3, 3) for _ in range(5)]
                        fig_forecast = px.line(
                            x=days,
                            y=temps,
                            title="Temperature Forecast",
                            markers=True
                        )
                        fig_forecast.update_layout(
                            xaxis_title="Day",
                            yaxis_title="Temperature (°C)"
                        )
                        st.plotly_chart(fig_forecast, use_container_width=True)
                    else:
                        st.error(f"Weather data error: {weather_data['error']}")
        
        with col2:
            st.subheader("🌍 Environmental Data")
            st.info("Environmental monitoring features coming soon!")
            
            # Water quality (mock data)
            st.markdown("### 💧 Water Quality")
            col_wq1, col_wq2 = st.columns(2)
            with col_wq1:
                st.metric("pH Level", "7.2")
                st.metric("Turbidity", "5.2 NTU")
            with col_wq2:
                st.metric("Dissolved Oxygen", "8.1 mg/L")
                st.metric("Temperature", "18.3°C")
            
            # Pollution levels (mock data)
            st.markdown("### 🏭 Pollution Levels")
            pollutants = {
                "PM2.5": 24,
                "PM10": 38,
                "NO2": 22,
                "SO2": 9,
                "O3": 45
            }
            for pol, level in pollutants.items():
                status = "Good" if level < 30 else "Moderate" if level < 60 else "Poor"
                st.progress(level/100, f"{pol}: {level} µg/m³ ({status})")
    
    with tab4:
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
            
    with tab5:
        st.header("📊 Analytics Dashboard")
        st.subheader("Usage Statistics and Insights")
        
        # Mock analytics data
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            st.metric("Total Users", "1,248", "12%")
        with col_a2:
            st.metric("Active Sessions", "87", "-3%")
        with col_a3:
            st.metric("Avg. Session Time", "14m 32s", "8%")
        
        # Usage chart
        st.subheader("Feature Usage")
        features = ["Market Analysis", "News", "Weather", "Custom Analysis"]
        usage = [45, 32, 18, 5]
        fig_usage = px.bar(
            x=features,
            y=usage,
            color=features,
            labels={'x': 'Feature', 'y': 'Usage (%)'},
            text=usage
        )
        st.plotly_chart(fig_usage, use_container_width=True)
        
        # User activity map
        st.subheader("User Activity Map")
        countries = ["USA", "India", "UK", "Germany", "Canada", "Australia", "Brazil"]
        activity = [65, 48, 32, 28, 22, 18, 15]
        fig_map = px.choropleth(
            locations=countries,
            locationmode="country names",
            color=activity,
            hover_name=countries,
            color_continuous_scale='Blues',
            title="User Activity by Country"
        )
        st.plotly_chart(fig_map, use_container_width=True)

if __name__ == "__main__":
    main()
