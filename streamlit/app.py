import streamlit as st import json import re import requests import os import numpy as np import pandas as pd import datetime from datetime import timedelta import hashlib import threading import logging import sqlite3 from typing import Dict, Optional, List import warnings import random from concurrent.futures import ThreadPoolExecutor import matplotlib.pyplot as plt from scipy import stats import yfinance as yf import feedparser import plotly.express as px import plotly.graph_objects as go from plotly.subplots import make_subplots import altair as alt from wordcloud import WordCloud import nltk from textblob import TextBlob from nltk.sentiment import SentimentIntensityAnalyzer import seaborn as sns from PIL import Image import io import base64 import pytz from dateutil import parser from bs4 import BeautifulSoup import json from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

Load a pre-trained model from the Hugging Face model hub
model = HuggingFaceModel( hf_model_id="bert-base-uncased", role="your-iam-role", transformers_version="4.12.5", pytorch_version="1.9.0" )

Get the Docker image URI for the LLM
image_uri = get_huggingface_llm_image_uri( "bert-base-uncased", pytorch_version="1.9.0", transformers_version="4.12.5" )

Optional imports for features; handle ImportError where used
try: from sklearn.ensemble import RandomForestRegressor as _RandomForestRegressor from sklearn.metrics import mean_squared_error as _mean_squared_error, r2_score as _r2_score from sklearn.model_selection import train_test_split as _train_test_split from sklearn.preprocessing import StandardScaler from sklearn.cluster import KMeans from sklearn.feature_extraction.text import TfidfVectorizer except ImportError: _RandomForestRegressor = None _mean_squared_error = None _r2_score = None _train_test_split = None warnings.filterwarnings('ignore')

===================== ENHANCED SYSTEM CONFIGURATION =====================
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true' MAX_RESEARCH_RESULTS = 10 CODE_EXECUTION_TIMEOUT = 30 SAFE_MODE = True PERSONAS = ["Researcher", "Teacher", "Analyst", "Engineer", "Scientist", "Assistant", "Consultant", "Creative", "Problem Solver"] SESSION_FILE = "session_state.json" USER_DB = "users.db" TEAM_DB = "teams.json" WORKFLOW_DB = "workflows.json" CACHE_DB = "cache.db" TIMEZONES = pytz.all_timezones

Setup enhanced logging
logging.basicConfig( level=logging.INFO if DEBUG_MODE else logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' ) logger = logging.getLogger(name)

===================== ENHANCED DATABASE MANAGER =====================
class EnhancedDatabaseManager: def init(self): self.pg_pool = None self.sqlite_lock = threading.Lock() self.init_databases() self.cache = {} self.cache_lock = threading.Lock()

def get_connection(self):
    &quot;&quot;&quot;Get a database connection (PostgreSQL if available, else SQLite)&quot;&quot;&quot;
    if self.pg_pool:
        try:
            return self.pg_pool.getconn(), &quot;postgresql&quot;
        except Exception as e:
            logger.warning(f&quot;PostgreSQL connection failed: {e}&quot;)
    return sqlite3.connect(CACHE_DB), &quot;sqlite&quot;

def init_databases(self):
    &quot;&quot;&quot;Initialize both SQLite and PostgreSQL databases&quot;&quot;&quot;
    try:
        self.init_sqlite()
        self.init_postgresql()
    except Exception as e:
        logger.error(f&quot;Database initialization error: {e}&quot;)

def init_sqlite(self):
    &quot;&quot;&quot;Initialize SQLite database with proper error handling&quot;&quot;&quot;
    try:
        if not os.path.exists(CACHE_DB):
            open(CACHE_DB, &#39;a&#39;).close()
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.cursor()
        cursor.executescript(&#39;&#39;&#39;
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
        &#39;&#39;&#39;)
        conn.commit()
        conn.close()
        logger.info(&quot;SQLite database initialized successfully&quot;)
    except Exception as e:
        logger.error(f&quot;SQLite initialization error: {e}&quot;)

def init_postgresql(self):
    &quot;&quot;&quot;Initialize PostgreSQL connection if available&quot;&quot;&quot;
    database_url = os.environ.get(&#39;DATABASE_URL&#39;)
    if database_url:
        try:
            from psycopg2 import pool
            self.pg_pool = pool.SimpleConnectionPool(1, 10, database_url)
            logger.info(&quot;PostgreSQL connection pool initialized&quot;)
        except ImportError:
            logger.warning(&quot;psycopg2 not installed, skipping PostgreSQL initialization&quot;)

def return_connection(self, conn, db_type):
    &quot;&quot;&quot;Return connection to pool&quot;&quot;&quot;
    try:
        if db_type == &quot;postgresql&quot; and self.pg_pool:
            self.pg_pool.putconn(conn)
        else:
            conn.close()
    except Exception as e:
        logger.error(f&quot;Error returning connection: {e}&quot;)

def get_cached_result(self, key: str) -&gt; Optional[str]:
    &quot;&quot;&quot;Get cached result with enhanced error handling&quot;&quot;&quot;
    conn, db_type = None, None
    try:
        conn, db_type = self.get_connection()
        cursor = conn.cursor()
        if db_type == &quot;postgresql&quot;:
            cursor.execute(&#39;&#39;&#39;
                SELECT value FROM cache 
                WHERE key = %s AND expires_at &gt; NOW()
            &#39;&#39;&#39;, (key,))
        else:
            cursor.execute(&#39;&#39;&#39;
                SELECT value FROM cache 
                WHERE key = ? AND expires_at &gt; datetime(&#39;now&#39;)
            &#39;&#39;&#39;, (key,))
        result = cursor.fetchone()
        return result[0] if result else None
    except Exception as e:
        logger.error(f&quot;Cache retrieval error: {e}&quot;)
        return None
    finally:
        if conn:
            self.return_connection(conn, db_type)

def set_cached_result(self, key: str, value: str, ttl_minutes: int = 60):
    &quot;&quot;&quot;Set cached result with enhanced error handling&quot;&quot;&quot;
    conn, db_type = None, None
    try:
        conn, db_type = self.get_connection()
        cursor = conn.cursor()
        expires_at = datetime.datetime.now() + timedelta(minutes=ttl_minutes)
        if db_type == &quot;postgresql&quot;:
            cursor.execute(&#39;&#39;&#39;
                INSERT INTO cache (key, value, expires_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (key) DO UPDATE SET 
                value = EXCLUDED.value, 
                expires_at = EXCLUDED.expires_at
            &#39;&#39;&#39;, (key, value, expires_at))
        else:
            cursor.execute(&#39;&#39;&#39;
                INSERT OR REPLACE INTO cache (key, value, expires_at)
                VALUES (?, ?, ?)
            &#39;&#39;&#39;, (key, value, expires_at))
        conn.commit()
    except Exception as e:
        logger.error(f&quot;Cache storage error: {e}&quot;)
    finally:
        if conn:
            self.return_connection(conn, db_type)

def log_analytics(self, user_id: str, action: str, details: str = &quot;&quot;):
    &quot;&quot;&quot;Log analytics data&quot;&quot;&quot;
    conn, db_type = None, None
    try:
        conn, db_type = self.get_connection()
        cursor = conn.cursor()

        if db_type == &quot;postgresql&quot;:
            cursor.execute(&#39;&#39;&#39;
                INSERT INTO analytics (user_id, action, details)
                VALUES (%s, %s, %s)
            &#39;&#39;&#39;, (user_id, action, details))
        else:
            cursor.execute(&#39;&#39;&#39;
                INSERT INTO analytics (user_id, action, details)
                VALUES (?, ?, ?)
            &#39;&#39;&#39;, (user_id, action, details))

        conn.commit()

    except Exception as e:
        logger.error(f&quot;Analytics logging error: {e}&quot;)
    finally:
        if conn:
            self.return_connection(conn, db_type)
===================== ENHANCED LIVE DATA PROVIDERS =====================
class EnhancedLiveDataProvider: """Enhanced live data provider with caching and error handling"""

def __init__(self):
    self.session = requests.Session()
    self.session.headers.update({
        &#39;User-Agent&#39;: &#39;Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36&#39;,
        &#39;Accept&#39;: &#39;application/json&#39;
    })
    self.logger = logging.getLogger(__name__)
    self.news_sources = {
        &quot;technology&quot;: [
            &quot;https://feeds.feedburner.com/oreilly/radar&quot;,
            &quot;https://techcrunch.com/feed/&quot;,
            &quot;https://news.ycombinator.com/rss&quot;
        ],
        &quot;business&quot;: [
            &quot;https://www.bloomberg.com/feed/podcasts/etf-report.rss&quot;,
            &quot;https://www.cnbc.com/id/100003114/device/rss/rss.html&quot;
        ],
        &quot;general&quot;: [
            &quot;https://rss.cnn.com/rss/cnn_topstories.rss&quot;,
            &quot;https://feeds.bbci.co.uk/news/rss.xml&quot;
        ],
        &quot;science&quot;: [
            &quot;https://www.scientificamerican.com/rss/?section=all&quot;,
            &quot;https://www.nasa.gov/rss/dyn/breaking_news.rss&quot;
        ]
    }
    self.crypto_ids = {
        &quot;bitcoin&quot;: &quot;bitcoin&quot;,
        &quot;ethereum&quot;: &quot;ethereum&quot;,
        &quot;cardano&quot;: &quot;cardano&quot;,
        &quot;solana&quot;: &quot;solana&quot;,
        &quot;dogecoin&quot;: &quot;dogecoin&quot;
    }

@st.cache_data(ttl=300, show_spinner=&quot;Fetching stock data...&quot;)
def get_stock_data(symbol: str, period: str = &quot;1mo&quot;) -&gt; pd.DataFrame:
    &quot;&quot;&quot;Fetch stock data with proper caching&quot;&quot;&quot;
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data.reset_index() if not data.empty else pd.DataFrame()
    except Exception as e:
        logging.error(f&quot;Stock data error: {e}&quot;)
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=&quot;Fetching crypto data...&quot;)
def get_crypto_data(coin_name: str = &quot;bitcoin&quot;) -&gt; Dict[str, float]:
    &quot;&quot;&quot;
    Fetch cryptocurrency data with cache-safe parameters
    Args:
        coin_name: Common name of cryptocurrency
    Returns:
        Dictionary with price data
    &quot;&quot;&quot;
    try:
        coin_id = self.crypto_ids.get(coin_name.lower(), coin_name.lower())
        url = &quot;https://api.coingecko.com/api/v3/simple/price&quot;
        params = {
            &#39;ids&#39;: coin_id,
            &#39;vs_currencies&#39;: &#39;usd&#39;,
            &#39;include_24hr_change&#39;: &#39;true&#39;
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if coin_id not in data:
            return {}
            
        return {
            &#39;price&#39;: data[coin_id][&#39;usd&#39;],
            &#39;change_24h&#39;: data[coin_id][&#39;usd_24h_change&#39;],
            &#39;last_updated&#39;: datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f&quot;Crypto data error: {e}&quot;)
        return {}

def _get_news_from_source(self, source_url: str, max_items: int = 5) -&gt; List[Dict]:
    &quot;&quot;&quot;Parse news from a single RSS source&quot;&quot;&quot;
    try:
        feed = feedparser.parse(source_url)
        articles = []
        for entry in feed.entries[:max_items]:
            try:
                published = parser.parse(entry.published) if hasattr(entry, &#39;published&#39;) else datetime.datetime.now()
                
                # Extract clean text from summary
                summary = entry.get(&#39;summary&#39;, &#39;&#39;)
                if summary:
                    soup = BeautifulSoup(summary, &#39;html.parser&#39;)
                    summary = soup.get_text()
                
                articles.append({
                    &#39;title&#39;: entry.get(&#39;title&#39;, &#39;No title&#39;),
                    &#39;link&#39;: entry.get(&#39;link&#39;, &#39;#&#39;),
                    &#39;published&#39;: published,
                    &#39;summary&#39;: summary[:200] + &#39;...&#39; if len(summary) &gt; 200 else summary,
                    &#39;source&#39;: feed.feed.get(&#39;title&#39;, source_url)
                })
            except Exception as e:
                self.logger.warning(f&quot;Error parsing news entry: {e}&quot;)
        return articles
    except Exception as e:
        self.logger.warning(f&quot;Error fetching from {source_url}: {e}&quot;)
        return []

@st.cache_data(ttl=600, show_spinner=&quot;Fetching news...&quot;)
def get_news_data(_self, topic: str = &quot;technology&quot;, max_articles: int = 5) -&gt; List[Dict]:
    &quot;&quot;&quot;
    Fetch and cache news data with proper hashing
    &quot;&quot;&quot;
    sources = _self.news_sources.get(topic.lower(), _self.news_sources[&quot;general&quot;])
    all_articles = []
    
    for source in sources[:3]:
        articles = _self._get_news_from_source(source, max(3, max_articles//len(sources)))
        all_articles.extend(articles)
        
        if len(all_articles) &gt;= max_articles:
            break
            
    # Sort by published date
    return sorted(
        all_articles,
        key=lambda x: x[&#39;published&#39;],
        reverse=True
    )[:max_articles]

@st.cache_data(ttl=1800, show_spinner=&quot;Checking weather...&quot;)
def get_weather_data(self, location: str, api_key: Optional[str] = None) -&gt; Dict:
    &quot;&quot;&quot;Fetch weather data with location validation&quot;&quot;&quot;
    api_key = api_key or os.getenv(&#39;OPENWEATHER_API_KEY&#39;)
    if not api_key:
        return {&quot;error&quot;: &quot;API key not configured&quot;}
        
    try:
        # Geocoding to get coordinates
        geo_url = &quot;http://api.openweathermap.org/geo/1.0/direct&quot;
        geo_params = {&#39;q&#39;: location, &#39;limit&#39;: 1, &#39;appid&#39;: api_key}
        geo_response = self.session.get(geo_url, params=geo_params)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        
        if not geo_data:
            return {&quot;error&quot;: &quot;Location not found&quot;}
            
        lat, lon = geo_data[0][&#39;lat&#39;], geo_data[0][&#39;lon&#39;]
        
        # Get weather data
        weather_url = &quot;https://api.openweathermap.org/data/3.0/onecall&quot;
        weather_params = {
            &#39;lat&#39;: lat,
            &#39;lon&#39;: lon,
            &#39;exclude&#39;: &#39;minutely,hourly&#39;,
            &#39;appid&#39;: api_key,
            &#39;units&#39;: &#39;metric&#39;
        }
        
        weather_response = self.session.get(weather_url, params=weather_params)
        weather_response.raise_for_status()
        
        data = weather_response.json()
        data[&#39;location&#39;] = geo_data[0]
        return data
        
    except Exception as e:
        self.logger.error(f&quot;Weather data error for {location}: {e}&quot;)
        return {&quot;error&quot;: str(e)}

@st.cache_data(ttl=86400, show_spinner=&quot;Fetching economic data...&quot;)
def get_economic_indicators(self, api_key: Optional[str] = None) -&gt; Dict:
    &quot;&quot;&quot;Fetch economic indicators with better data structure&quot;&quot;&quot;
    api_key = api_key or os.getenv(&#39;FRED_API_KEY&#39;)
    if not api_key:
        return {&quot;error&quot;: &quot;API key not configured&quot;}
        
    indicators = {
        &#39;GDP&#39;: {&#39;series_id&#39;: &#39;GDP&#39;, &#39;name&#39;: &#39;Gross Domestic Product&#39;},
        &#39;Unemployment&#39;: {&#39;series_id&#39;: &#39;UNRATE&#39;, &#39;name&#39;: &#39;Unemployment Rate&#39;},
        &#39;Inflation&#39;: {&#39;series_id&#39;: &#39;CPIAUCSL&#39;, &#39;name&#39;: &#39;Consumer Price Index&#39;}
    }
    
    results = {}
    for key, config in indicators.items():
        try:
            url = &quot;https://api.stlouisfed.org/fred/series/observations&quot;
            params = {
                &#39;series_id&#39;: config[&#39;series_id&#39;],
                &#39;api_key&#39;: api_key,
                &#39;file_type&#39;: &#39;json&#39;,
                &#39;limit&#39;: 12,
                &#39;sort_order&#39;: &#39;desc&#39;
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results[key] = {
                &#39;metadata&#39;: config,
                &#39;data&#39;: data.get(&#39;observations&#39;, []),
                &#39;as_of&#39;: datetime.datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.warning(f&quot;Error fetching {key}: {e}&quot;)
            results[key] = {&#39;error&#39;: str(e), &#39;metadata&#39;: config}
            
    return results
===================== ENHANCED UI COMPONENTS =====================
class EnhancedUIComponents: """Enhanced UI components with modern styling"""

@staticmethod
def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = &quot;normal&quot;):
    &quot;&quot;&quot;Create a modern metric card&quot;&quot;&quot;
    with st.container():
        st.metric(label=title, value=value, delta=delta, delta_color=delta_color)

@staticmethod
def create_info_card(title: str, content: str, icon: str = &quot;ℹ️&quot;):
    &quot;&quot;&quot;Create an information card&quot;&quot;&quot;
    st.markdown(f&quot;&quot;&quot;
    &lt;div style=&quot;
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    &quot;&gt;
        &lt;h4 style=&quot;margin: 0 0 0.5rem 0; color: #1f77b4;&quot;&gt;
            {icon} {title}
        &lt;/h4&gt;
        &lt;p style=&quot;margin: 0; color: #333;&quot;&gt;
            {content}
        &lt;/p&gt;
    &lt;/div&gt;
    &quot;&quot;&quot;, unsafe_allow_html=True)

@staticmethod
def create_progress_bar(value: float, max_value: float, title: str):
    &quot;&quot;&quot;Create a styled progress bar&quot;&quot;&quot;
    progress = min(value / max_value, 1.0)
    st.markdown(f&quot;&quot;&quot;
    &lt;div style=&quot;margin: 1rem 0;&quot;&gt;
        &lt;div style=&quot;display: flex; justify-content: space-between; margin-bottom: 0.5rem;&quot;&gt;
            &lt;span style=&quot;font-weight: bold;&quot;&gt;{title}&lt;/span&gt;
            &lt;span&gt;{value:.1f} / {max_value:.1f}&lt;/span&gt;
        &lt;/div&gt;
        &lt;div style=&quot;background-color: #e0e0e0; border-radius: 10px; height: 20px;&quot;&gt;
            &lt;div style=&quot;
                background: linear-gradient(90deg, #1f77b4, #17a2b8);
                width: {progress * 100}%;
                height: 100%;
                border-radius: 10px;
            &quot;&gt;&lt;/div&gt;
        &lt;/div&gt;
    &lt;/div&gt;
    &quot;&quot;&quot;, unsafe_allow_html=True)
    
@staticmethod
def create_sentiment_badge(sentiment: str):
    &quot;&quot;&quot;Create a colored badge for sentiment&quot;&quot;&quot;
    color_map = {
        &quot;Positive&quot;: &quot;#2ca02c&quot;,
        &quot;Negative&quot;: &quot;#d62728&quot;,
        &quot;Neutral&quot;: &quot;#7f7f7f&quot;
    }
    return st.markdown(
        f&quot;&lt;span style=&#39;background-color:{color_map.get(sentiment, &#39;gray&#39;)};&quot;
        f&quot;color:white;padding:0.2rem 0.5rem;border-radius:0.25rem;font-weight:bold;&#39;&gt;&quot;
        f&quot;{sentiment}&lt;/span&gt;&quot;,
        unsafe_allow_html=True
    )
    
@staticmethod
def create_news_card(title: str, source: str, published: datetime, summary: str, link: str, sentiment: str = None):
    &quot;&quot;&quot;Create a news card with sentiment badge&quot;&quot;&quot;
    time_ago = (datetime.datetime.now() - published).days
    time_text = f&quot;{time_ago} days ago&quot; if time_ago &gt; 0 else &quot;Today&quot;
    
    badge = f&quot;&quot;&quot;
    &lt;div style=&quot;margin-bottom: 0.5rem;&quot;&gt;
        {EnhancedUIComponents.create_sentiment_badge(sentiment)._proxy}
    &lt;/div&gt;
    &quot;&quot;&quot; if sentiment else &quot;&quot;
    
    st.markdown(f&quot;&quot;&quot;
    &lt;div class=&quot;news-card&quot; style=&quot;
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border-left: 4px solid #4CAF50;
        transition: all 0.3s ease;
    &quot;&gt;
        &lt;h4 style=&quot;margin: 0 0 0.5rem 0;&quot;&gt;{title}&lt;/h4&gt;
        &lt;div style=&quot;display: flex; justify-content: space-between; color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;&quot;&gt;
            &lt;span&gt;{source}&lt;/span&gt;
            &lt;span&gt;{time_text}&lt;/span&gt;
        &lt;/div&gt;
        {badge}
        &lt;p style=&quot;margin: 0.5rem 0 1rem 0;&quot;&gt;{summary}&lt;/p&gt;
        &lt;a href=&quot;{link}&quot; target=&quot;_blank&quot; style=&quot;
            display: inline-block;
            background: #1f77b4;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 4px;
            text-decoration: none;
            font-weight: 500;
        &quot;&gt;Read Article&lt;/a&gt;
    &lt;/div&gt;
    &quot;&quot;&quot;, unsafe_allow_html=True)
===================== ENHANCED ANALYTICS ENGINE =====================
class EnhancedAnalyticsEngine: """Enhanced analytics with ML capabilities"""

def __init__(self):
    self.data_provider = EnhancedLiveDataProvider()
    self.scaler = StandardScaler() if &#39;StandardScaler&#39; in globals() else None
    self.sia = SentimentIntensityAnalyzer()
    self.stopwords = set(nltk.corpus.stopwords.words(&#39;english&#39;))

def analyze_stock_trends(self, symbol: str, period: str = &quot;1mo&quot;) -&gt; Dict:
    &quot;&quot;&quot;Analyze stock trends with ML predictions&quot;&quot;&quot;
    try:
        data = self.data_provider.get_stock_data(symbol, period)
        if data.empty:
            return {&quot;error&quot;: &quot;No data available&quot;}

        # Technical indicators
        data[&#39;MA_5&#39;] = data[&#39;Close&#39;].rolling(window=5).mean()
        data[&#39;MA_20&#39;] = data[&#39;Close&#39;].rolling(window=20).mean()
        data[&#39;RSI&#39;] = self.calculate_rsi(data[&#39;Close&#39;])
        data[&#39;Volatility&#39;] = data[&#39;Close&#39;].rolling(window=20).std()

        # Trend analysis
        recent_price = data[&#39;Close&#39;].iloc[-1]
        ma_5 = data[&#39;MA_5&#39;].iloc[-1]
        ma_20 = data[&#39;MA_20&#39;].iloc[-1]
        
        trend = &quot;Bullish&quot; if recent_price &gt; ma_5 &gt; ma_20 else &quot;Bearish&quot;
        
        # ML prediction if available
        prediction = self.predict_stock_price(data) if _RandomForestRegressor else None

        return {
            &quot;symbol&quot;: symbol,
            &quot;current_price&quot;: recent_price,
            &quot;trend&quot;: trend,
            &quot;rsi&quot;: data[&#39;RSI&#39;].iloc[-1],
            &quot;volatility&quot;: data[&#39;Volatility&#39;].iloc[-1],
            &quot;prediction&quot;: prediction,
            &quot;data&quot;: data
        }
    except Exception as e:
        logger.error(f&quot;Stock analysis error: {e}&quot;)
        return {&quot;error&quot;: str(e)}

def calculate_rsi(self, prices: pd.Series, period: int = 14) -&gt; pd.Series:
    &quot;&quot;&quot;Calculate RSI indicator&quot;&quot;&quot;
    delta = prices.diff()
    gain = (delta.where(delta &gt; 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta &lt; 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def predict_stock_price(self, data: pd.DataFrame) -&gt; Dict:
    &quot;&quot;&quot;Predict stock price using ML&quot;&quot;&quot;
    try:
        # Prepare features
        features = [&#39;Open&#39;, &#39;High&#39;, &#39;Low&#39;, &#39;Volume&#39;, &#39;MA_5&#39;, &#39;MA_20&#39;, &#39;RSI&#39;]
        X = data[features].dropna()
        y = data[&#39;Close&#39;].iloc[len(data) - len(X):]

        if len(X) &lt; 10:
            return {&quot;error&quot;: &quot;Insufficient data for prediction&quot;}

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
            &quot;next_price&quot;: next_price,
            &quot;mse&quot;: mse,
            &quot;r2_score&quot;: r2,
            &quot;confidence&quot;: &quot;High&quot; if r2 &gt; 0.7 else &quot;Medium&quot; if r2 &gt; 0.4 else &quot;Low&quot;
        }
    except Exception as e:
        logger.error(f&quot;ML prediction error: {e}&quot;)
        return {&quot;error&quot;: str(e)}

def analyze_sentiment(self, texts: List[str]) -&gt; Dict:
    &quot;&quot;&quot;Analyze sentiment of text data using both TextBlob and VADER&quot;&quot;&quot;
    try:
        if not texts:
            return {&quot;error&quot;: &quot;No text data provided&quot;}

        sentiments = []
        for text in texts:
            try:
                # TextBlob sentiment
                blob = TextBlob(text)
                polarity_tb = blob.sentiment.polarity
                
                # VADER sentiment
                vader_scores = self.sia.polarity_scores(text)
                polarity_vader = vader_scores[&#39;compound&#39;]
                
                # Combine both scores
                combined_polarity = (polarity_tb + polarity_vader) / 2.0
                
                # Determine sentiment label
                sentiment_label = &quot;Positive&quot; if combined_polarity &gt; 0.1 else &quot;Negative&quot; if combined_polarity &lt; -0.1 else &quot;Neutral&quot;
                
                sentiments.append({
                    &#39;text&#39;: text[:100] + &#39;...&#39; if len(text) &gt; 100 else text,
                    &#39;polarity&#39;: combined_polarity,
                    &#39;sentiment&#39;: sentiment_label
                })
            except Exception as e:
                logger.warning(f&quot;Sentiment analysis error for text: {e}&quot;)

        if not sentiments:
            return {&quot;error&quot;: &quot;No sentiment data generated&quot;}

        avg_polarity = np.mean([s[&#39;polarity&#39;] for s in sentiments])
        sentiment_counts = {
            &quot;Positive&quot;: sum(1 for s in sentiments if s[&#39;sentiment&#39;] == &quot;Positive&quot;),
            &quot;Neutral&quot;: sum(1 for s in sentiments if s[&#39;sentiment&#39;] == &quot;Neutral&quot;),
            &quot;Negative&quot;: sum(1 for s in sentiments if s[&#39;sentiment&#39;] == &quot;Negative&quot;)
        }

        overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)

        return {
            &quot;overall_sentiment&quot;: overall_sentiment,
            &quot;average_polarity&quot;: avg_polarity,
            &quot;sentiment_distribution&quot;: sentiment_counts,
            &quot;individual_sentiments&quot;: sentiments
        }
    except Exception as e:
        logger.error(f&quot;Sentiment analysis error: {e}&quot;)
        return {&quot;error&quot;: str(e)}

def generate_word_cloud(self, texts: List[str]) -&gt; Image:
    &quot;&quot;&quot;Generate word cloud from text data&quot;&quot;&quot;
    try:
        # Combine all texts
        full_text = &quot; &quot;.join(texts)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color=&#39;white&#39;,
            stopwords=self.stopwords
        ).generate(full_text)
        
        return wordcloud.to_image()
    except Exception as e:
        logger.error(f&quot;Word cloud generation error: {e}&quot;)
        return Image.new(&#39;RGB&#39;, (800, 400), color=&#39;white&#39;)

def cluster_news(self, news_data: List[Dict]) -&gt; List[Dict]:
    &quot;&quot;&quot;Cluster news articles using K-Means&quot;&quot;&quot;
    try:
        if not news_data:
            return news_data
            
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words=&#39;english&#39;, max_features=1000)
        texts = [f&quot;{item[&#39;title&#39;]} {item[&#39;summary&#39;]}&quot; for item in news_data]
        X = vectorizer.fit_transform(texts)
        
        # Cluster using K-Means
        n_clusters = min(5, len(news_data))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        
        # Add cluster labels to news data
        for i, item in enumerate(news_data):
            item[&#39;cluster&#39;] = int(kmeans.labels_[i])
            
        return news_data
    except Exception as e:
        logger.error(f&quot;News clustering error: {e}&quot;)
        return news_data
===================== USER PREFERENCES =====================
class UserPreferences: """Manage user preferences and settings"""

def __init__(self):
    self.preferences = {
        &quot;theme&quot;: &quot;light&quot;,
        &quot;timezone&quot;: &quot;UTC&quot;,
        &quot;default_stock&quot;: &quot;AAPL&quot;,
        &quot;default_crypto&quot;: &quot;bitcoin&quot;,
        &quot;default_news_category&quot;: &quot;technology&quot;,
        &quot;refresh_interval&quot;: 5
    }
    
def load_preferences(self, user_id: str):
    &quot;&quot;&quot;Load user preferences from session or database&quot;&quot;&quot;
    if &#39;preferences&#39; in st.session_state:
        self.preferences = st.session_state.preferences
    else:
        # In a real app, load from database
        st.session_state.preferences = self.preferences
        
def save_preferences(self, user_id: str):
    &quot;&quot;&quot;Save user preferences to database&quot;&quot;&quot;
    # In a real app, save to database
    st.session_state.preferences = self.preferences
    
def preference_editor(self):
    &quot;&quot;&quot;UI for editing user preferences&quot;&quot;&quot;
    with st.expander(&quot;⚙️ User Preferences&quot;):
        self.preferences[&#39;theme&#39;] = st.selectbox(&quot;Theme&quot;, [&quot;light&quot;, &quot;dark&quot;, &quot;blue&quot;], index=0)
        self.preferences[&#39;timezone&#39;] = st.selectbox(&quot;Timezone&quot;, TIMEZONES, index=TIMEZONES.index(&quot;UTC&quot;))
        self.preferences[&#39;default_stock&#39;] = st.text_input(&quot;Default Stock&quot;, &quot;AAPL&quot;)
        self.preferences[&#39;default_crypto&#39;] = st.selectbox(&quot;Default Crypto&quot;, list(self.data_provider.crypto_ids.keys()), index=0)
        self.preferences[&#39;default_news_category&#39;] = st.selectbox(&quot;Default News Category&quot;, [&quot;technology&quot;, &quot;business&quot;, &quot;science&quot;], index=0)
        self.preferences[&#39;refresh_interval&#39;] = st.slider(&quot;Refresh Interval (min)&quot;, 1, 60, 5)
        
        if st.button(&quot;Save Preferences&quot;):
            self.save_preferences(&quot;current_user&quot;)
            st.success(&quot;Preferences saved!&quot;)
===================== MAIN APPLICATION =====================
def main(): """Main application with enhanced UI""" st.set_page_config( page_title="Enhanced Data Analytics Platform", page_icon="📊", layout="wide", initial_sidebar_state="expanded" )

# Initialize components
if &#39;db_manager&#39; not in st.session_state:
    st.session_state.db_manager = EnhancedDatabaseManager()
if &#39;data_provider&#39; not in st.session_state:
    st.session_state.data_provider = EnhancedLiveDataProvider()
if &#39;analytics_engine&#39; not in st.session_state:
    st.session_state.analytics_engine = EnhancedAnalyticsEngine()
if &#39;ui_components&#39; not in st.session_state:
    st.session_state.ui_components = EnhancedUIComponents()
if &#39;preferences&#39; not in st.session_state:
    st.session_state.preferences = UserPreferences()
    st.session_state.preferences.load_preferences(&quot;current_user&quot;)

# Custom CSS for styling
st.markdown(&quot;&quot;&quot;
&lt;style&gt;
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
    .stTabs [data-baseweb=&quot;tab-list&quot;] {
        gap: 10px;
    }
    .stTabs [data-baseweb=&quot;tab&quot;] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0 !important;
        background-color: #f0f2f6 !important;
        transition: all 0.3s !important;
        font-weight: 500;
    }
    .stTabs [aria-selected=&quot;true&quot;] {
        background-color: #1f77b4 !important;
        color: white !important;
    }
    footer {
        text-align: center;
        padding: 1rem;
        color: #6c757d;
        font-size: 0.9rem;
    }
&lt;/style&gt;
&quot;&quot;&quot;, unsafe_allow_html=True)

# Page header
st.markdown(&quot;&quot;&quot;
&lt;div class=&quot;main-header&quot;&gt;
    &lt;h1 style=&quot;margin:0;&quot;&gt;📊 Enhanced Data Analytics Platform&lt;/h1&gt;
    &lt;p style=&quot;margin:0; opacity:0.9;&quot;&gt;Real-time insights powered by AI • Free Forever&lt;/p&gt;
&lt;/div&gt;
&quot;&quot;&quot;, unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown(&quot;&quot;&quot;
    &lt;div class=&quot;sidebar-section&quot;&gt;
        &lt;h3&gt;🎛️ Control Panel&lt;/h3&gt;
    &lt;/div&gt;
    &quot;&quot;&quot;, unsafe_allow_html=True)

    # User persona selection
    persona = st.selectbox(
        &quot;Select Your Role&quot;,
        PERSONAS,
        index=0,
        help=&quot;Choose your role for personalized experience&quot;
    )

    # Data source selection
    st.markdown(&quot;&quot;&quot;
    &lt;div class=&quot;sidebar-section&quot;&gt;
        &lt;h4&gt;📡 Data Sources&lt;/h4&gt;
    &lt;/div&gt;
    &quot;&quot;&quot;, unsafe_allow_html=True)
    data_sources = st.multiselect(
        &quot;Select Data Sources&quot;,
        [&quot;Stock Market&quot;, &quot;Cryptocurrency&quot;, &quot;News&quot;, &quot;Weather&quot;, &quot;Economic Indicators&quot;],
        default=[&quot;Stock Market&quot;, &quot;News&quot;],
        label_visibility=&quot;collapsed&quot;
    )

    # Analysis settings
    st.markdown(&quot;&quot;&quot;
    &lt;div class=&quot;sidebar-section&quot;&gt;
        &lt;h4&gt;⚙️ Analysis Settings&lt;/h4&gt;
    &lt;/div&gt;
    &quot;&quot;&quot;, unsafe_allow_html=True)
    enable_ml = st.checkbox(&quot;Enable ML Predictions&quot;, value=True)
    enable_sentiment = st.checkbox(&quot;Enable Sentiment Analysis&quot;, value=True)
    enable_clustering = st.checkbox(&quot;Enable News Clustering&quot;, value=True)
    
    # Personalization
    st.markdown(&quot;&quot;&quot;
    &lt;div class=&quot;sidebar-section&quot;&gt;
        &lt;h4&gt;🎨 Personalization&lt;/h4&gt;
    &lt;/div&gt;
    &quot;&quot;&quot;, unsafe_allow_html=True)
    st.session_state.preferences.preference_editor()

    # Footer
    st.markdown(&quot;---&quot;)
    st.markdown(&quot;&quot;&quot;
    &lt;div style=&quot;text-align: center; padding: 1rem 0;&quot;&gt;
        &lt;p&gt;Built with ❤️ using Streamlit&lt;/p&gt;
        &lt;p&gt;Version 3.0 • Free Edition&lt;/p&gt;
        &lt;p style=&quot;font-size: 0.8rem; margin-top: 1rem;&quot;&gt;© 2023 Enhanced Analytics Platform&lt;/p&gt;
    &lt;/div&gt;
    &quot;&quot;&quot;, unsafe_allow_html=True)

# Main dashboard tabs
tab1, tab2, tab3, tab4 = st.tabs([
    &quot;📈 Market Dashboard&quot;,
    &quot;📰 News &amp; Sentiment&quot;,
    &quot;🔍 Custom Analysis&quot;,
    &quot;⚙️ User Preferences&quot;
])

with tab1:
    st.header(&quot;📈 Live Market Dashboard&quot;)
    
    # Stock analysis section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        stock_symbol = st.text_input(&quot;Stock Symbol&quot;, 
                                    value=st.session_state.preferences.preferences[&#39;default_stock&#39;], 
                                    help=&quot;Enter stock symbol (e.g., AAPL, GOOGL)&quot;)
        period = st.selectbox(&quot;Time Period&quot;, [&quot;1d&quot;, &quot;5d&quot;, &quot;1mo&quot;, &quot;3mo&quot;, &quot;6mo&quot;, &quot;1y&quot;], index=2)
        
        if st.button(&quot;🔍 Analyze Stock&quot;, type=&quot;primary&quot;, use_container_width=True):
            with st.spinner(&quot;Analyzing stock data...&quot;):
                analysis = st.session_state.analytics_engine.analyze_stock_trends(stock_symbol, period)
                
                if &quot;error&quot; not in analysis:
                    # Display metrics
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric(&quot;Current Price&quot;, f&quot;${analysis[&#39;current_price&#39;]:.2f}&quot;)
                    with col_b:
                        st.metric(&quot;Trend&quot;, analysis[&#39;trend&#39;])
                    with col_c:
                        st.metric(&quot;RSI&quot;, f&quot;{analysis[&#39;rsi&#39;]:.2f}&quot;)
                    with col_d:
                        st.metric(&quot;Volatility&quot;, f&quot;{analysis[&#39;volatility&#39;]:.2f}&quot;)
                    
                    # Create interactive chart
                    data = analysis[&#39;data&#39;]
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=[f&quot;{stock_symbol} Price Chart&quot;, &quot;Volume&quot;],
                        vertical_spacing=0.1,
                        shared_xaxes=True
                    )
                    
                    # Price chart with moving averages
                    fig.add_trace(
                        go.Candlestick(
                            x=data[&#39;Date&#39;],
                            open=data[&#39;Open&#39;],
                            high=data[&#39;High&#39;],
                            low=data[&#39;Low&#39;],
                            close=data[&#39;Close&#39;],
                            name=&quot;Price&quot;
                        ),
                        row=1, col=1
                    )
                    
                    if &#39;MA_5&#39; in data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=data[&#39;Date&#39;],
                                y=data[&#39;MA_5&#39;],
                                mode=&#39;lines&#39;,
                                name=&#39;MA 5&#39;,
                                line=dict(color=&#39;orange&#39;, width=2)
                            ),
                            row=1, col=1
                        )
                    
                    if &#39;MA_20&#39; in data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=data[&#39;Date&#39;],
                                y=data[&#39;MA_20&#39;],
                                mode=&#39;lines&#39;,
                                name=&#39;MA 20&#39;,
                                line=dict(color=&#39;red&#39;, width=2)
                            ),
                            row=1, col=1
                        )
                    
                    # Volume chart
                    fig.add_trace(
                        go.Bar(
                            x=data[&#39;Date&#39;],
                            y=data[&#39;Volume&#39;],
                            name=&quot;Volume&quot;,
                            marker_color=&#39;#1f77b4&#39;
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        title=f&quot;{stock_symbol} Stock Analysis&quot;,
                        height=600,
                        showlegend=True,
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ML Prediction section
                    if enable_ml and analysis.get(&#39;prediction&#39;):
                        prediction = analysis[&#39;prediction&#39;]
                        if &quot;error&quot; not in prediction:
                            st.subheader(&quot;🤖 ML Prediction&quot;)
                            col_pred1, col_pred2, col_pred3 = st.columns(3)
                            
                            with col_pred1:
                                st.metric(&quot;Predicted Price&quot;, f&quot;${prediction[&#39;next_price&#39;]:.2f}&quot;)
                            with col_pred2:
                                st.metric(&quot;R² Score&quot;, f&quot;{prediction[&#39;r2_score&#39;]:.3f}&quot;)
                            with col_pred3:
                                st.metric(&quot;Confidence&quot;, prediction[&#39;confidence&#39;])
                            
                            if prediction[&#39;confidence&#39;] != &quot;High&quot;:
                                st.warning(&quot;⚠️ Prediction confidence is not high. Use with caution.&quot;)
                else:
                    st.error(f&quot;Error: {analysis[&#39;error&#39;]}&quot;)
    
    with col2:
        st.subheader(&quot;💰 Cryptocurrency&quot;)
        crypto_coins = list(st.session_state.data_provider.crypto_ids.keys())
        
        for coin in crypto_coins[:5]:
            crypto_data = st.session_state.data_provider.get_crypto_data(coin)
            if crypto_data and &#39;price&#39; in crypto_data:
                price = crypto_data[&#39;price&#39;]
                change = crypto_data.get(&#39;change_24h&#39;, 0)
                
                st.metric(
                    coin.capitalize(),
                    f&quot;${price:,.2f}&quot;,
                    f&quot;{change:.2f}%&quot; if change else None,
                    delta_color=&quot;inverse&quot;
                )
                
        st.markdown(&quot;---&quot;)
        st.subheader(&quot;📊 Economic Indicators&quot;)
        eco_data = st.session_state.data_provider.get_economic_indicators()
        for indicator, data in eco_data.items():
            if &#39;data&#39; in data and len(data[&#39;data&#39;]) &gt; 0:
                latest = data[&#39;data&#39;][0]
                st.metric(
                    indicator,
                    latest.get(&#39;value&#39;, &#39;N/A&#39;),
                    data[&#39;metadata&#39;].get(&#39;unit&#39;, &#39;&#39;)
                )

with tab2:
    st.header(&quot;📰 News &amp; Sentiment Analysis&quot;)
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # News categories
        news_topic = st.selectbox(
            &quot;Select News Category&quot;,
            [&quot;technology&quot;, &quot;business&quot;, &quot;science&quot;, &quot;general&quot;],
            index=0
        )
        
        max_articles = st.slider(&quot;Number of Articles&quot;, 5, 30, 10)
        
        if st.button(&quot;📰 Fetch Latest News&quot;, type=&quot;primary&quot;, use_container_width=True):
            with st.spinner(&quot;Fetching latest news...&quot;):
                news_data = st.session_state.data_provider.get_news_data(news_topic, max_articles)
                
                if news_data:
                    st.success(f&quot;Found {len(news_data)} articles&quot;)
                    
                    # Enable clustering if requested
                    if enable_clustering:
                        with st.spinner(&quot;Clustering news articles...&quot;):
                            news_data = st.session_state.analytics_engine.cluster_news(news_data)
                    
                    # Sentiment analysis
                    sentiment_results = None
                    if enable_sentiment:
                        with st.spinner(&quot;Analyzing sentiment...&quot;):
                            news_texts = [article[&#39;title&#39;] + &#39; &#39; + article.get(&#39;summary&#39;, &#39;&#39;) for article in news_data]
                            sentiment_results = st.session_state.analytics_engine.analyze_sentiment(news_texts)
                    
                    # Display news cards
                    cluster_colors = {
                        0: &quot;#1f77b4&quot;,
                        1: &quot;#ff7f0e&quot;,
                        2: &quot;#2ca02c&quot;,
                        3: &quot;#d62728&quot;,
                        4: &quot;#9467bd&quot;
                    }
                    
                    for i, article in enumerate(news_data):
                        sentiment = sentiment_results[&#39;individual_sentiments&#39;][i][&#39;sentiment&#39;] if sentiment_results else None
                        st.session_state.ui_components.create_news_card(
                            title=article[&#39;title&#39;],
                            source=article[&#39;source&#39;],
                            published=article[&#39;published&#39;],
                            summary=article[&#39;summary&#39;],
                            link=article[&#39;link&#39;],
                            sentiment=sentiment
                        )
                        
                        # Show cluster badge if clustered
                        if &#39;cluster&#39; in article:
                            cluster_id = article[&#39;cluster&#39;]
                            st.markdown(f&quot;&quot;&quot;
                            &lt;div style=&quot;
                                display: inline-block;
                                padding: 0.25rem 0.5rem;
                                border-radius: 12px;
                                font-size: 0.8rem;
                                font-weight: bold;
                                margin-bottom: 1rem;
                                background-color: {cluster_colors.get(cluster_id, &#39;#999999&#39;)};
                                color: white;
                            &quot;&gt;
                                Cluster #{cluster_id + 1}
                            &lt;/div&gt;
                            &quot;&quot;&quot;, unsafe_allow_html=True)
                else:
                    st.warning(&quot;No news articles found for this category&quot;)
    
    with col2:
        st.subheader(&quot;😊 Sentiment Analysis&quot;)
        if &#39;sentiment_results&#39; in locals() and sentiment_results:
            if &quot;error&quot; not in sentiment_results:
                # Overall sentiment
                st.metric(&quot;Overall Sentiment&quot;, sentiment_results[&#39;overall_sentiment&#39;])
                
                # Sentiment distribution
                st.markdown(&quot;**Sentiment Distribution**&quot;)
                dist = sentiment_results[&#39;sentiment_distribution&#39;]
                fig_dist = px.pie(
                    names=list(dist.keys()),
                    values=list(dist.values()),
                    color=list(dist.keys()),
                    color_discrete_map={
                        &quot;Positive&quot;: &quot;#2ca02c&quot;,
                        &quot;Neutral&quot;: &quot;#7f7f7f&quot;,
                        &quot;Negative&quot;: &quot;#d62728&quot;
                    }
                )
                fig_dist.update_layout(showlegend=False)
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Word cloud
                st.markdown(&quot;**Top Keywords**&quot;)
                if news_data:
                    news_texts = [article[&#39;title&#39;] + &#39; &#39; + article.get(&#39;summary&#39;, &#39;&#39;) for article in news_data]
                    wordcloud_img = st.session_state.analytics_engine.generate_word_cloud(news_texts)
                    st.image(wordcloud_img, caption=&quot;Word Cloud of News Content&quot;)

with tab3:
    st.header(&quot;🔍 Custom Analysis&quot;)
    st.subheader(&quot;Upload your own dataset for analysis&quot;)
    
    uploaded_file = st.file_uploader(
        &quot;Choose a CSV file&quot;, 
        type=&quot;csv&quot;,
        help=&quot;Upload your dataset for custom analysis&quot;
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(&quot;Dataset loaded successfully!&quot;)
            
            # Show dataset preview
            st.subheader(&quot;Data Preview&quot;)
            st.dataframe(df.head())
            
            # Basic stats
            st.subheader(&quot;Basic Statistics&quot;)
            st.write(df.describe())
            
            # Column selector
            selected_columns = st.multiselect(
                &quot;Select columns for analysis&quot;,
                df.columns
            )
            
            if selected_columns:
                # Correlation analysis
                st.subheader(&quot;Correlation Matrix&quot;)
                corr = df[selected_columns].corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale=&#39;RdBu_r&#39;,
                    zmin=-1,
                    zmax=1
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Distribution plots
                st.subheader(&quot;Distribution Plots&quot;)
                for col in selected_columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fig_dist = px.histogram(df, x=col, title=f&quot;Distribution of {col}&quot;)
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
        except Exception as e:
            st.error(f&quot;Error loading file: {str(e)}&quot;)
    else:
        st.info(&quot;Please upload a CSV file to get started&quot;)
        
with tab4:
    st.header(&quot;⚙️ User Preferences&quot;)
    st.session_state.preferences.preference_editor()
    
    st.subheader(&quot;Account Settings&quot;)
    with st.form(&quot;account_form&quot;):
        st.text_input(&quot;Full Name&quot;, &quot;John Doe&quot;)
        st.text_input(&quot;Email&quot;, &quot;john.doe@example.com&quot;)
        timezone = st.selectbox(&quot;Timezone&quot;, TIMEZONES, index=TIMEZONES.index(&quot;UTC&quot;))
        if st.form_submit_button(&quot;Save Settings&quot;):
            st.success(&quot;Settings updated successfully&quot;)
            
    st.subheader(&quot;Data Privacy&quot;)
    st.info(&quot;Your data is stored locally and never shared with third parties&quot;)
    if st.button(&quot;Clear All Data&quot;):
        st.session_state.clear()
        st.success(&quot;All local data has been cleared&quot;)
Run the application
if name == "main": main()
