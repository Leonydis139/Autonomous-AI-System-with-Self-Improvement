
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

# Optional imports for features; handle ImportError where used
try:
    from sklearn.ensemble import RandomForestRegressor as _RandomForestRegressor
    from sklearn.metrics import mean_squared_error as _mean_squared_error, r2_score as _r2_score
    from sklearn.model_selection import train_test_split as _train_test_split
    from sklearn.preprocessing import StandardScaler
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
class LiveDataProvider:
    """Enhanced live data provider with multiple sources"""
    
    def __init__(self):
        self.cache_manager = EnhancedDatabaseManager()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_stock_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Fetch real-time stock data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            logger.error(f"Stock data error for {symbol}: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=300)
    def get_crypto_data(self, symbol: str = "bitcoin") -> Dict:
        """Fetch cryptocurrency data from CoinGecko"""
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': symbol,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true'
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Crypto data error: {e}")
            return {}

    @st.cache_data(ttl=600)  # Cache for 10 minutes
    def get_news_data(self, topic: str = "technology", max_articles: int = 10) -> List[Dict]:
        """Fetch news data from multiple sources"""
        try:
            news_sources = [
                f"https://feeds.feedburner.com/oreilly/radar",
                f"https://rss.cnn.com/rss/edition.rss",
                f"https://feeds.bbci.co.uk/news/rss.xml"
            ]
            
            all_articles = []
            for source in news_sources:
                try:
                    feed = feedparser.parse(source)
                    for entry in feed.entries[:max_articles//len(news_sources)]:
                        all_articles.append({
                            'title': entry.title,
                            'link': entry.link,
                            'published': entry.get('published', ''),
                            'summary': entry.get('summary', ''),
                            'source': feed.feed.get('title', 'Unknown')
                        })
                except Exception as e:
                    logger.warning(f"Error fetching from {source}: {e}")
                    
            return all_articles[:max_articles]
        except Exception as e:
            logger.error(f"News data error: {e}")
            return []

    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def get_weather_data(self, city: str = "New York") -> Dict:
        """Fetch weather data from OpenWeatherMap"""
        try:
            api_key = os.environ.get('OPENWEATHER_API_KEY')
            if not api_key:
                return {"error": "OpenWeather API key not configured"}
            
            url = f"https://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': city,
                'appid': api_key,
                'units': 'metric'
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Weather data error: {e}")
            return {"error": str(e)}

    def get_economic_indicators(self) -> Dict:
        """Fetch economic indicators from FRED API"""
        try:
            fred_api_key = os.environ.get('FRED_API_KEY')
            if not fred_api_key:
                return {"error": "FRED API key not configured"}
            
            indicators = {
                'GDP': 'GDP',
                'Unemployment': 'UNRATE',
                'Inflation': 'CPIAUCSL',
                'Interest_Rate': 'FEDFUNDS'
            }
            
            data = {}
            for name, series_id in indicators.items():
                try:
                    url = f"https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        'series_id': series_id,
                        'api_key': fred_api_key,
                        'file_type': 'json',
                        'limit': 12,
                        'sort_order': 'desc'
                    }
                    response = self.session.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data[name] = response.json()
                except Exception as e:
                    logger.warning(f"Error fetching {name}: {e}")
                    
            return data
        except Exception as e:
            logger.error(f"Economic data error: {e}")
            return {"error": str(e)}

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

# ===================== ENHANCED ANALYTICS ENGINE =====================
class EnhancedAnalyticsEngine:
    """Enhanced analytics with ML capabilities"""
    
    def __init__(self):
        self.data_provider = LiveDataProvider()
        self.scaler = StandardScaler() if 'StandardScaler' in globals() else None

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
        """Analyze sentiment of text data"""
        try:
            if not texts:
                return {"error": "No text data provided"}

            sentiments = []
            for text in texts:
                try:
                    blob = TextBlob(text)
                    sentiments.append({
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'polarity': blob.sentiment.polarity,
                        'subjectivity': blob.sentiment.subjectivity
                    })
                except Exception as e:
                    logger.warning(f"Sentiment analysis error for text: {e}")

            if not sentiments:
                return {"error": "No sentiment data generated"}

            avg_polarity = np.mean([s['polarity'] for s in sentiments])
            avg_subjectivity = np.mean([s['subjectivity'] for s in sentiments])

            overall_sentiment = "Positive" if avg_polarity > 0.1 else "Negative" if avg_polarity < -0.1 else "Neutral"

            return {
                "overall_sentiment": overall_sentiment,
                "average_polarity": avg_polarity,
                "average_subjectivity": avg_subjectivity,
                "individual_sentiments": sentiments
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"error": str(e)}

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
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        .sidebar-header {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
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
                    
                    # Sentiment analysis
                    if enable_sentiment:
                        news_texts = [article['title'] + ' ' + article.get('summary',
