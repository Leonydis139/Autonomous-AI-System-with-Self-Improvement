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

# Download NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

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
    # ... (Omitting for brevity, see the original code above for full implementation)
    pass

# ===================== ENHANCED UI COMPONENTS =====================
class EnhancedUIComponents:
    # ... (Omitting for brevity, see the original code above for full implementation)
    pass

# ===================== ENHANCED ANALYTICS ENGINE =====================
class EnhancedAnalyticsEngine:
    # ... (Omitting for brevity, see the original code above for full implementation)
    pass

# ===================== USER PREFERENCES =====================
class UserPreferences:
    # ... (Omitting for brevity, see the original code above for full implementation)
    pass

# ===================== MAIN APPLICATION =====================
def main():
    # ... (Omitting for brevity, see the original code above for full implementation)
    pass

# Run the application
if __name__ == "__main__":
    main()
