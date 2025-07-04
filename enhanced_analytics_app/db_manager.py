import os
import sqlite3
import threading
import datetime
import logging
from typing import Dict, Optional

CACHE_DB = "cache.db"

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedDatabaseManager:
    def __init__(self):
        self.pg_pool = None
        self.sqlite_lock = threading.Lock()
        self.init_databases()

    def get_connection(self):
        if self.pg_pool:
            try:
                return self.pg_pool.getconn(), "postgresql"
            except Exception as e:
                logger.warning(f"PostgreSQL connection failed: {e}")
        return sqlite3.connect(CACHE_DB), "sqlite"

    def init_databases(self):
        try:
            self.init_sqlite()
            self.init_postgresql()
        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    def init_sqlite(self):
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
        database_url = os.environ.get('DATABASE_URL')
        if database_url:
            try:
                from psycopg2 import pool
                self.pg_pool = pool.SimpleConnectionPool(1, 10, database_url)
                logger.info("PostgreSQL connection pool initialized")
            except ImportError:
                logger.warning("psycopg2 not installed, skipping PostgreSQL initialization")

    def return_connection(self, conn, db_type):
        try:
            if db_type == "postgresql" and self.pg_pool:
                self.pg_pool.putconn(conn)
            else:
                conn.close()
        except Exception as e:
            logger.error(f"Error returning connection: {e}")

    def get_cached_result(self, key: str) -> Optional[str]:
        conn, db_type = None, None
        try:
            conn, db_type = self.get_connection()
            cursor = conn.cursor()
            if db_type == "postgresql":
                cursor.execute(
                    '''SELECT value FROM cache WHERE key = %s AND expires_at > NOW()''', (key,))
            else:
                cursor.execute(
                    '''SELECT value FROM cache WHERE key = ? AND expires_at > datetime('now')''', (key,))
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None
        finally:
            if conn:
                self.return_connection(conn, db_type)

    def set_cached_result(self, key: str, value: str, ttl_minutes: int = 60):
        conn, db_type = None, None
        try:
            conn, db_type = self.get_connection()
            cursor = conn.cursor()
            expires_at = datetime.datetime.now() + datetime.timedelta(minutes=ttl_minutes)
            if db_type == "postgresql":
                cursor.execute(
                    '''
                    INSERT INTO cache (key, value, expires_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (key) DO UPDATE SET 
                    value = EXCLUDED.value, expires_at = EXCLUDED.expires_at
                    ''', (key, value, expires_at))
            else:
                cursor.execute(
                    '''
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
        conn, db_type = None, None
        try:
            conn, db_type = self.get_connection()
            cursor = conn.cursor()
            if db_type == "postgresql":
                cursor.execute(
                    '''
                    INSERT INTO analytics (user_id, action, details)
                    VALUES (%s, %s, %s)
                    ''', (user_id, action, details))
            else:
                cursor.execute(
                    '''
                    INSERT INTO analytics (user_id, action, details)
                    VALUES (?, ?, ?)
                    ''', (user_id, action, details))
            conn.commit()
        except Exception as e:
            logger.error(f"Analytics logging error: {e}")
        finally:
            if conn:
                self.return_connection(conn, db_type)
