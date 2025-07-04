import requests
import datetime
import pandas as pd
import feedparser
import yfinance as yf
import pytz
import os
from typing import Dict, List, Optional
from dateutil import parser as date_parser
from bs4 import BeautifulSoup
import streamlit as st
import concurrent.futures

class EnhancedLiveDataProvider:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
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
    def get_stock_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data.reset_index() if not data.empty else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=300, show_spinner="Fetching crypto data...")
    def get_crypto_data(self, coin_name: str = "bitcoin") -> Dict[str, float]:
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
                'change_24h': data[coin_id].get('usd_24h_change', 0),
                'last_updated': datetime.datetime.now().isoformat()
            }
        except Exception:
            return {}

    def _get_news_from_source(self, source_url: str, max_items: int = 5) -> List[Dict]:
        try:
            feed = feedparser.parse(source_url)
            articles = []
            for entry in feed.entries[:max_items]:
                try:
                    published = date_parser.parse(entry.published) if hasattr(entry, 'published') else datetime.datetime.now()
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
                except Exception:
                    continue
            return articles
        except Exception:
            return []

    @st.cache_data(ttl=600, show_spinner="Fetching news...")
    def get_news_data(self, topic: str = "technology", max_articles: int = 5) -> List[Dict]:
        sources = self.news_sources.get(topic.lower(), self.news_sources["general"])
        all_articles = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda src: self._get_news_from_source(src, max(3, max_articles//len(sources))),
                sources[:3]
            ))
            for articles in results:
                all_articles.extend(articles)
        return sorted(
            all_articles,
            key=lambda x: x['published'],
            reverse=True
        )[:max_articles]

    @st.cache_data(ttl=1800, show_spinner="Checking weather...")
    def get_weather_data(self, location: str, api_key: Optional[str] = None) -> Dict:
        api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        if not api_key:
            return {"error": "API key not configured"}
        try:
            geo_url = "http://api.openweathermap.org/geo/1.0/direct"
            geo_params = {'q': location, 'limit': 1, 'appid': api_key}
            geo_response = self.session.get(geo_url, params=geo_params)
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            if not geo_data:
                return {"error": "Location not found"}
            lat, lon = geo_data[0]['lat'], geo_data[0]['lon']
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
            return {"error": str(e)}

    @st.cache_data(ttl=86400, show_spinner="Fetching economic data...")
    def get_economic_indicators(self, api_key: Optional[str] = None) -> Dict:
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
                results[key] = {'error': str(e), 'metadata': config}
        return results
