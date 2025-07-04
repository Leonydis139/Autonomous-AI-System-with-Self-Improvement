import concurrent.futures

class EnhancedLiveDataProvider:
    # ... previous code ...

    def get_news_data(self, topic: str = "technology", max_articles: int = 5) -> List[Dict]:
        sources = self.news_sources.get(topic.lower(), self.news_sources["general"])

        articles = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._get_news_from_source, src, max(3, max_articles//len(sources))) for src in sources[:3]]
            for future in concurrent.futures.as_completed(futures):
                try:
                    articles.extend(future.result())
                except Exception as e:
                    self.logger.warning(f"Parallel fetch error: {e}")

        articles = sorted(articles, key=lambda x: x['published'], reverse=True)[:max_articles]
        return articles
