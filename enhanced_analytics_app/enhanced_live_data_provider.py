import concurrent.futures

class EnhancedLiveDataProvider:
    # ... (other methods) ...
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
