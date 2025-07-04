import streamlit as st
import plotly.express as px

def render_news_sentiment():
    st.header("📰 News & Sentiment Analysis")
    analytics_engine = st.session_state.analytics_engine
    data_provider = st.session_state.data_provider
    ui_components = st.session_state.ui_components

    col1, col2 = st.columns([3, 1])
    with col1:
        news_topic = st.selectbox(
            "Select News Category",
            ["technology", "business", "science", "general"],
            index=0
        )
        max_articles = st.slider("Number of Articles", 5, 30, 10)
        enable_sentiment = st.checkbox("Enable Sentiment Analysis", value=True)
        enable_clustering = st.checkbox("Enable News Clustering", value=True)
        if st.button("📰 Fetch Latest News", type="primary", use_container_width=True):
            with st.spinner("Fetching latest news..."):
                news_data = data_provider.get_news_data(news_topic, max_articles)
                if news_data:
                    st.success(f"Found {len(news_data)} articles")
                    # Clustering
                    if enable_clustering:
                        with st.spinner("Clustering news articles..."):
                            news_data = analytics_engine.cluster_news(news_data)
                    # Sentiment
                    sentiment_results = None
                    if enable_sentiment:
                        with st.spinner("Analyzing sentiment..."):
                            news_texts = [article['title'] + ' ' + article.get('summary', '') for article in news_data]
                            sentiment_results = analytics_engine.analyze_sentiment(news_texts)
                    cluster_colors = {
                        0: "#1f77b4",
                        1: "#ff7f0e",
                        2: "#2ca02c",
                        3: "#d62728",
                        4: "#9467bd"
                    }
                    for i, article in enumerate(news_data):
                        sentiment = None
                        if sentiment_results and i < len(sentiment_results.get('individual_sentiments', [])):
                            sentiment = sentiment_results['individual_sentiments'][i]['sentiment']
                        ui_components.create_news_card(
                            title=article['title'],
                            source=article['source'],
                            published=article['published'],
                            summary=article['summary'],
                            link=article['link'],
                            sentiment=sentiment
                        )
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
                st.metric("Overall Sentiment", sentiment_results['overall_sentiment'])
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
                st.markdown("**Top Keywords**")
                if news_data:
                    news_texts = [article['title'] + ' ' + article.get('summary', '') for article in news_data]
                    wordcloud_img = analytics_engine.generate_word_cloud(news_texts)
                    st.image(wordcloud_img, caption="Word Cloud of News Content")
