import streamlit as st
from db_manager import EnhancedDatabaseManager
from live_data_provider import EnhancedLiveDataProvider
from analytics_engine import EnhancedAnalyticsEngine
from ui_components import EnhancedUIComponents
from user_preferences import UserPreferences
from modules.notifications import send_notification

import pytz

PERSONAS = [
    "Researcher", "Teacher", "Analyst", "Engineer", "Scientist",
    "Assistant", "Consultant", "Creative", "Problem Solver"
]
TIMEZONES = pytz.all_timezones

def main():
    st.set_page_config(
        page_title="Enhanced Data Analytics Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize stateful components
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

    st.session_state.ui_components.set_theme()

    st.session_state.ui_components.render_header()
    st.session_state.ui_components.render_sidebar(PERSONAS, TIMEZONES)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Market Dashboard",
        "📰 News & Sentiment",
        "🔍 Custom Analysis",
        "⚙️ User Preferences"
    ])
    from modules.market_dashboard import render_market_dashboard
    from modules.news_sentiment import render_news_sentiment
    from modules.custom_analysis import render_custom_analysis
    from modules.user_prefs_tab import render_user_prefs_tab

    with tab1:
        render_market_dashboard()
    with tab2:
        render_news_sentiment()
    with tab3:
        render_custom_analysis()
    with tab4:
        render_user_prefs_tab()

if __name__ == "__main__":
    main()
