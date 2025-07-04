import streamlit as st
import datetime

class EnhancedUIComponents:
    @staticmethod
    def set_theme():
        theme = st.session_state.preferences.preferences.get('theme', 'light')
        st.markdown(f"""
        <style>
        body {{
            background-color: {'#181A1B' if theme == 'dark' else '#f0f2f6'};
        }}
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_header():
        st.markdown("""
        <div class="main-header">
            <h1 style="margin:0;">📊 Enhanced Data Analytics Platform</h1>
            <p style="margin:0; opacity:0.9;">Real-time insights powered by AI • Free Forever</p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_sidebar(personas, timezones):
        st.markdown("""
        <div class="sidebar-section">
            <h3>🎛️ Control Panel</h3>
        </div>
        """, unsafe_allow_html=True)
        st.selectbox(
            "Select Your Role",
            personas,
            index=0,
            help="Choose your role for personalized experience"
        )
        st.markdown("""
        <div class="sidebar-section">
            <h4>🎨 Personalization</h4>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.preferences.preference_editor()

    @staticmethod
    def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal"):
        with st.container():
            st.metric(label=title, value=value, delta=delta, delta_color=delta_color)

    @staticmethod
    def create_news_card(title: str, source: str, published: datetime.datetime, summary: str, link: str, sentiment: str = None):
        time_ago = (datetime.datetime.now() - published).days
        time_text = f"{time_ago} days ago" if time_ago > 0 else "Today"
        st.markdown(f"""
        <div class="news-card" style="
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
            border-left: 4px solid #4CAF50;
            transition: all 0.3s ease;
        ">
            <h4 style="margin: 0 0 0.5rem 0;">{title}</h4>
            <div style="display: flex; justify-content: space-between; color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">
                <span>{source}</span>
                <span>{time_text}</span>
            </div>
            <p style="margin: 0.5rem 0 1rem 0;">{summary}</p>
            <a href="{link}" target="_blank" style="
                display: inline-block;
                background: #1f77b4;
                color: white;
                padding: 0.3rem 0.8rem;
                border-radius: 4px;
                text-decoration: none;
                font-weight: 500;
            ">Read Article</a>
        </div>
        """, unsafe_allow_html=True)
