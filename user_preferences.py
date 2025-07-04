import streamlit as st
import pytz

TIMEZONES = pytz.all_timezones

class UserPreferences:
    def __init__(self):
        self.preferences = {
            "theme": "light",
            "timezone": "UTC",
            "default_stock": "AAPL",
            "default_crypto": "bitcoin",
            "default_news_category": "technology",
            "refresh_interval": 5
        }
        self.crypto_options = ["bitcoin", "ethereum", "cardano", "solana", "dogecoin"]

    def load_preferences(self, user_id: str):
        if 'preferences' in st.session_state:
            self.preferences = st.session_state.preferences
        else:
            st.session_state.preferences = self.preferences

    def save_preferences(self, user_id: str):
        st.session_state.preferences = self.preferences

    def preference_editor(self):
        with st.expander("⚙️ User Preferences"):
            theme = self.preferences['theme']
            timezone = self.preferences['timezone']
            default_stock = self.preferences['default_stock']
            default_crypto = self.preferences['default_crypto']
            default_news_category = self.preferences['default_news_category']
            refresh_interval = self.preferences['refresh_interval']
            self.preferences['theme'] = st.selectbox(
                "Theme",
                ["light", "dark", "blue"],
                index=["light", "dark", "blue"].index(theme)
            )
            self.preferences['timezone'] = st.selectbox(
                "Timezone",
                TIMEZONES,
                index=TIMEZONES.index(timezone)
            )
            self.preferences['default_stock'] = st.text_input(
                "Default Stock",
                value=default_stock
            )
            self.preferences['default_crypto'] = st.selectbox(
                "Default Crypto",
                self.crypto_options,
                index=self.crypto_options.index(default_crypto)
            )
            self.preferences['default_news_category'] = st.selectbox(
                "Default News Category",
                ["technology", "business", "science"],
                index=["technology", "business", "science"].index(default_news_category)
            )
            self.preferences['refresh_interval'] = st.slider(
                "Refresh Interval (min)",
                1, 60, refresh_interval
            )
            if st.button("Save Preferences"):
                self.save_preferences("current_user")
                st.success("Preferences saved!")
