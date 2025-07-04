import streamlit as st
import pytz

def render_user_prefs_tab():
    st.header("⚙️ User Preferences")
    st.session_state.preferences.preference_editor()
    st.subheader("Account Settings")
    with st.form("account_form"):
        st.text_input("Full Name", "John Doe")
        st.text_input("Email", "john.doe@example.com")
        timezone = st.selectbox("Timezone", pytz.all_timezones, index=pytz.all_timezones.index("UTC"))
        if st.form_submit_button("Save Settings"):
            st.success("Settings updated successfully")
    st.subheader("Data Privacy")
    st.info("Your data is stored locally and never shared with third parties")
    if st.button("Clear All Data"):
        st.session_state.clear()
        st.success("All local data has been cleared")
