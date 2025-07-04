def main():
    # ... (existing setup code) ...

    # Responsive theme from user preferences
    theme = st.session_state.preferences.preferences.get('theme', 'light')
    st.markdown(f"""
    <style>
    body {{
        background-color: {'#181A1B' if theme == 'dark' else '#f0f2f6'};
    }}
    /* More theme CSS based on {theme} */
    </style>
    """, unsafe_allow_html=True)

    # ... (rest of your UI code) ...
