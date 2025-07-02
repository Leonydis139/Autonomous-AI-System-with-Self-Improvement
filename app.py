import streamlit as st
from ui import streamlit_plugins

st.set_page_config(
    page_title="🤖 Autonomous AI Plugin System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Autonomous AI Plugin System")
st.markdown("""
Welcome to the **Autonomous AI Plugin System**!  
- Easily run, test, and extend plugins for data, analytics, ML, chatbots, and more.
- Plugins are hot-reloadable: just drop a `.py` file in the `plugins/` directory—no restart needed!
- All UI is auto-generated from plugin metadata.  
- See the `plugins/README.md` for authoring guidelines.
""")

with st.sidebar:
    st.header("🔌 Plugin Manager")
    try:
        from ui.streamlit_plugins import pm
        plugins = pm.list_plugins()
        if plugins:
            st.write("Active plugins detected:")
            for plugin in plugins:
                st.write(f"- **{plugin['meta'].get('display_name', plugin['name'])}**")
        else:
            st.info("No plugins found in 'plugins/' directory.")
    except Exception as e:
        st.error(f"Could not load plugins: {e}")

st.divider()
st.subheader("🧩 Run a Plugin")

# Modularize the UI runner for easier upgrades
if hasattr(streamlit_plugins, "run_plugin_ui"):
    streamlit_plugins.run_plugin_ui()
else:
    st.error("Plugin UI runner not found. Please update ui/streamlit_plugins.py to include a `run_plugin_ui()` function.")

st.markdown("""
---
<small>
Built with ❤️ using [Streamlit](https://streamlit.io/).  
For plugin authoring guide, see `plugins/README.md`.
</small>
""", unsafe_allow_html=True)
