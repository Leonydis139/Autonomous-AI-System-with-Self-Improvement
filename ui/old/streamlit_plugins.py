import streamlit as st
import pandas as pd
from core.plugins import PluginManager

# Instantiate PluginManager once for the session
if "pm" not in st.session_state:
    st.session_state.pm = PluginManager()
pm = st.session_state.pm

def run_plugin_ui():
    pm.load_plugins()  # Hot-reload plugins each time
    plugin_list = pm.list_plugins()
    if not plugin_list:
        st.info("No plugins found in 'plugins/' directory.")
        return

    plugin_names = [p["name"] for p in plugin_list]
    selected = st.selectbox("Choose plugin", plugin_names, key="plugin_selector")
    plugin = pm.get_plugin(selected)
    meta = plugin["meta"]
    st.write(f"**{meta.get('display_name', selected)}**  \n{meta.get('description', '')}")

    args = {}
    for arg in meta.get("args", []):
        if isinstance(arg, dict):
            name, typ = arg["name"], arg.get("type", "text")
            label = arg.get("label", name)
            default = arg.get("default", "")
            if typ == "text":
                args[name] = st.text_input(label, value=default, key=f"{selected}_{name}")
            elif typ == "int":
                args[name] = st.number_input(label, value=default, key=f"{selected}_{name}")
            elif typ == "list":
                val = st.text_area(label, value=",".join(default) if default else "", key=f"{selected}_{name}")
                args[name] = [x.strip() for x in val.split(",") if x.strip()]
            elif typ == "dataframe":
                file = st.file_uploader(label, type=["csv"], key=f"{selected}_{name}")
                args[name] = pd.read_csv(file) if file else None
            elif typ == "column_selector":
                df = args.get("df")
                if df is not None:
                    args[name] = st.selectbox(label, list(df.columns), key=f"{selected}_{name}")
                else:
                    args[name] = None
        else:
            args[arg] = st.text_input(f"Argument: {arg}", key=f"{selected}_{arg}")

    if st.button("Run Plugin", key=f"{selected}_run"):
        if all(v is not None for v in args.values()):
            result = pm.execute_plugin(selected, **args)
            # Stepwise/interactive plugin support
            if isinstance(result, dict):
                if "question" in result:
                    st.markdown(f"**Q:** {result['question']}")
                    user_ans = st.text_input("Your answer", key=f"{selected}_quiz_ans")
                    current_q = result.get("current_q", 0)
                    answers = result.get("answers", [])
                    if st.button("Next Question", key=f"{selected}_next"):
                        answers.append(user_ans)
                        st.session_state[f"{selected}_quiz_state"] = dict(args, current_q=current_q+1, answers=answers)
                        st.experimental_rerun()
                else:
                    st.json(result)
            elif isinstance(result, str) and result.startswith("data:image/png;base64,"):
                st.image(result)
            else:
                st.write(result)
        else:
            st.warning("Please fill all arguments.")

# For direct execution/testing
if __name__ == "__main__":
    run_plugin_ui()
