import streamlit as st
import pandas as pd
from core.plugins import PluginManager

pm = PluginManager()
pm.load_plugins()
st.header("🧩 Plugins (Advanced & Interactive)")

plugin_list = pm.list_plugins()
if not plugin_list:
    st.info("No plugins found in 'plugins/' directory.")
else:
    plugin_names = [p["name"] for p in plugin_list]
    selected = st.selectbox("Choose plugin", plugin_names)
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
                args[name] = st.text_input(label, value=default)
            elif typ == "int":
                args[name] = st.number_input(label, value=default)
            elif typ == "list":
                args[name] = st.text_area(label, value=",".join(default) if default else "")
                args[name] = [x.strip() for x in args[name].split(",") if x.strip()]
            elif typ == "dataframe":
                file = st.file_uploader(label, type=["csv"])
                args[name] = pd.read_csv(file) if file else None
            elif typ == "column_selector":
                df = args.get("df")
                if df is not None:
                    args[name] = st.selectbox(label, list(df.columns))
                else:
                    args[name] = None
        else:
            args[arg] = st.text_input(f"Argument: {arg}")

    if st.button("Run Plugin"):
        if all(v is not None for v in args.values()):
            result = pm.execute_plugin(selected, **args)
            if isinstance(result, dict):
                # For stepwise plugins, show next question or result
                if "question" in result:
                    st.markdown(f"**Q:** {result['question']}")
                    user_ans = st.text_input("Your answer", key="quiz_ans")
                    current_q = result.get("current_q", 0)
                    answers = result.get("answers", [])
                    if st.button("Next Question"):
                        answers.append(user_ans)
                        st.session_state["plugin_args"] = dict(args, current_q=current_q+1, answers=answers)
                        st.experimental_rerun()
                else:
                    st.json(result)
            elif isinstance(result, str) and result.startswith("data:image/png;base64,"):
                st.image(result)
            else:
                st.write(result)
        else:
            st.warning("Please fill all arguments.")
