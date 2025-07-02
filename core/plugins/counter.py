PLUGIN_META = {
    "display_name": "Persistent Counter",
    "args": [],
    "description": "Counts the number of times the plugin has been run.",
    "state": {"count": 0},
    "validator": lambda args: (True, "")
}

def run(state=None, **kwargs):
    state = state or {}
    state["count"] = state.get("count", 0) + 1
    return {"output": f"This plugin has been run {state['count']} times.", "_state": state}
