# Autonomous AI System: Advanced Plugin Specification

## Plugin Structure

- Each plugin is a `.py` file in the `plugins/` directory.
- Must define:
    - `run(**kwargs, state=None)`: main logic. Use/update the `state` dict for persistence.
    - `PLUGIN_META` dict for UI and validation.

## PLUGIN_META Fields

- `display_name` (str): UI name
- `args` (list): Each arg is dict with:
    - `name`: argument name (str)
    - `type`: `text`, `int`, `list`, `dataframe`, `column_selector`
    - `label`: UI label (str)
    - `default`: default value
- `description` (str)
- `validator`: optional function(args) → (bool, str)
- `state` (dict, optional): persistent state (counter, settings, history, etc.)

## Stateful Plugin Example

```python
PLUGIN_META = {
    "display_name": "Persistent Counter",
    "args": [],
    "description": "Counts plugin runs.",
    "state": {"count": 0}
}
def run(state=None, **kwargs):
    state["count"] += 1
    return {"output": f"Run {state['count']}", "_state": state}
```

## Multi-step/Interactive Example

- Use `args` and logic in `run` to guide the user step by step.
- Store progress in `state` or as returned args.

---

**Security:**  
Plugins run in the main process—no file/network access unless explicitly coded. Validate all user input.

---

**Hot Reload:**  
Plugins are reloaded each run, no restart needed.

---

**To add ML, web, or notebook plugins—just drop them into `plugins/`!**
