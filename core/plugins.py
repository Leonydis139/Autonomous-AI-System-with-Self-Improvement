import importlib.util
import os
import sys
from typing import Any, Dict, List, Callable

class PluginManager:
    """
    Loads, validates, and executes plugins with metadata, persistent state, and UI description.
    Supports hot-reloading and advanced metadata for auto-generating UIs.
    """

    def __init__(self, plugin_path: str = "plugins"):
        self.plugin_path = plugin_path
        os.makedirs(self.plugin_path, exist_ok=True)
        self.plugins: List[Dict[str, Any]] = []
        self._plugin_state: Dict[str, Any] = {}
        self.load_plugins()

    def load_plugins(self):
        self.plugins.clear()
        sys.path.insert(0, self.plugin_path)
        for fname in os.listdir(self.plugin_path):
            if not fname.endswith(".py"):
                continue
            path = os.path.join(self.plugin_path, fname)
            module_name = fname[:-3]
            try:
                spec = importlib.util.spec_from_file_location(module_name, path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, "run") and hasattr(mod, "PLUGIN_META"):
                    # Initialize state if needed
                    if "state" in mod.PLUGIN_META:
                        self._plugin_state[module_name] = mod.PLUGIN_META["state"]
                    self.plugins.append({
                        "name": module_name,
                        "run": mod.run,
                        "meta": mod.PLUGIN_META,
                        "module": mod
                    })
            except Exception as e:
                print(f"[PluginLoader] Failed to load {fname}: {e}")

    def list_plugins(self) -> List[Dict[str, Any]]:
        return [{"name": p["name"], "meta": p["meta"]} for p in self.plugins]

    def get_plugin(self, name: str) -> Dict[str, Any]:
        for p in self.plugins:
            if p["name"] == name:
                return p
        raise ValueError("Plugin not found")

    def get_state(self, name: str) -> Any:
        return self._plugin_state.get(name, None)

    def set_state(self, name: str, state: Any):
        self._plugin_state[name] = state

    def execute_plugin(self, name: str, **kwargs) -> Any:
        plugin = self.get_plugin(name)
        validator: Callable = plugin["meta"].get("validator", None)
        state = self.get_state(name)
        if validator:
            valid, msg = validator(kwargs)
            if not valid:
                return {"error": f"Input validation failed: {msg}"}
        result = plugin["run"](**kwargs, state=state)
        if isinstance(result, dict) and "_state" in result:
            self.set_state(name, result["_state"])
            return result["output"]
        return result
