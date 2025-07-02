import importlib.util
import os
import sys

class PluginManager:
    def __init__(self, plugin_path: str = "plugins"):
        self.plugin_path = plugin_path
        os.makedirs(self.plugin_path, exist_ok=True)
        self.plugins = []
        self._plugin_state = {}
        self.load_plugins()

    def load_plugins(self):
        """Load all valid plugin modules from the plugins directory."""
        self.plugins.clear()
        abs_plugin_path = os.path.abspath(self.plugin_path)
        if abs_plugin_path not in sys.path:
            sys.path.insert(0, abs_plugin_path)
        for fname in os.listdir(self.plugin_path):
            # Only load .py files, skip __init__.py and files starting with '_'
            if (
                not fname.endswith(".py")
                or fname == "__init__.py"
                or fname.startswith("_")
            ):
                continue
            path = os.path.join(self.plugin_path, fname)
            module_name = fname[:-3]
            try:
                spec = importlib.util.spec_from_file_location(module_name, path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "run") and hasattr(mod, "PLUGIN_META"):
                    self.plugins.append({
                        "name": module_name,
                        "run": mod.run,
                        "meta": mod.PLUGIN_META,
                        "module": mod
                    })
            except Exception as e:
                import streamlit as st
                st.error(f"[PluginLoader] Failed to load {fname}: {e}")

    def list_plugins(self):
        """List available plugins with their metadata."""
        return [{"name": p["name"], "meta": p["meta"]} for p in self.plugins]

    def get_plugin(self, name):
        """Get a plugin dict by its module name."""
        for plugin in self.plugins:
            if plugin["name"] == name:
                return plugin
        return None

    def execute_plugin(self, name, **kwargs):
        """
        Execute the plugin's run function.
        Passes kwargs to the run method.
        """
        plugin = self.get_plugin(name)
        if plugin is None:
            return f"Plugin '{name}' not found."
        try:
            return plugin["run"](**kwargs)
        except Exception as e:
            return f"Error running plugin '{name}': {e}"
