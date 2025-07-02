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
        self.plugins.clear()
        sys.path.insert(0, os.path.abspath(self.plugin_path))  # <<-- FIXED HERE
        for fname in os.listdir(self.plugin_path):
            if not fname.endswith(".py"):
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
                        "meta": mod.PLUGIN_META
                    })
            except Exception as e:
                import streamlit as st
                st.error(f"[PluginLoader] Failed to load {fname}: {e}")

    def list_plugins(self):
        return [{"name": p["name"], "meta": p["meta"]} for p in self.plugins]
