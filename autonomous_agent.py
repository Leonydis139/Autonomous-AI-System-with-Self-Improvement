from core.db_manager import DBManager
from core.research import ddg_search
from utils.charts import lengths_chart

class AutonomousAgent:
    """
    Core agent for autonomous research and reporting.
    Uses a database cache, web search, and charting utilities.
    """
    def __init__(self):
        self.db = DBManager()

    def run(self, query: str):
        """
        Run a research query, using cache if available.
        Returns a (report, fig) tuple.
        """
        cached = self.db.get(query)
        if cached:
            return cached.get("report"), cached.get("fig")
        results = ddg_search(query)
        fig = lengths_chart(results)
        report = "\n".join("- " + r for r in results)
        self.db.set(query, {"report": report, "fig": fig})
        return report, fig
