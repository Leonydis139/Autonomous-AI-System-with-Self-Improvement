
from core.db_manager import DBManager
from core.research import ddg_search
from utils.charts import lengths_chart
class AutonomousAgent:
    def __init__(self):
        self.db=DBManager()
    def run(self, query):
        cached=self.db.get(query)
        if cached:
            return cached['report'], cached['fig']
        res=ddg_search(query)
        fig=lengths_chart(res)
        rpt='
'.join('- '+r for r in res)
        self.db.set(query, {'report':rpt,'fig':fig})
        return rpt, fig
