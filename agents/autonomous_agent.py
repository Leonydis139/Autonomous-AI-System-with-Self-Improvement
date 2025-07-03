
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

import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        SystemMessage("You are a helpful assistant."),
        UserMessage("What is the capital of France?"),
    ],
    temperature=1.0,
    top_p=1.0,
    model=model
)

print(response.choices[0].message.content)

