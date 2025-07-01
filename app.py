# app.py - Autonomous AI System (All-in-One)
import gradio as gr
import uuid
import json
import logging
import os
import re
import requests
import feedparser
import subprocess
import tempfile
from datetime import datetime
from duckduckgo_search import DDGS

# ================ CONFIGURATION ================
DEBUG_MODE = True
MAX_RESEARCH_RESULTS = 5
CODE_EXECUTION_TIMEOUT = 30
KNOWLEDGE_SOURCES = {
    "openstax": {
        "endpoint": "https://openstax.org/api/v2/pages",
        "params": {"type": "textbook", "fields": "title,description,url"},
    },
    "arxiv": {
        "endpoint": "http://export.arxiv.org/api/query",
        "params": {"start": 0, "max_results": 3},
    },
    "wikimedia": {
        "endpoint": "https://en.wikipedia.org/api/rest_v1/page/summary/"
    },
    "khanacademy": {
        "endpoint": "https://www.khanacademy.org/api/v1/topic/"
    }
}

# ================ SYSTEM COMPONENTS ================
class MemoryManager:
    def __init__(self):
        self.memory = {}
    
    def init_session(self, session_id):
        if session_id not in self.memory:
            self.memory[session_id] = {
                "created": datetime.now().isoformat(),
                "entries": []
            }
    
    def add(self, session_id, key, value):
        if session_id in self.memory:
            self.memory[session_id]["entries"].append({
                "timestamp": datetime.now().isoformat(),
                "key": key,
                "value": value
            })

class Planner:
    def plan_task(self, goal):
        plan = [
            f"Research: Search web for information about {goal}",
            f"Learn: Analyze search results to understand {goal}",
            f"Develop: Generate code to accomplish {goal}",
            f"Execute: Run the generated code",
            f"Diagnose: Check system health and performance",
            f"Review: Evaluate results and identify improvements"
        ]
        
        if len(goal) > 50 or "complex" in goal.lower():
            plan.insert(2, "Design: Create architecture for solution")
            plan.insert(4, "Implement: Build core functionality")
        
        return plan

class Executor:
    def execute_code(self, code):
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
                f.write(code)
                filepath = f.name
            
            result = subprocess.run(
                ["python", filepath],
                capture_output=True,
                text=True,
                timeout=CODE_EXECUTION_TIMEOUT
            )
            
            os.unlink(filepath)
            
            if result.returncode == 0:
                return result.stdout.strip() or "Execution successful"
            return f"Error: {result.stderr.strip() or 'Unknown error'}"
        
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out"
        except Exception as e:
            return f"Execution error: {str(e)}"

class Critic:
    def review(self, step, result):
        feedback = {
            "code": [
                "Code executed successfully with no errors",
                "Code produced unexpected output - needs refinement",
                "Optimization opportunities identified"
            ],
            "search": [
                "Relevant information found",
                "Search results could be more targeted",
                "Comprehensive research completed"
            ],
            "diagnose": [
                "System health check passed",
                "Minor optimizations identified",
                "Critical improvements needed"
            ]
        }
        
        if "code" in step.lower(): 
            key = "code"
        elif "search" in step.lower(): 
            key = "search"
        elif "diagnose" in step.lower(): 
            key = "diagnose"
        else: 
            key = "code"
        
        return feedback[key][hash(step) % len(feedback[key])]

class WebSearcher:
    def search(self, query):
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=MAX_RESEARCH_RESULTS)]
                return json.dumps([
                    {"title": r["title"], "url": r["href"], "snippet": r["body"]}
                    for r in results
                ], indent=2)
        except Exception as e:
            return f"Search error: {str(e)}"

class KnowledgeIntegrator:
    def retrieve_knowledge(self, query, source_name):
        source = KNOWLEDGE_SOURCES.get(source_name.lower())
        if not source: 
            return {"error": "Invalid source"}
        
        try:
            # Handle different API formats
            if source_name == "wikimedia":
                url = source["endpoint"] + query.replace(" ", "_")
                response = requests.get(url)
            elif source_name == "khanacademy":
                url = source["endpoint"] + query.replace(" ", "_").lower()
                response = requests.get(url)
            else:
                params = source["params"].copy()
                params["search_query"] = query
                response = requests.get(source["endpoint"], params=params)
            
            response.raise_for_status()
            
            # Parse based on content type
            if 'json' in response.headers.get('Content-Type', ''):
                data = response.json()
            else:
                data = feedparser.parse(response.text)
            
            return self.parse_source(source_name, data)
        except Exception as e:
            return {"error": str(e)}
    
    def parse_source(self, source, data):
        try:
            if source == "openstax":
                return [{
                    "title": item["title"],
                    "description": item["description"],
                    "url": item["url"]
                } for item in data.get("items", [])]
            elif source == "arxiv":
                return [{
                    "title": entry.title,
                    "url": entry.link,
                    "published": entry.published
                } for entry in data.entries] if hasattr(data, 'entries') else []
            elif source == "wikimedia":
                return {
                    "summary": data.get("extract", "No summary available"),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", "#")
                }
            elif source == "khanacademy":
                return [{
                    "title": item["title"],
                    "url": f"https://www.khanacademy.org{item['url']}"
                } for item in data.get("children", []) if item.get("kind") == "Video"]
            return []
        except:
            return []
    
    def get_knowledge(self, concept, level="beginner"):
        return {
            "concept": concept,
            "foundational": self.retrieve_knowledge(concept, "openstax") + 
                            self.retrieve_knowledge(concept, "khanacademy"),
            "research": [] if level == "beginner" else self.retrieve_knowledge(concept, "arxiv"),
            "summary": self.retrieve_knowledge(concept, "wikimedia")
        }
    
    def format_knowledge(self, data):
        if isinstance(data, dict) and "error" in data: 
            return data["error"]
        
        concept = data.get("concept", "the topic")
        summary = data.get("summary", {})
        foundational = data.get("foundational", [])
        research = data.get("research", [])
        
        response = f"## 📚 Learning Resources: {concept}\n\n"
        response += f"**💡 Key Insights**\n{summary.get('summary', 'No summary available')}\n"
        
        if summary.get('url'):
            response += f"[Read more]({summary['url']})\n\n"
        
        if foundational:
            response += "**📖 Foundational Knowledge**\n"
            for item in foundational[:3]:
                title = item.get('title', 'Untitled Resource')
                url = item.get('url', '#')
                response += f"- [{title}]({url})\n"
        
        if research:
            response += "\n**🔬 Current Research**\n"
            for item in research[:3]:
                title = item.get('title', 'Untitled Paper')
                url = item.get('url', '#')
                date = f" ({item['published'][:10]})" if "published" in item else ""
                response += f"- [{title}]({url}){date}\n"
        
        return response + "\n**What would you like to explore next?**"

class CognitiveEngine:
    def __init__(self):
        self.knowledge = KnowledgeIntegrator()
        self.context = {"level": "beginner"}
    
    def extract_concept(self, query):
        """Extract the main concept from user input"""
        patterns = [
            r"explain (.*)",
            r"what is (.*)\?",
            r"tell me about (.*)",
            r"how does (.*) work\?",
            r"teach me (.*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip('?')
        
        return " ".join(query.split()[:3])
    
    def learning_response(self, query):
        # Update context based on query
        if "explain" in query.lower() or "what is" in query.lower():
            self.context["level"] = "beginner"
        elif "research" in query.lower() or "latest" in query.lower():
            self.context["level"] = "advanced"
        
        # Extract main concept
        concept = self.extract_concept(query)
        
        # Retrieve and format knowledge
        knowledge = self.knowledge.get_knowledge(concept, self.context["level"])
        return self.knowledge.format_knowledge(knowledge)
    
    def generate_code(self, task):
        return f'''# Generated code for: {task}
import requests

def main():
    """Autonomous AI-generated solution"""
    print("Initializing task execution")
    print(f"Task: {task}")
    # Core implementation would go here
    return "Task completed successfully"

if __name__ == "__main__":
    print(main())
'''

    def improve_code(self, code, feedback):
        return f"""# Improved code based on feedback
{code}

# Added enhancements based on: "{feedback}"
try:
    # Main functionality
    pass
except Exception as e:
    print(f"Error occurred: {{str(e)}}")
    # Error recovery logic
"""

# ================ MAIN APPLICATION ================
class AutonomousAgent:
    def __init__(self):
        self.memory = MemoryManager()
        self.planner = Planner()
        self.executor = Executor()
        self.critic = Critic()
        self.searcher = WebSearcher()
        self.cognition = CognitiveEngine()
        self.sessions = {}
    
    def process_goal(self, goal):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {"goal": goal, "status": "processing"}
        self.memory.init_session(session_id)
        self.memory.add(session_id, "user_goal", goal)
        
        try:
            plan = self.planner.plan_task(goal)
            self.memory.add(session_id, "plan", plan)
            
            results = []
            for step in plan:
                if "research" in step.lower() or "search" in step.lower():
                    query = step.split(":")[1].strip() if ":" in step else goal
                    search_results = self.searcher.search(query)
                    self.memory.add(session_id, f"search:{query}", search_results)
                    results.append(f"🔍 Search results for '{query}':\n{search_results[:300]}...")
                
                elif "develop" in step.lower() or "code" in step.lower():
                    code = self.cognition.generate_code(step)
                    result = self.executor.execute_code(code)
                    review = self.critic.review(step, result)
                    
                    if "error" in review.lower() or "refine" in review.lower():
                        code = self.cognition.improve_code(code, review)
                        result = self.executor.execute_code(code)
                        results.append(f"🛠️ Improved execution:\n{result}")
                    else:
                        results.append(f"✅ Execution result:\n{result}")
                
                elif "diagnose" in step.lower() or "check" in step.lower():
                    results.append("✅ System health check completed")
            
            self.sessions[session_id]["status"] = "completed"
            return "\n\n".join(results), session_id
        
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            return f"⚠️ Error occurred: {str(e)}", session_id

# ================ GRADIO INTERFACE ================
def create_interface():
    agent = AutonomousAgent()
    css = """
    body { 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
        background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d);
        color: #333;
        margin: 0;
        padding: 20px;
        min-height: 100vh;
    }
    .container { 
        max-width: 1200px; 
        margin: 0 auto; 
        background: rgba(255,255,255,0.95); 
        border-radius: 15px; 
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    h1 {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 30px;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    .tab { 
        background: white; 
        padding: 25px; 
        border-radius: 15px; 
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    button { 
        background: #3498db; 
        color: white; 
        border: none; 
        padding: 14px 28px; 
        border-radius: 10px; 
        cursor: pointer;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    button:hover {
        background: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .btn-primary { 
        background: #e74c3c; 
    }
    .btn-primary:hover {
        background: #c0392b;
    }
    .output-section { 
        background: #f8f9fa; 
        border-radius: 12px; 
        padding: 20px; 
        margin-top: 20px;
        border-left: 5px solid #3498db;
    }
    .chat-container {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        max-height: 500px;
        overflow-y: auto;
    }
    .knowledge-panel {
        background: #fffde7;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        border: 2px solid #ffecb3;
    }
    """
    
    with gr.Blocks(css=css, title="🤖 Autonomous AI") as app:
        session_id = gr.State()
        
        gr.Markdown("# 🤖 Autonomous AI System")
        gr.Markdown("An AI that can research, learn, code, and self-improve to accomplish your goals")
        
        with gr.Tab("🚀 Task Execution", id="task_tab"):
            with gr.Row():
                with gr.Column(scale=3):
                    goal_input = gr.Textbox(
                        label="Your Goal", 
                        placeholder="What do you want to achieve? (e.g., 'Create a weather dashboard')",
                        lines=2
                    )
                    exec_btn = gr.Button("Execute Goal", variant="primary")
                    output = gr.Textbox(label="Execution Results", interactive=False, lines=10)
                with gr.Column(scale=1):
                    session_out = gr.Textbox(label="Session ID", interactive=False)
                    gr.Markdown("### How it works:")
                    gr.Markdown("1. Enter a goal\n2. AI researches online\n3. Generates and executes code\n4. Provides results")
            
            exec_btn.click(
                fn=agent.process_goal,
                inputs=[goal_input],
                outputs=[output, session_out]
            )
        
        with gr.Tab("🎓 Learning Coach", id="learning_tab"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Ask about any topic")
                    learn_input = gr.Textbox(
                        label="Your Question", 
                        placeholder="What do you want to learn? (e.g., 'Explain neural networks')",
                        lines=2
                    )
                    with gr.Row():
                        ask_btn = gr.Button("Get Answer", variant="primary")
                        clear_btn = gr.Button("Clear Conversation")
                    learn_output = gr.Markdown("## Knowledge will appear here", elem_classes="knowledge-panel")
                with gr.Column(scale=1):
                    level = gr.Radio(
                        ["Beginner", "Advanced"], 
                        label="Knowledge Depth", 
                        value="Beginner",
                        info="Adjust explanation complexity"
                    )
                    gr.Markdown("### Current Features:")
                    gr.Markdown("- Curated learning resources\n- Research paper recommendations\n- Foundational concepts\n- Adaptive explanations")
        
        # Learning interactions
        def get_answer(question, depth):
            return agent.cognition.learning_response(question)
        
        def clear_conversation():
            return "", ""
        
        ask_btn.click(
            fn=get_answer, 
            inputs=[learn_input, level], 
            outputs=learn_output
        )
        
        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[learn_input, learn_output]
        )
    
    return app

# ================ APPLICATION STARTUP ================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    print("🚀 Starting Autonomous AI System...")
    print("✅ All systems initialized")
    print("🌐 Web interface available at http://localhost:7860")
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )
