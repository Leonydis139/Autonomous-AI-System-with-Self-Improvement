PLUGIN_META = {
    "display_name": "Stepwise Quiz",
    "args": [
        {"name": "current_q", "type": "int", "label": "Current Question", "default": 0},
        {"name": "answers", "type": "list", "label": "Your Answers", "default": []}
    ],
    "description": "Test yourself with a multi-step quiz!",
    "questions": [
        {"q": "What is 2+2?", "a": "4"},
        {"q": "What is the capital of France?", "a": "paris"}
    ],
    "validator": lambda args: (True, ""),
    "state": {}
}

def run(current_q=0, answers=None, state=None, **kwargs):
    answers = answers or []
    questions = PLUGIN_META["questions"]
    if current_q < len(questions):
        return {"output": {
            "question": questions[current_q]["q"],
            "current_q": current_q,
            "answers": answers
        }, "_state": state}
    else:
        score = sum(
            str(ans).strip().lower() == questions[i]["a"]
            for i, ans in enumerate(answers)
        )
        return {"output": f"Quiz complete! Your score: {score}/{len(questions)}", "_state": state}
