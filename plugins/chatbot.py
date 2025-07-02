PLUGIN_META = {
    "display_name": "Simple Chatbot",
    "args": [
        {"name": "user_message", "type": "text", "label": "Your message", "default": ""}
    ],
    "description": "Talk to a basic chatbot."
}

def run(user_message, state=None, **kwargs):
    user_message = user_message.strip().lower()
    if "hello" in user_message:
        return "Hello! How can I help you today?"
    elif "weather" in user_message:
        return "I can't check the weather, but it's always sunny in code!"
    elif "bye" in user_message:
        return "Goodbye! Have a great day!"
    else:
        return f"You said: {user_message}"
