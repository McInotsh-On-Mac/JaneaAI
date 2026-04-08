import gradio as gr

from .chatbot import JaneaChatbot


def launch_ui(chatbot: JaneaChatbot) -> None:
    interface = gr.ChatInterface(
        fn=chatbot.respond,
        title="JaneaAI",
        description="Ask me anything about mental health!",
    )
    interface.launch()
