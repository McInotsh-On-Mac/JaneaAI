from janea_ai.chatbot import JaneaChatbot
from janea_ai.ui import launch_ui

def main() -> None:
    try:
        chatbot = JaneaChatbot()
        chatbot.initialize()
    except ValueError as exc:
        raise SystemExit(
            f"{exc}\nSet it with: export GROQ_API_KEY='your_new_key_here'"
        ) from exc
    launch_ui(chatbot)


if __name__ == "__main__":
    main()
