from __future__ import annotations

from typing import Any, Optional

from .config import Settings, load_settings
from .llm import initialize_llm
from .qa import setup_qa_chain
from .vector_db import load_or_create_vector_db


class JaneaChatbot:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or load_settings()
        self.qa_chain = None

    def initialize(self) -> None:
        llm = initialize_llm(self.settings)
        vector_db = load_or_create_vector_db(self.settings)
        self.qa_chain = setup_qa_chain(vector_db, llm)

    def _format_history(self, history: Any, max_turns: int = 6) -> str:
        if not history:
            return ""

        lines: list[str] = []

        if isinstance(history, list) and history and isinstance(history[0], dict):
            for entry in history[-(max_turns * 2) :]:
                role = entry.get("role")
                content = entry.get("content", "")
                if role in {"user", "assistant"} and content:
                    speaker = "User" if role == "user" else "Assistant"
                    lines.append(f"{speaker}: {content}")
            return "\n".join(lines)

        if isinstance(history, list):
            for turn in history[-max_turns:]:
                if isinstance(turn, (tuple, list)) and len(turn) == 2:
                    user_text, assistant_text = turn
                    if user_text:
                        lines.append(f"User: {user_text}")
                    if assistant_text:
                        lines.append(f"Assistant: {assistant_text}")
            return "\n".join(lines)

        return ""

    def _build_query(self, message: str, history: Any) -> str:
        history_text = self._format_history(history)
        if not history_text:
            return message
        return (
            "Conversation history:\n"
            f"{history_text}\n\n"
            "Current user message:\n"
            f"{message}"
        )

    def respond(self, message: str, history) -> str:
        if self.qa_chain is None:
            return "Chatbot is initializing. Please wait..."
        try:
            query = self._build_query(message, history)
            result = self.qa_chain.invoke({"query": query})
            if isinstance(result, dict):
                return result.get("result", "")
            return str(result)
        except Exception as exc:
            return f"Error during query: {exc}"
