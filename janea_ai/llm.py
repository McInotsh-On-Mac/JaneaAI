from langchain_groq import ChatGroq

from .config import Settings


def initialize_llm(settings: Settings) -> ChatGroq:
    return ChatGroq(
        temperature=0,
        groq_api_key=settings.groq_api_key,
        model_name=settings.model_name,
    )
