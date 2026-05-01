from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    model_name: str
    embeddings_model_name: str
    persist_directory: Path
    pdf_source_path: Path
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def load_settings() -> Settings:
    # Load local .env for development convenience.
    load_dotenv(override=False)

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Missing GROQ_API_KEY environment variable.")

    # Backward compatible: prefer PDF_SOURCE_PATH, then PDF_FOLDER_PATH.
    pdf_source = os.getenv("PDF_SOURCE_PATH") or os.getenv("PDF_FOLDER_PATH", "./data")
    # Backward compatible: prefer VECTOR_DB_PATH, then CHROMA_DB_PATH.
    persist_directory = os.getenv("VECTOR_DB_PATH") or os.getenv(
        "CHROMA_DB_PATH", "./vector_db"
    )

    return Settings(
        groq_api_key=groq_api_key,
        model_name=os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b"),
        embeddings_model_name=os.getenv(
            "EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        persist_directory=Path(persist_directory),
        pdf_source_path=Path(pdf_source),
        chunk_size=_get_int("CHUNK_SIZE", 500),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 50),
        retrieval_k=_get_int("RETRIEVAL_K", 4),
    )
