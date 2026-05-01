from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.vectorstores import VectorStore

from .config import Settings
from .llm import initialize_llm
from .qa import setup_qa_chain
from .vector_db import load_or_create_vector_db


@dataclass(frozen=True)
class AIMethod:
    name: str
    method_type: str
    implementation: str
    purpose: str


@dataclass(frozen=True)
class DeepLearningLLMPipeline:
    llm: Any
    vector_db: VectorStore
    qa_chain: Any
    methods: tuple[AIMethod, ...]


def build_method_catalog(settings: Settings) -> tuple[AIMethod, ...]:
    return (
        AIMethod(
            name="Transformer sentence embeddings",
            method_type="deep_learning",
            implementation=settings.embeddings_model_name,
            purpose=(
                "Encode PDF chunks and user questions into dense semantic vectors "
                "for meaning-based retrieval."
            ),
        ),
        AIMethod(
            name="Vector similarity retrieval",
            method_type="deep_learning_retrieval",
            implementation=f"InMemoryVectorStore top-{settings.retrieval_k} search",
            purpose=(
                "Retrieve the most relevant mental-health document chunks before "
                "answer generation."
            ),
        ),
        AIMethod(
            name="Retrieval augmented generation",
            method_type="llm",
            implementation=settings.model_name,
            purpose=(
                "Ground the LLM response in retrieved document context instead of "
                "asking the model to answer from memory alone."
            ),
        ),
        AIMethod(
            name="Safety-aware prompting",
            method_type="llm_safety",
            implementation="JaneaAI compassionate mental-health prompt",
            purpose=(
                "Steer the LLM toward cautious, supportive answers and clear "
                "uncertainty when the retrieved context is incomplete."
            ),
        ),
    )


def build_deep_learning_llm_pipeline(settings: Settings) -> DeepLearningLLMPipeline:
    llm = initialize_llm(settings)
    vector_db = load_or_create_vector_db(settings)
    qa_chain = setup_qa_chain(
        vector_db=vector_db,
        llm=llm,
        search_k=settings.retrieval_k,
    )

    return DeepLearningLLMPipeline(
        llm=llm,
        vector_db=vector_db,
        qa_chain=qa_chain,
        methods=build_method_catalog(settings),
    )
