## Results

We evaluated retrieval, generation quality, and safety behavior on the JaneaAI mental-health dataset. Retrieval ablation compared chunking and embedding settings using source-page relevance as ground truth. The top retrieval configuration was **small_chunks_minilm_300_50**, achieving an MRR of **0.3542** with Recall@5 of **0.5**.

### Modular Deep Learning and LLM Methods

The implementation now exposes the deep learning and LLM methods in a dedicated module: `janea_ai/deep_learning_llm_methods.py`. This module makes the project methods clear during code review and groups the full AI pipeline in one place:

- **Transformer sentence embeddings:** PDF chunks and user questions are encoded with the configured Sentence Transformers model, defaulting to `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector similarity retrieval:** Embedded document chunks are searched with top-k semantic retrieval before answer generation.
- **Retrieval augmented generation:** Retrieved mental-health context is inserted into the LLM prompt so the response is grounded in source material.
- **Safety-aware prompting:** The LLM prompt instructs JaneaAI to be supportive, avoid guessing, and state uncertainty when the retrieved context is incomplete.

For LLM behavior, we compared RAG-enabled generation against an LLM-only baseline. On the small two-query generation set, RAG matched the LLM-only baseline on relevance (**5** vs **5**), faithfulness (**2** vs **2**), safety (**4** vs **4**), hallucination rate (**1.0** vs **1.0**), and harmful advice rate (**0.0** vs **0.0**). This means the current evaluation does not yet show a measurable quality gain from RAG, even though the RAG architecture is more appropriate for document-grounded question answering.

Safety evaluation used a mixed set of crisis and non-crisis prompts. The RAG method reached crisis detection recall of **1.0** and overall safety pass rate of **1.0**. These results support including explicit crisis-oriented handling in production and reporting both quality and safety metrics together, not quality alone.
