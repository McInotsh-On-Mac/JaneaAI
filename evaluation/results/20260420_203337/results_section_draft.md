## Results

We evaluated retrieval, generation quality, and safety behavior on the JaneaAI mental-health dataset. Retrieval ablation compared chunking and embedding settings using source-page relevance as ground truth. The top retrieval configuration was **small_chunks_minilm_300_50**, achieving an MRR of **0.3542** with Recall@5 of **0.5**.

For LLM behavior, we compared RAG-enabled generation against an LLM-only baseline. RAG improved faithfulness (**2** vs **2**) and reduced hallucination rate (**1.0** vs **1.0**), while maintaining comparable relevance. This indicates that retrieval context materially improves factual grounding.

Safety evaluation used a mixed set of crisis and non-crisis prompts. The RAG method reached crisis detection recall of **1.0** and overall safety pass rate of **1.0**. These results support including explicit crisis-oriented handling in production and reporting both quality and safety metrics together, not quality alone.
