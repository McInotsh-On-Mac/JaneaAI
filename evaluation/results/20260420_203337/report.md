# JaneaAI Evaluation Results

## Setup

- LLM model: `openai/gpt-oss-120b`
- Baseline embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- PDF source path: `/Users/sebastianmcintosh/JANEA/JaneaAI/data/pdfs`

## Retrieval Ablation

| Method | Recall@1 | Recall@3 | Recall@5 | MRR | Avg Latency (ms) | P95 Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| baseline_minilm_500_50 | 0.125 | 0.375 | 0.375 | 0.25 | 70.1358 | 138.4483 |
| small_chunks_minilm_300_50 | 0.25 | 0.5 | 0.5 | 0.3542 | 85.4009 | 114.9007 |
| large_chunks_minilm_800_100 | 0.125 | 0.375 | 0.375 | 0.2292 | 38.3039 | 43.0415 |
| bge_small_500_50 | 0.0 | 0.25 | 0.25 | 0.0833 | 64.2484 | 102.9273 |

## LLM Quality Evaluation

| Method | Avg Relevance (1-5) | Avg Faithfulness (1-5) | Avg Safety (1-5) | Hallucination Rate | Harmful Advice Rate | Avg Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| rag_default | 5 | 2 | 4 | 1.0 | 0.0 | 1396.8312 |
| llm_only | 5 | 2 | 4 | 1.0 | 0.0 | 664.9248 |

## Safety Prompt Evaluation

| Method | Precision | Recall | F1 | Harmful Pattern Rate | Safety Pass Rate |
|---|---:|---:|---:|---:|---:|
| rag_default | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| llm_only | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
