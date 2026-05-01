# JaneaAI

JaneaAI is an AI mental health chatbot with retrieval-augmented generation (RAG).

## Modular structure

The codebase is organized into focused modules:

- `janea_ai/config.py`: environment and runtime settings
- `janea_ai/deep_learning_llm_methods.py`: explicit deep learning and LLM method pipeline
- `janea_ai/llm.py`: LLM initialization
- `janea_ai/vector_db.py`: PDF ingestion and vector store
- `janea_ai/qa.py`: retrieval QA chain wiring
- `janea_ai/chatbot.py`: application orchestration
- `janea_ai/ui.py`: Gradio interface
- `main.py`: app entry point

## Deep learning and LLM methods

The main deep learning and LLM section is intentionally grouped in
`janea_ai/deep_learning_llm_methods.py` so the project methods are easy to
review in one file. The module builds and documents these methods:

- Transformer sentence embeddings with `sentence-transformers/all-MiniLM-L6-v2`
  by default. These neural embeddings convert document chunks and user questions
  into dense semantic vectors.
- Vector similarity retrieval over embedded PDF chunks. The retriever returns
  the top `RETRIEVAL_K` chunks for each user question.
- Retrieval augmented generation using the configured Groq-hosted LLM. Retrieved
  context is inserted into the prompt before the model generates an answer.
- Safety-aware prompting for mental-health conversations. The prompt tells the
  LLM to be supportive, grounded in retrieved context, and explicit about
  uncertainty.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a local `.env` file (one-time):

```bash
cp .env.example .env
```

Then edit `.env` and set:

```env
GROQ_API_KEY=your_api_key
```

Optional variables:

- `GROQ_MODEL_NAME` (default: `openai/gpt-oss-120b`)
- `EMBEDDINGS_MODEL_NAME` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `VECTOR_DB_PATH` (default: `./vector_db`)
- `CHROMA_DB_PATH` (backward compatible alias for `VECTOR_DB_PATH`)
- `PDF_SOURCE_PATH` (preferred; file or folder path, default: `./data`)
- `PDF_FOLDER_PATH` (backward compatible alias for `PDF_SOURCE_PATH`)
- `CHUNK_SIZE` (default: `500`)
- `CHUNK_OVERLAP` (default: `50`)
- `RETRIEVAL_K` (default: `4`)

Example PDF path in `.env`:

```env
PDF_SOURCE_PATH="/Users/sebastianmcintosh/Downloads/Coding/JenAI_App/data/Mental_Health_Document.pdf"
```

The app tracks source PDF metadata and automatically rebuilds `vector_db` when the source file(s), chunk settings, or embedding model change.

## Run

```bash
python3 main.py
```
