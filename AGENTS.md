# AGENTS.md

Guidance for AI coding agents (GitHub Copilot, Claude, Codex and similar) working in this repository. Read this before making changes.

## Project overview

A document Q&A assistant: users upload files, the app builds a FAISS vector index, and a Gemini model with LangChain tool calling decides whether to search the documents, use a utility tool, or answer directly. The UI is Streamlit.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your_key_here" > .env
```

Python 3.11 is the reference version (see `.devcontainer/devcontainer.json`).

## Commands

| Task | Command |
|---|---|
| Run the app | `streamlit run streamlit_app.py` |
| Rebuild the index from `./data` | `python ingest.py` |
| Test retrieval without the LLM | `python test_retrieval.py` |
| Check the Chroma store used by `app.py` | `python check_db.py` |
| Command-line RAG pipeline | `python app.py` |

## Code map

- `streamlit_app.py`: UI, file upload, triggers re-indexing, runs the chat loop
- `agent.py`: creates the Gemini LLM, binds tools, agent loop with rate-limit retries
- `tools.py`: all tools, defined with LangChain's `@tool` decorator
- `ingest.py`: loads PDF/DOCX/TXT/MD, chunks, embeds and saves the FAISS index
- `app.py` and `check_db.py`: separate CLI pipeline that uses ChromaDB in `./chroma_db`

## Conventions

- **Adding a tool:** add a function with `@tool` in `tools.py` and append it to `all_tools`. The docstring is what the model reads to decide when to call the tool, so describe *when* to use it, not only what it does. Update the routing rules in the system prompt in `agent.py` if needed.
- **Index path:** the FAISS index lives at `tempfile.gettempdir()/rag_faiss_db`. `ingest.py`, `tools.py` and `streamlit_app.py` each compute this path; change all three together or move it into one shared helper.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` is used for indexing and searching. If you change the model, rebuild the index, because old vectors will not match.
- **Chunking:** 1,000 characters with 150 overlap in `ingest.py`. Keep the `source_file` metadata on every chunk.
- **API key:** read from `.env` locally or `st.secrets` on Streamlit Cloud. Never hard-code keys, and never commit `.env` or `.streamlit/secrets.toml` (both are in `.gitignore`).

## Gotchas

- The model is `gemini-2.5-flash-lite` to stay within free-tier quotas. On HTTP 429 the agent waits about 50 seconds and retries up to 3 times; keep this behaviour when editing `run_agent`.
- The agent must always return a non-empty answer. Keep the fallback that returns the last tool result if the model replies with empty content.
- `calculator` uses `eval` with builtins removed and only `math` functions allowed. Do not widen what it can access; prefer replacing it with a safe expression parser.
- `FAISS.load_local` uses `allow_dangerous_deserialization=True`. Only load indexes this app created itself.

## Before you open a pull request

1. Run `python test_retrieval.py` and confirm relevant chunks come back.
2. Start the app and ask one document question and one calculator question.
3. Keep changes small and explain the user-facing effect in the PR description.
