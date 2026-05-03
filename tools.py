# tools.py
from langchain_core.tools import tool
import datetime
import math
import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def get_db_path():
    """Return the same path as ingest.py."""
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, "rag_faiss_db")

DB_PATH = get_db_path()


@tool
def get_current_time() -> str:
    """Returns the current date and time. Use this when the user asks
    about the current time, today's date, or anything time-related."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluates a math expression and returns the result.
    Use this for any arithmetic, e.g. '2 + 2', '15 * 23', 'sqrt(144)'."""
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def word_counter(text: str) -> str:
    """Counts words and characters in text."""
    return f"Words: {len(text.split())}, Characters: {len(text)}"


@tool
def reverse_text(text: str) -> str:
    """Reverses a given string."""
    return text[::-1]


# ---- Vector store setup ----
_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Lazy-load: only load FAISS when search_documents is first called
_vectorstore = None
_retriever = None

def _get_retriever():
    """Load FAISS index lazily on first use."""
    global _vectorstore, _retriever
    if _retriever is None:
        if not os.path.exists(DB_PATH):
            return None
        _vectorstore = FAISS.load_local(
            DB_PATH,
            _embeddings,
            allow_dangerous_deserialization=True
        )
        _retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})
    return _retriever


@tool
def search_documents(query: str) -> str:
    """Searches the user's PDF documents stored in their knowledge base.
    
    USE THIS TOOL WHENEVER the user mentions:
    - "my PDF", "my document", "my notes", "my file"
    - "the document", "the PDF"
    - asks to "summarize", "explain", or describe what their document is about
    - asks any question whose answer might be inside their uploaded PDF
    
    For vague questions like "what is my PDF about?" or "summarize my document",
    pass a broad query like "main topic summary overview"."""
    retriever = _get_retriever()
    if retriever is None:
        return "Knowledge base not yet built. Please refresh the page."
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the documents."
    return "\n\n---\n\n".join(d.page_content for d in docs)


all_tools = [
    get_current_time,
    calculator,
    word_counter,
    reverse_text,
    search_documents,
]