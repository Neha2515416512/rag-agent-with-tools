# tools.py
from langchain_core.tools import tool
import datetime
import math
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

@tool
def get_current_time() -> str:
    """Returns the current date and time. Use this when the user asks
    about the current time, today's date, or anything time-related."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluates a math expression and returns the result.
    Use this for any arithmetic, e.g. '2 + 2', '15 * 23', 'sqrt(144)'.
    Supports +, -, *, /, **, sqrt, sin, cos, log, pi, e."""
    try:
        # Safe-ish eval: only allow math functions
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def word_counter(text: str) -> str:
    """Counts the number of words and characters in a given text.
    Use this when the user wants to know the length of a sentence,
    paragraph, or any piece of text."""
    words = len(text.split())
    chars = len(text)
    return f"Words: {words}, Characters: {chars}"


@tool
def reverse_text(text: str) -> str:
    """Reverses a given string. Use this when the user asks to
    reverse, flip, or read backwards any text."""
    return text[::-1]

# ---- Vector store setup (loaded once when this file is imported) ----
# Must use the SAME embedding model as ingest.py
_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

_vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=_embeddings,
)

_retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})


@tool
def search_documents(query: str) -> str:
    """Searches the user's PDF documents stored in their knowledge base.
    
    USE THIS TOOL WHENEVER the user mentions:
    - "my PDF", "my document", "my notes", "my file"
    - "the document", "the PDF"
    - asks to "summarize", "explain", or describe what their document is about
    - asks any question whose answer might be inside their uploaded PDF
    
    For vague questions like "what is my PDF about?" or "summarize my document",
    pass a broad query like "main topic summary overview" to get general content.
    
    DO NOT ask the user for clarification before calling this tool — just call it
    with your best guess at the query. The tool will return the most relevant
    chunks from the documents."""
    docs = _retriever.invoke(query)
    if not docs:
        return "No relevant information found in the documents."
    return "\n\n---\n\n".join(d.page_content for d in docs)
# Export as a list so agent.py can import them all at once
all_tools = [
    get_current_time,
    calculator,
    word_counter,
    reverse_text,
    search_documents,   # ← add this
]