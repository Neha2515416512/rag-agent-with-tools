from langchain_core.tools import tool
import datetime
import math
import os
import tempfile
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def get_db_path():
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, "rag_chroma_db")

DB_PATH = get_db_path()


@tool
def get_current_time() -> str:
    """Returns the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluates a math expression."""
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def word_counter(text: str) -> str:
    """Counts words and characters."""
    return f"Words: {len(text.split())}, Characters: {len(text)}"


@tool
def reverse_text(text: str) -> str:
    """Reverses a string."""
    return text[::-1]


_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

_vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=_embeddings,
    collection_name="pdf_collection"
)

_retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})


@tool
def search_documents(query: str) -> str:
    """Searches the user's PDF documents.
    Use this whenever the user asks about their PDF, document, or notes."""
    docs = _retriever.invoke(query)
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