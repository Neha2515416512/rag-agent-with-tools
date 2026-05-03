# ingest.py
import os
import shutil
import tempfile
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def get_db_path():
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, "rag_faiss_db")


def load_file(filepath):
    """Choose the right loader based on file extension."""
    ext = filepath.lower().split(".")[-1]
    
    if ext == "pdf":
        return PyPDFLoader(filepath).load()
    elif ext == "txt":
        return TextLoader(filepath, encoding="utf-8").load()
    elif ext == "docx":
        return Docx2txtLoader(filepath).load()
    elif ext == "md":
        return UnstructuredMarkdownLoader(filepath).load()
    else:
        print(f"⚠️ Unsupported file type: {ext}")
        return []


DB_PATH = get_db_path()

# Read source directory from env var (set by streamlit_app.py)
DATA_DIR = os.environ.get("PDF_SOURCE_DIR", "./data")
print(f"📍 Source directory: {DATA_DIR}")

# Find all supported files
SUPPORTED_EXTS = (".pdf", ".txt", ".docx", ".md")
files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(SUPPORTED_EXTS)]

if not files:
    raise FileNotFoundError(f"No supported files found in {DATA_DIR}")

print(f"📄 Found {len(files)} file(s): {files}")

# Load every supported file
all_documents = []
for file in files:
    file_path = os.path.join(DATA_DIR, file)
    try:
        docs = load_file(file_path)
        for doc in docs:
            doc.metadata["source_file"] = file
        all_documents.extend(docs)
        print(f"  ✅ {file}: {len(docs)} document(s)")
    except Exception as e:
        print(f"  ❌ Failed to load {file}: {e}")

if not all_documents:
    raise RuntimeError("No documents could be loaded from the supported files")

print(f"📊 Total documents: {len(all_documents)}")

# Chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = text_splitter.split_documents(all_documents)
print(f"✅ Created {len(chunks)} chunks")

# Embed
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Build FAISS index
vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

# Save
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
os.makedirs(DB_PATH, exist_ok=True)
vectorstore.save_local(DB_PATH)
print(f"🎉 Saved {len(chunks)} chunks from {len(files)} file(s) to {DB_PATH}")