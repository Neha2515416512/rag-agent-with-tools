# ingest.py
# Purpose: Load PDF → Chunk → Embed → Store in ChromaDB

import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Use writable directory: /tmp on cloud, ./chroma_db locally
def get_db_path():
    """Return a writable path for ChromaDB."""
    if os.path.exists("/tmp"):
        return "/tmp/chroma_db"
    return "./chroma_db"

DB_PATH = get_db_path()

# Auto-detect any PDF in data/ folder
DATA_DIR = "./data"
pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

if not pdfs:
    raise FileNotFoundError(f"No PDFs found in {DATA_DIR}")

pdf_path = os.path.join(DATA_DIR, pdfs[0])
print(f"📄 Loading PDF: {pdf_path}")

# Remove old DB if exists
if os.path.exists(DB_PATH):
    try:
        shutil.rmtree(DB_PATH)
        print(f"🗑️  Removed old DB at {DB_PATH}")
    except Exception as e:
        print(f"⚠️ Could not remove old DB: {e}")

# Load PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages")

# Chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

# Embed
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("🔢 Created embeddings model")

# Store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)

print(f"🎉 Ingestion complete! {len(chunks)} chunks saved to {DB_PATH}")