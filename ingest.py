# ingest.py
# Purpose: Load PDF → Chunk → Embed → Store in FAISS

import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def get_db_path():
    """Return a writable path for FAISS index."""
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, "rag_faiss_db")

DB_PATH = get_db_path()
print(f"📍 DB_PATH: {DB_PATH}")

# Auto-detect any PDF in data/ folder
DATA_DIR = "./data"
pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

if not pdfs:
    raise FileNotFoundError(f"No PDFs found in {DATA_DIR}")

pdf_path = os.path.join(DATA_DIR, pdfs[0])
print(f"📄 Loading PDF: {pdf_path}")

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

# Build FAISS index from documents
vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
print(f"✅ Built FAISS index with {len(chunks)} chunks")

# Save to disk
os.makedirs(DB_PATH, exist_ok=True)
vectorstore.save_local(DB_PATH)
print(f"🎉 Ingestion complete! Saved to {DB_PATH}")