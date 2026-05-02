# ingest.py
# Purpose: Load PDF → Chunk it → Embed → Store in ChromaDB
import shutil, os
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

from dotenv import load_dotenv

# At the start of ingest.py, before creating the vector store:
if os.path.exists("./chroma_db"):
    print("🗑️  Removing existing ChromaDB to avoid duplicates...")
    shutil.rmtree("./chroma_db")


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables (API keys) from .env
load_dotenv()

# ---------- STEP 1: LOAD DOCUMENT ----------
print("📄 Loading PDF...")
pdf_path = "data/sample.pdf"   # change to your file name
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages")


# ---------- STEP 2: CHUNKING ----------
# Why? LLMs have token limits. We split big text into small chunks.
print("✂️ Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # bigger chunks = more context per retrieval
    chunk_overlap=150,      # ~15% overlap, good rule of thumb
    separators=["\n\n", "\n", ". ", " ", ""],  # split on paragraph breaks first
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")


# ---------- STEP 3: EMBEDDING ----------
# Convert text chunks into numerical vectors (so computer can compare meaning)
# Using free HuggingFace model (no API cost)
print("🔢 Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------- STEP 4: STORE IN VECTOR DB ----------
# ChromaDB stores vectors locally on your computer
print("💾 Storing in ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"   # saves to disk
)

print("🎉 Ingestion complete! Vector DB saved at ./chroma_db")