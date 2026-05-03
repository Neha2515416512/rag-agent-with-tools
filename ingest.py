# ingest.py
import os
import shutil
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def get_db_path():
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, "rag_faiss_db")

DB_PATH = get_db_path()

# Read source directory from env var (set by streamlit_app.py)
DATA_DIR = os.environ.get("PDF_SOURCE_DIR", "./data")
print(f"📍 Source directory: {DATA_DIR}")

pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

if not pdfs:
    raise FileNotFoundError(f"No PDFs found in {DATA_DIR}")

print(f"📄 Found {len(pdfs)} PDF(s): {pdfs}")

# Load ALL PDFs (not just the first one)
all_documents = []
for pdf_file in pdfs:
    pdf_path = os.path.join(DATA_DIR, pdf_file)
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    # Add source filename to metadata so search results know which doc they came from
    for doc in docs:
        doc.metadata["source_file"] = pdf_file
    all_documents.extend(docs)
    print(f"  ✅ {pdf_file}: {len(docs)} pages")

print(f"📊 Total pages: {len(all_documents)}")

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
print(f"🎉 Saved {len(chunks)} chunks from {len(pdfs)} PDF(s) to {DB_PATH}")