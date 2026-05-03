import os
import shutil
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def get_db_path():
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, "rag_chroma_db")

DB_PATH = get_db_path()
print(f"📍 DB_PATH: {DB_PATH}")

DATA_DIR = "./data"
pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

if not pdfs:
    raise FileNotFoundError(f"No PDFs found in {DATA_DIR}")

pdf_path = os.path.join(DATA_DIR, pdfs[0])
print(f"📄 Loading PDF: {pdf_path}")

if os.path.exists(DB_PATH):
    try:
        shutil.rmtree(DB_PATH)
    except Exception as e:
        print(f"⚠️ {e}")

os.makedirs(DB_PATH, exist_ok=True)

loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH,
    collection_name="pdf_collection"
)

print(f"🎉 {len(chunks)} chunks saved to {DB_PATH}")