# check_db.py
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Does the folder even exist?
print("DB folder exists:", os.path.exists("./chroma_db"))

# How many documents are in it?
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vs = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
print("Number of docs in DB:", vs._collection.count())