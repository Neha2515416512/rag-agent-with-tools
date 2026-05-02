# app.py
# Purpose: User asks question → Retrieve relevant chunks → LLM answers
# Uses: Google Gemini (FREE LLM) + ChromaDB + HuggingFace Embeddings

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load API keys
load_dotenv()

# ---------- STEP 1: LOAD THE EXISTING VECTOR DB ----------
print("📂 Loading vector database...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# ---------- STEP 2: CREATE RETRIEVER ----------
# Retriever fetches the top-k most relevant chunks for a question
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}   # return top 3 most relevant chunks
)

# ---------- STEP 3: CONNECT TO LLM (Google Gemini - free & fast) ----------
print("🤖 Initializing LLM...")
llm = None

def get_llm():
    global llm
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,           # lower = more factual
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    return llm

# ---------- STEP 4: CREATE A PROMPT TEMPLATE ----------
# Tells the LLM HOW to use the retrieved context
prompt_template = """You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ---------- STEP 5: BUILD THE RAG CHAIN (LCEL) ----------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_qa_chain():
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | PROMPT
        | get_llm()
        | StrOutputParser()
    )

# ---------- STEP 6: ASK QUESTIONS IN A LOOP ----------
print("\n✅ RAG system ready! Type your questions (or 'quit' to exit)\n")

qa_chain = None
while True:
    question = input("❓ Your question: ")
    if question.lower() in ["quit", "exit", "q"]:
        print("👋 Goodbye!")
        break

    if not question.strip():
        continue

    try:
        if qa_chain is None:
            qa_chain = build_qa_chain()
        
        result = qa_chain.invoke(question)
        print("\n💡 Answer:", result)
        
        # Also show sources
        docs = retriever.invoke(question)
        print("\n📚 Sources used:")
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get('page', 'N/A')
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"  {i}. Page {page}: {preview}...")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 60)