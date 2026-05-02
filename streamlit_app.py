# streamlit_app.py
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from agent import llm_with_tools, tool_map
import os
import sys
import shutil

# Use writable directory on cloud
def get_db_path():
    """Return the same writable path as ingest.py and tools.py."""
    if os.path.exists("/tmp"):
        return "/tmp/chroma_db"
    return "./chroma_db"

DB_PATH = get_db_path()

# ============ Auto-ingest PDF on every cold start ============
if os.path.exists(DB_PATH):
    try:
        shutil.rmtree(DB_PATH)
    except (PermissionError, OSError):
        pass

st.warning("📚 Building vector database from PDF...")

if not os.path.exists("./data"):
    st.error("❌ 'data' folder not found in repo!")
    st.stop()

pdfs = [f for f in os.listdir("./data") if f.lower().endswith(".pdf")]
st.info(f"📄 Files in data/: {os.listdir('./data')}")
st.info(f"📑 PDFs detected: {pdfs}")
st.info(f"💾 Using DB path: {DB_PATH}")

if not pdfs:
    st.error("❌ No PDF files found in 'data' folder!")
    st.stop()

try:
    with st.spinner("⏳ Embedding PDF (1-2 min on first run)..."):
        if "ingest" in sys.modules:
            del sys.modules["ingest"]
        import ingest

    st.success("✅ Vector database built successfully!")

    import chromadb
    client = chromadb.PersistentClient(path=DB_PATH)
    collections = client.list_collections()
    if collections:
        count = collections[0].count()
        st.success(f"📊 {count} chunks loaded into vector database")

except Exception as e:
    st.error("❌ Ingestion failed with error:")
    st.exception(e)
    st.stop()

# ============ Streamlit UI Setup ============
st.set_page_config(page_title="My RAG Agent", page_icon="🤖")
st.title("🤖 RAG Agent with Tools")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant. Use tools when needed.")
    ]
    st.session_state.display = []

# Show chat history
for role, content in st.session_state.display:
    with st.chat_message(role):
        st.write(content)

# Chat input
if user_input := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.display.append(("user", user_input))
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ai_msg = llm_with_tools.invoke(st.session_state.messages)
            st.session_state.messages.append(ai_msg)

            if ai_msg.tool_calls:
                for call in ai_msg.tool_calls:
                    st.info(f"🔧 Using tool: `{call['name']}`")
                    result = tool_map[call["name"]].invoke(call["args"])
                    st.session_state.messages.append(
                        ToolMessage(content=str(result), tool_call_id=call["id"])
                    )
                ai_msg = llm_with_tools.invoke(st.session_state.messages)
                st.session_state.messages.append(ai_msg)

            st.write(ai_msg.content)
            st.session_state.display.append(("assistant", ai_msg.content))