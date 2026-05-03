# streamlit_app.py
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from agent import llm_with_tools, tool_map
import os
import sys
import shutil
import tempfile

# ============ Setup ============
def get_db_path():
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, "rag_faiss_db")

def get_upload_path():
    temp_dir = tempfile.gettempdir()
    upload_dir = os.path.join(temp_dir, "user_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir

DB_PATH = get_db_path()
UPLOAD_PATH = get_upload_path()

st.set_page_config(page_title="My RAG Agent", page_icon="🤖")
st.title("🤖 RAG Agent with Tools")

# ============ Sidebar: PDF Upload ============
with st.sidebar:
    st.header("📄 Document Manager")
    
    # File uploader
    uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "docx", "md"],
    accept_multiple_files=True,
    help="Upload PDFs, Word docs, text files, or Markdown files"
)
    
    
    # Process uploads button
    if uploaded_files and st.button("📥 Process documents", type="primary"):
        # Clear old uploads and DB
        if os.path.exists(UPLOAD_PATH):
            shutil.rmtree(UPLOAD_PATH)
        os.makedirs(UPLOAD_PATH, exist_ok=True)
        
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        
        # Save uploaded files
        for file in uploaded_files:
            file_path = os.path.join(UPLOAD_PATH, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            st.success(f"✅ Saved: {file.name}")
        
        # Trigger ingestion
        with st.spinner("⏳ Embedding documents..."):
            try:
                if "ingest" in sys.modules:
                    del sys.modules["ingest"]
                # Set env var so ingest.py knows where to find PDFs
                os.environ["PDF_SOURCE_DIR"] = UPLOAD_PATH
                import ingest
                st.success(f"🎉 Processed {len(uploaded_files)} document(s)!")
                st.session_state.docs_ready = True
                # Clear chat history when new docs uploaded
                st.session_state.messages = [
                    SystemMessage(content="You are a helpful assistant. Use tools when needed.")
                ]
                st.session_state.display = []
                # Force tools.py to reload retriever
                if "tools" in sys.modules:
                    del sys.modules["tools"]
                if "agent" in sys.modules:
                    del sys.modules["agent"]
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed: {e}")
                st.exception(e)
    
    # Show current docs
    st.divider()
    st.subheader("Current Documents")
    if os.path.exists(UPLOAD_PATH) and os.listdir(UPLOAD_PATH):
        for f in os.listdir(UPLOAD_PATH):
            st.write(f"📄 {f}")
    else:
        # Fall back to default sample.pdf
        st.info("Using default `data/sample.pdf`. Upload your own PDFs above.")

# ============ Initial ingestion (uses default PDF if none uploaded) ============
if "initial_ingestion_done" not in st.session_state:
    if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
        with st.spinner("⏳ Building knowledge base from default PDF..."):
            try:
                os.environ["PDF_SOURCE_DIR"] = "./data"
                if "ingest" in sys.modules:
                    del sys.modules["ingest"]
                import ingest
                st.session_state.initial_ingestion_done = True
            except Exception as e:
                st.error(f"❌ Initial ingestion failed: {e}")
                st.stop()
    else:
        st.session_state.initial_ingestion_done = True

# ============ Chat UI ============
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant. Use tools when needed.")
    ]
    st.session_state.display = []

for role, content in st.session_state.display:
    with st.chat_message(role):
        st.write(content)

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