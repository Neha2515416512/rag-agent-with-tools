import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from agent import llm_with_tools, tool_map
import os

# Auto-ingest PDF if vector DB doesn't exist (first run on cloud)
if not os.path.exists("./chroma_db"):
    with st.spinner("📚 Setting up knowledge base for the first time (1-2 min)..."):
        import ingest
    st.success("✅ Ready!")
st.set_page_config(page_title="My RAG Agent", page_icon="🤖")
st.title("🤖 RAG Agent with Tools")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant. Use tools when needed.")
    ]
    st.session_state.display = []  # what to show in the UI

# Show chat history
for role, content in st.session_state.display:
    with st.chat_message(role):
        st.write(content)

# Chat input
if user_input := st.chat_input("Ask me anything..."):
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.display.append(("user", user_input))
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Get response
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