# agent.py
import os
import streamlit as st
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from tools import all_tools

load_dotenv()

# 1. Create the LLM
#    Using flash-lite to stay under free-tier daily quota.
#    Switch back to "gemini-2.5-flash" if/when you have quota.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY"),
    temperature=0.7,
    max_tokens=500,
)

# 2. Bind the tools to the LLM
llm_with_tools = llm.bind_tools(all_tools)

# 3. Lookup so we can execute a tool by name
tool_map = {t.name: t for t in all_tools}


def run_agent(user_question: str, max_retries: int = 3) -> str:
    """Send a question to the LLM, run any tool calls it requests,
    then return the final answer. Includes retry logic for rate limits."""

    messages = [
        SystemMessage(content=(
    "You are a helpful assistant with access to tools.\n\n"
    "TOOL USAGE RULES:\n"
    "- Use tools when relevant: math → calculator, time → get_current_time, "
    "text manipulation → reverse_text/word_counter, "
    "questions about the user's PDF/documents → search_documents.\n"
    "- For ANY mention of 'my PDF', 'my document', 'my notes', or asking to "
    "summarize/explain/describe a document, ALWAYS call search_documents "
    "FIRST. Do not ask the user for clarification — just search with your "
    "best guess.\n"
    "- For general knowledge (facts, explanations, fun trivia), answer "
    "directly from your own knowledge without tools.\n"
    "- After a tool returns a result, ALWAYS write a clear natural-language "
    "reply that includes the result. Never return an empty response.\n"
    "- Never refuse a question just because no tool fits — fall back to "
    "your own knowledge."
)),
        HumanMessage(content=user_question),
    ]

    last_tool_result = None  # remember the last tool result for fallback

    for attempt in range(max_retries):
        try:
            # First pass: LLM decides whether to call a tool
            ai_msg = llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            # If the LLM wants to call tools, execute them
            if ai_msg.tool_calls:
                for call in ai_msg.tool_calls:
                    tool_name = call["name"]
                    tool_args = call["args"]
                    print(f"  -> Calling tool: {tool_name}({tool_args})")

                    last_tool_result = tool_map[tool_name].invoke(tool_args)

                    messages.append(
                        ToolMessage(
                            content=str(last_tool_result),
                            tool_call_id=call["id"],
                        )
                    )

                # Second pass: LLM uses the tool result to write the final answer
                ai_msg = llm_with_tools.invoke(messages)

            # Defensive fallback if the model returns empty content
            final = (ai_msg.content or "").strip()
            if not final and last_tool_result is not None:
                final = f"Result: {last_tool_result}"
            return final or "Sorry, I couldn't generate a response."

        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err or "429" in err:
                wait = 50  # free-tier per-minute window resets every ~60s
                print(f"  ⏳ Rate limit hit. Waiting {wait}s... "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            raise

    return "Failed after retries (likely daily quota exhausted)."


# 4. Try it out
if __name__ == "__main__":
    print("\n🤖 Agent ready! Ask anything (type 'quit' to exit)\n")

    while True:
        question = input("❓ You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        if not question:
            continue

        try:
            answer = run_agent(question)
            print(f"🤖 Agent: {answer}\n")
        except Exception as e:
            print(f"⚠️ Error: {e}\n")

def chat():
    """Interactive chat loop with memory across turns."""
    messages = [
        SystemMessage(content="You are a helpful assistant. Use tools when needed.")
    ]

    print("\n🤖 Agent ready! (type 'quit' to exit)\n")

    while True:
        user_input = input("❓ You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))

        try:
            ai_msg = llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            if ai_msg.tool_calls:
                for call in ai_msg.tool_calls:
                    print(f"  -> Calling tool: {call['name']}({call['args']})")
                    tool_result = tool_map[call["name"]].invoke(call["args"])
                    messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=call["id"])
                    )
                ai_msg = llm_with_tools.invoke(messages)
                messages.append(ai_msg)

            print(f"🤖 Agent: {ai_msg.content}\n")

        except Exception as e:
            print(f"⚠️ Error: {e}\n")


if __name__ == "__main__":
    chat()            