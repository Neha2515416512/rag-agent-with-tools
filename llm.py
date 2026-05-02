# test_gemini.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load API key from .env
load_dotenv()

# Create the LLM client
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # was "gemini-1.5-flash" — that family is shut down
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
    max_tokens=500
)

# Build messages (system + user)
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="tell me a joke about programming")
]

# Get response using .invoke()
response = llm.invoke(messages)

# Print the answer
print(response.content)