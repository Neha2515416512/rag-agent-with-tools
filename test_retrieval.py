# test_retrieval.py
from tools import search_documents

# Call the tool directly, bypassing the LLM entirely
result = search_documents.invoke({"query": "main topic summary overview"})
print(result)
print("\n---\n")

result = search_documents.invoke({"query": "key points"})
print(result)
