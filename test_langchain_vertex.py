import os
from langchain_google_vertexai import ChatVertexAI

try:
    llm = ChatVertexAI(model_name="gemini-1.5-flash", project="907432642312", location="us-central1", api_key=os.environ.get("GEMINI_API_KEY"))
    response = llm.invoke("Hello")
    print(response)
except Exception as e:
    print(f"Error: {e}")
