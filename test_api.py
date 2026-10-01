"""Quick test: Vertex AI genai client + embeddings"""
from google import genai

client = genai.Client(
    vertexai=True,
    project='project-d6b77157-fe8f-48f4-afd',
    location='us-central1',
)

# Test 1: LLM call
print("Test 1: LLM generation...")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in one sentence.",
)
print(f"  ✅ LLM response: {response.text}")

# Test 2: Embeddings
print("\nTest 2: Embeddings...")
response = client.models.embed_content(
    model="text-embedding-004",
    contents=["This is a test sentence."],
)
print(f"  ✅ Embedding length: {len(response.embeddings[0].values)}")

print("\n🎉 All tests passed! API is working.")
