import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Tell the client explicitly to use Vertex AI and provide the project ID
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"), 
    vertexai=True, 
    project='907432642312', 
    location='us-central1'
)

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='How does AI work?',
)
print(response.text)
