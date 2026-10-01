import os
from google import genai
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"), vertexai=True, project='project-d6b77157-fe8f-48f4-afd', location='us-central1')
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='How does AI work?',
)
print(response.text)
