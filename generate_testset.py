import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field
from google import genai
import nest_asyncio

nest_asyncio.apply()
load_dotenv()

_vertex_client = None

def get_client():
    global _vertex_client
    if _vertex_client is None:
        _vertex_client = genai.Client(
            vertexai=True,
            project='project-d6b77157-fe8f-48f4-afd',
            location='us-central1',
        )
    return _vertex_client

class VertexGenAILLM(BaseChatModel):
    model_name: str = "gemini-2.5-flash" 

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager=None, **kwargs) -> ChatResult:
        contents = "\n".join([f"{m.type}: {m.content}" for m in messages])
        response = get_client().models.generate_content(
            model=self.model_name,
            contents=contents,
        )
        msg = AIMessage(content=response.text or "")
        return ChatResult(generations=[ChatGeneration(message=msg)])
        
    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager=None, **kwargs) -> ChatResult:
        # Awaitable wrapper for Ragas concurrent execution
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._generate(messages, stop, run_manager, **kwargs))

    @property
    def _llm_type(self) -> str:
        return "vertex-genai-llm"

class VertexGenAIEmbeddings(Embeddings):
    def __init__(self, model_name="text-embedding-004"):
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = get_client().models.embed_content(
            model=self.model_name,
            contents=texts,
        )
        return [emb.values for emb in response.embeddings]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is missing.")
        return

    print("Loading documents...")
    doc_paths = ["data/10-Q4-2024-As-Filed.pdf", "data/_10-K-2025-As-Filed.pdf"]
    documents = []
    for path in doc_paths:
        if os.path.exists(path):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        else:
            print(f"Warning: Could not find {path}")
        
    print(f"Loaded {len(documents)} pages from SEC filings.")
    if not documents: return

    print("Initializing models...")
    generator_llm = VertexGenAILLM(model_name="gemini-2.5-flash")
    critic_llm = VertexGenAILLM(model_name="gemini-2.5-flash")
    embeddings = VertexGenAIEmbeddings(model_name="text-embedding-004")

    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )

    distributions = {
        simple: 0.20,
        conditional: 0.25,
        reasoning: 0.30,
        multi_context: 0.25,
    }

    print("Generating 70 testset candidates... This will take a few minutes (API limits apply).")
    from ragas.run_config import RunConfig
    run_config = RunConfig(max_workers=4, max_retries=15, max_wait=120)
    try:
        testset = generator.generate_with_langchain_docs(
            documents,
            test_size=70,
            distributions=distributions,
            run_config=run_config
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        return

    print("Generation complete! Formatting and exporting...")
    df = testset.to_pandas()
    df.to_json("candidates_70.json", orient="records", indent=4)
    print(f"Saved {len(df)} candidates to candidates_70.json.")
    
    if len(df) >= 50:
        df.head(50).to_json("golden_testset_50.json", orient="records", indent=4)
        print("Saved exactly 50 validated questions to golden_testset_50.json.")
    else:
        df.to_json("golden_testset_50.json", orient="records", indent=4)
        print(f"Saved {len(df)} validated questions.")

if __name__ == "__main__":
    main()
