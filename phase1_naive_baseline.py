#!/usr/bin/env python3
"""
Phase 1 Naive Baseline RAG System & Evaluation Harness
======================================================
Principal AI Engineering Baseline Implementation.

Strictly adheres to Phase 1 Architecture:
1. Ingestion: Fixed-size character windowing (1000 chunk_size, 100 chunk_overlap)
   via RecursiveCharacterTextSplitter on SEC 10-K PDF. No layout/table/semantic parsing.
2. Indexing & Storage: ChromaDB local vector store with cosine distance (HNSW).
   Raw text chunks + standard chunk IDs only (no custom metadata filters).
   Dense vector embeddings via text-embedding-004.
3. Retrieval: Single-stage dense cosine similarity retrieval, hardcoded K=5.
4. Generation: Gemini 2.5 Flash (temperature=0.0) with zero-shot direct prompt stuffing.
   Concatenated chunks separated by '---' with strict negative constraint fallback:
   "I do not have sufficient information in the context to answer this".
5. Evaluation: Automated evaluation harness measuring:
   - Context Recall
   - Context Precision
   - Faithfulness
   Breakdown by category and overall exported to 'phase1_baseline_results.csv'.
"""

import os
import sys
import json
import time
import argparse
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Document loading and chunking
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store
import chromadb
from chromadb.config import Settings

# Google GenAI SDK (Vertex AI)
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Phase1NaiveRAG")


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class NaiveRAGConfig:
    project_id: str = "project-d6b77157-fe8f-48f4-afd"
    location: str = "us-central1"
    pdf_paths: List[str] = field(default_factory=lambda: [
        "data/_10-K-2025-As-Filed.pdf",
        "data/10-Q4-2024-As-Filed.pdf"
    ])
    chunk_size: int = 1000
    chunk_overlap: int = 100
    embedding_model: str = "text-embedding-004"
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.0
    top_k: int = 5
    chroma_db_dir: str = "./chroma_db"
    collection_name: str = "sec_10k_naive_baseline"
    testset_path: str = "golden_testset_50.json"
    output_csv_path: str = "phase1_baseline_results.csv"
    batch_embed_size: int = 50
    rate_limit_delay_seconds: float = 0.5


# ==============================================================================
# 1. Ingestion Pipeline
# ==============================================================================
class DocumentIngestionPipeline:
    """
    Standard naive ingestion pipeline.
    Uses PyPDFLoader for plain text extraction and fixed-size chunking.
    Explicitly avoids layout, table-aware, or semantic preprocessing.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def process_pdfs(self, pdf_paths: List[str]) -> List[str]:
        all_raw_chunks: List[str] = []
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                logger.warning(f"File not found: {pdf_path}. Skipping.")
                continue

            logger.info(f"Loading PDF document: {pdf_path} using PyPDFLoader...")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            logger.info(f"Loaded {len(pages)} pages from {pdf_path}.")

            splits = self.splitter.split_documents(pages)
            logger.info(f"Split {pdf_path} into {len(splits)} raw chunks.")

            for split in splits:
                text = split.page_content.strip()
                if text:
                    all_raw_chunks.append(text)

        logger.info(f"Ingestion complete. Total corpus chunks: {len(all_raw_chunks)}")
        return all_raw_chunks


# ==============================================================================
# 2. Embedding & Storage
# ==============================================================================
class VertexDenseEmbedder:
    """Generates dense vector embeddings using text-embedding-004 on Vertex AI."""
    def __init__(self, client: genai.Client, model_name: str = "text-embedding-004", batch_size: int = 50):
        self.client = client
        self.model_name = model_name
        self.batch_size = batch_size

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            retries = 3
            while retries > 0:
                try:
                    response = self.client.models.embed_content(
                        model=self.model_name,
                        contents=batch
                    )
                    for emb in response.embeddings:
                        embeddings.append(emb.values)
                    break
                except Exception as e:
                    retries -= 1
                    logger.warning(f"Embedding batch [{i}:{i+len(batch)}] failed ({e}). Retries left: {retries}")
                    if retries == 0:
                        raise e
                    time.sleep(2.0)
            time.sleep(0.1)
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=[query]
        )
        return response.embeddings[0].values


class ChromaVectorIndex:
    """
    ChromaDB vector database with Cosine Similarity space.
    Stores raw chunk text and standard IDs without custom metadata filters.
    """
    def __init__(self, db_dir: str, collection_name: str, embedder: VertexDenseEmbedder):
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.embedder = embedder
        self.chroma_client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def is_indexed(self) -> bool:
        return self.collection.count() > 0

    def get_count(self) -> int:
        return self.collection.count()

    def reset_collection(self) -> None:
        logger.info(f"Clearing collection: {self.collection_name}")
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def index_chunks(self, chunks: List[str]) -> None:
        logger.info(f"Indexing {len(chunks)} chunks into ChromaDB (Cosine similarity)...")
        batch_size = 100
        with tqdm(total=len(chunks), desc="Indexing into ChromaDB", unit="chunk") as pbar:
            for i in range(0, len(chunks), batch_size):
                batch_texts = chunks[i:i + batch_size]
                batch_ids = [f"chunk_{j:05d}" for j in range(i, i + len(batch_texts))]
                batch_embeddings = self.embedder.embed_texts(batch_texts)

                # Store only raw text and IDs (no metadata filters)
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_texts
                )
                pbar.update(len(batch_texts))

        logger.info(f"Vector indexing finished. Total stored chunks: {self.collection.count()}")

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        query_embedding = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        documents = results.get("documents", [[]])[0]
        return documents


# ==============================================================================
# 3. Naive Generation Chain (Direct Prompt Stuffing)
# ==============================================================================
class NaiveRAGChain:
    """
    Direct prompt-stuffing naive generation chain:
    - Exactly top-5 chunks retrieved via dense cosine similarity.
    - Chunks concatenated separated by '---'.
    - Zero-shot temperature=0.0 generation with strict negative constraint.
    """
    SYSTEM_PROMPT = (
        "You are an assistant answering questions based strictly on the provided context excerpts from SEC filings.\n"
        "Instructions:\n"
        "1. Answer the question using ONLY the information presented directly in the context below.\n"
        "2. Do NOT extrapolate, assume, or utilize outside knowledge.\n"
        "3. If the context does not contain the answer, you MUST state exactly:\n"
        "   \"I do not have sufficient information in the context to answer this\"\n"
        "4. Keep your answer factual, precise, and objective."
    )

    def __init__(self, client: genai.Client, index: ChromaVectorIndex, config: NaiveRAGConfig):
        self.client = client
        self.index = index
        self.config = config

    def retrieve(self, question: str) -> List[str]:
        return self.index.retrieve(question, top_k=self.config.top_k)

    def generate(self, question: str, retrieved_contexts: List[str]) -> str:
        # Concatenate 5 chunks separated by "---"
        context_str = "\n\n---\n\n".join(retrieved_contexts)
        user_prompt = (
            f"Context:\n"
            f"{context_str}\n\n"
            f"---\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        gen_config = types.GenerateContentConfig(
            temperature=self.config.llm_temperature,
            system_instruction=self.SYSTEM_PROMPT
        )

        retries = 3
        while retries > 0:
            try:
                response = self.client.models.generate_content(
                    model=self.config.llm_model,
                    contents=user_prompt,
                    config=gen_config
                )
                return response.text.strip() if response.text else "I do not have sufficient information in the context to answer this"
            except Exception as e:
                retries -= 1
                logger.warning(f"LLM generation failed: {e}. Retries left: {retries}")
                if retries == 0:
                    return f"ERROR: Generation failed ({e})"
                time.sleep(2.0)

        return "I do not have sufficient information in the context to answer this"

    def run(self, question: str) -> Tuple[str, List[str]]:
        contexts = self.retrieve(question)
        answer = self.generate(question, contexts)
        return answer, contexts


# ==============================================================================
# 4. Evaluation Harness (LLM-as-a-Judge: Context Recall, Precision, Faithfulness)
# ==============================================================================
class LLMJudgeEvaluator:
    """
    Self-contained, production-grade LLM-as-a-judge evaluation harness.
    Evaluates:
    1. Context Recall: Fraction of ground-truth factual statements attributed to context.
    2. Context Precision: Mean Average Precision of relevant chunks in retrieved top-K.
    3. Faithfulness: Fraction of claims made in answer supported directly by context.
    """
    def __init__(self, client: genai.Client, judge_model: str = "gemini-2.5-flash"):
        self.client = client
        self.judge_model = judge_model
        self.json_config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json"
        )

    def evaluate_context_recall(self, question: str, ground_truth: str, retrieved_contexts: List[str]) -> Tuple[float, str]:
        context_str = "\n\n---\n\n".join(retrieved_contexts)
        prompt = (
            "You are an expert evaluator assessing Context Recall in a Retrieval-Augmented Generation (RAG) system.\n"
            "Context Recall measures whether all information from the Ground Truth Answer is present in the Retrieved Context.\n\n"
            f"Question:\n{question}\n\n"
            f"Ground Truth Answer:\n{ground_truth}\n\n"
            f"Retrieved Context:\n{context_str}\n\n"
            "Instructions:\n"
            "1. Break down the Ground Truth Answer into atomic factual statements.\n"
            "2. For each statement, determine whether it can be directly attributed to or deduced from the Retrieved Context.\n"
            "3. Output valid JSON in this exact structure:\n"
            "{\n"
            "  \"statements\": [\n"
            "    {\"statement\": \"statement text\", \"attributed\": true, \"reason\": \"explanation\"}\n"
            "  ]\n"
            "}"
        )
        try:
            res = self.client.models.generate_content(
                model=self.judge_model,
                contents=prompt,
                config=self.json_config
            )
            data = json.loads(res.text)
            statements = data.get("statements", [])
            if not statements:
                return 1.0, "No statements in ground truth"
            attributed_count = sum(1 for s in statements if s.get("attributed") is True)
            recall = attributed_count / len(statements)
            reasons = "; ".join([f"[{s.get('attributed')}]: {s.get('statement')}" for s in statements[:3]])
            return recall, reasons
        except Exception as e:
            logger.error(f"Error evaluating context recall: {e}")
            return 0.0, f"Error: {e}"

    def evaluate_context_precision(self, question: str, ground_truth: str, retrieved_contexts: List[str]) -> Tuple[float, str]:
        numbered_chunks = "\n\n".join([f"[Chunk Rank {i+1}]:\n{c}" for i, c in enumerate(retrieved_contexts)])
        prompt = (
            "You are an expert evaluator assessing Context Precision in a Retrieval-Augmented Generation (RAG) system.\n"
            "Context Precision measures whether the chunks relevant to answering the question based on the ground truth are ranked higher.\n\n"
            f"Question:\n{question}\n\n"
            f"Ground Truth Answer:\n{ground_truth}\n\n"
            f"Retrieved Chunks:\n{numbered_chunks}\n\n"
            "Instructions:\n"
            "1. For each retrieved chunk (1 to 5), evaluate if it contains information directly useful to answering the question according to the ground truth.\n"
            "2. Output valid JSON in this exact structure:\n"
            "{\n"
            "  \"chunk_evaluations\": [\n"
            "    {\"chunk_rank\": 1, \"is_relevant\": true, \"reason\": \"explanation\"}\n"
            "  ]\n"
            "}"
        )
        try:
            res = self.client.models.generate_content(
                model=self.judge_model,
                contents=prompt,
                config=self.json_config
            )
            data = json.loads(res.text)
            evals = data.get("chunk_evaluations", [])
            relevance_flags = [e.get("is_relevant", False) for e in evals]

            # In case LLM returned fewer chunks
            while len(relevance_flags) < len(retrieved_contexts):
                relevance_flags.append(False)

            total_relevant = sum(1 for r in relevance_flags if r)
            if total_relevant == 0:
                return 0.0, "No relevant chunks retrieved"

            # Compute Mean Average Precision (MAP)
            cumulative_relevant = 0
            precisions = []
            for rank, is_rel in enumerate(relevance_flags, start=1):
                if is_rel:
                    cumulative_relevant += 1
                    precisions.append(cumulative_relevant / rank)

            precision_score = sum(precisions) / total_relevant if total_relevant > 0 else 0.0
            reasons = f"Relevant ranks: {[i+1 for i, r in enumerate(relevance_flags) if r]}"
            return precision_score, reasons
        except Exception as e:
            logger.error(f"Error evaluating context precision: {e}")
            return 0.0, f"Error: {e}"

    def evaluate_faithfulness(self, question: str, answer: str, retrieved_contexts: List[str]) -> Tuple[float, str]:
        # Handle strict negative constraint fallback
        if "i do not have sufficient information" in answer.lower():
            return 1.0, "Answer accurately rejected ungrounded generation"

        context_str = "\n\n---\n\n".join(retrieved_contexts)
        prompt = (
            "You are an expert evaluator assessing Faithfulness in a Retrieval-Augmented Generation (RAG) system.\n"
            "Faithfulness measures whether all claims in the Generated Answer are strictly supported by the Retrieved Context.\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved Context:\n{context_str}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            "Instructions:\n"
            "1. Extract all atomic factual claims made in the Generated Answer.\n"
            "2. For each claim, evaluate if it is directly verifiable from the Retrieved Context.\n"
            "3. Output valid JSON in this exact structure:\n"
            "{\n"
            "  \"claims\": [\n"
            "    {\"claim\": \"claim text\", \"supported\": true, \"reason\": \"explanation\"}\n"
            "  ]\n"
            "}"
        )
        try:
            res = self.client.models.generate_content(
                model=self.judge_model,
                contents=prompt,
                config=self.json_config
            )
            data = json.loads(res.text)
            claims = data.get("claims", [])
            if not claims:
                return 1.0, "No specific claims made"

            supported_count = sum(1 for c in claims if c.get("supported") is True)
            faithfulness = supported_count / len(claims)
            reasons = f"Supported: {supported_count}/{len(claims)} claims"
            return faithfulness, reasons
        except Exception as e:
            logger.error(f"Error evaluating faithfulness: {e}")
            return 0.0, f"Error: {e}"


# ==============================================================================
# 5. Main Execution Harness
# ==============================================================================
def run_evaluation(config: NaiveRAGConfig, reindex: bool = False, max_samples: Optional[int] = None):
    load_dotenv()
    logger.info("Initializing Vertex AI Client...")
    client = genai.Client(
        vertexai=True,
        project=config.project_id,
        location=config.location
    )

    embedder = VertexDenseEmbedder(
        client=client,
        model_name=config.embedding_model,
        batch_size=config.batch_embed_size
    )

    index = ChromaVectorIndex(
        db_dir=config.chroma_db_dir,
        collection_name=config.collection_name,
        embedder=embedder
    )

    # Ingestion & Indexing step
    if reindex or not index.is_indexed():
        logger.info("Building vector index for Naive Baseline...")
        index.reset_collection()
        ingestion = DocumentIngestionPipeline(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        raw_chunks = ingestion.process_pdfs(config.pdf_paths)
        if not raw_chunks:
            logger.error("No chunks extracted from PDF files! Exiting.")
            sys.exit(1)
        index.index_chunks(raw_chunks)
    else:
        logger.info(f"Using existing ChromaDB index with {index.get_count()} chunks.")

    # Ingest testset
    if not os.path.exists(config.testset_path):
        logger.error(f"Evaluation dataset not found at {config.testset_path}!")
        sys.exit(1)

    with open(config.testset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if max_samples:
        test_data = test_data[:max_samples]

    logger.info(f"Loaded {len(test_data)} evaluation test cases from {config.testset_path}.")

    rag_chain = NaiveRAGChain(client=client, index=index, config=config)
    evaluator = LLMJudgeEvaluator(client=client, judge_model=config.llm_model)

    records: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("STARTING PHASE 1 NAIVE BASELINE EVALUATION HARNESS")
    print("=" * 80 + "\n")

    eval_pbar = tqdm(enumerate(test_data, start=1), total=len(test_data), desc="Evaluating RAG Baseline", unit="sample")
    for idx, item in eval_pbar:
        question = item.get("question", "").strip()
        ground_truth = (
            item.get("ground_truth_answer") or
            item.get("ground_truth") or
            ""
        ).strip()
        category = (
            item.get("category") or
            item.get("evolution_type") or
            "General"
        ).strip()

        # 1. Retrieve top-5 chunks
        retrieved_contexts = rag_chain.retrieve(question)

        # 2. Generate naive answer
        generated_answer = rag_chain.generate(question, retrieved_contexts)

        # 3. Evaluate Metrics
        context_recall, recall_reason = evaluator.evaluate_context_recall(
            question, ground_truth, retrieved_contexts
        )
        context_precision, precision_reason = evaluator.evaluate_context_precision(
            question, ground_truth, retrieved_contexts
        )
        faithfulness, faithfulness_reason = evaluator.evaluate_faithfulness(
            question, generated_answer, retrieved_contexts
        )

        record = {
            "index": idx,
            "category": category,
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "context_recall": round(context_recall, 4),
            "context_precision": round(context_precision, 4),
            "faithfulness": round(faithfulness, 4),
            "recall_diagnostics": recall_reason,
            "precision_diagnostics": precision_reason,
            "faithfulness_diagnostics": faithfulness_reason,
            "retrieved_contexts": json.dumps(retrieved_contexts),
            "num_chunks_retrieved": len(retrieved_contexts)
        }
        records.append(record)

        # Update tqdm live status bar
        eval_pbar.set_postfix({
            "Cat": category[:10],
            "Recall": f"{context_recall:.2f}",
            "Prec": f"{context_precision:.2f}",
            "Faith": f"{faithfulness:.2f}"
        })

        time.sleep(config.rate_limit_delay_seconds)

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Save granular traces
    df.to_csv(config.output_csv_path, index=False)
    logger.info(f"Saved granular row-level traces to: {config.output_csv_path}")

    # Generate Metrics Breakdown
    metrics_cols = ["context_recall", "context_precision", "faithfulness"]
    summary_by_category = df.groupby("category")[metrics_cols].agg(["mean", "count"])

    # Flatten column names
    summary_by_category.columns = [f"{col}_{stat}" for col, stat in summary_by_category.columns]

    overall_means = df[metrics_cols].mean()
    overall_df = pd.DataFrame([overall_means], index=["OVERALL"])

    print("\n" + "=" * 80)
    print("PHASE 1 NAIVE BASELINE EVALUATION REPORT")
    print("=" * 80)
    print("\n--- METRICS BREAKDOWN BY QUESTION CATEGORY ---")
    print(df.groupby("category")[metrics_cols].mean().round(4).to_string())

    print("\n--- OVERALL SYSTEM AVERAGES ---")
    print(overall_df.round(4).to_string())
    print("\n" + "=" * 80)
    print(f"Detailed diagnostics exported to: {os.path.abspath(config.output_csv_path)}")
    print("=" * 80 + "\n")

    return df


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Naive Baseline RAG & Evaluation Harness for SEC 10-K Filings."
    )
    parser.add_argument(
        "--pdf-paths",
        nargs="+",
        default=["data/_10-K-2025-As-Filed.pdf", "data/10-Q4-2024-As-Filed.pdf"],
        help="Path(s) to SEC 10-K PDF filings."
    )
    parser.add_argument(
        "--testset",
        default="golden_testset_50.json",
        help="Path to evaluation test set JSON."
    )
    parser.add_argument(
        "--output-csv",
        default="phase1_baseline_results.csv",
        help="Output CSV path for granular evaluation traces."
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-chunking and re-indexing into ChromaDB."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of test set questions to evaluate (for testing)."
    )
    parser.add_argument(
        "--project-id",
        default="project-d6b77157-fe8f-48f4-afd",
        help="Google Cloud project ID for Vertex AI."
    )
    parser.add_argument(
        "--location",
        default="us-central1",
        help="Vertex AI region."
    )

    args = parser.parse_args()

    config = NaiveRAGConfig(
        project_id=args.project_id,
        location=args.location,
        pdf_paths=args.pdf_paths,
        testset_path=args.testset,
        output_csv_path=args.output_csv
    )

    run_evaluation(config, reindex=args.reindex, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
