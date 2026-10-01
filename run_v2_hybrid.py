#!/usr/bin/env python3
"""
================================================================================
Phase 2 Hybrid Search Architecture: Dense-Sparse Reciprocal Rank Fusion (RRF)
================================================================================
Principal AI Engineering Implementation for SEC 10-K RAG Systems.

Architectural Objectives:
-------------------------
1. Overcome "Lost in the Middle" and Context Precision Failures:
   Dense embeddings compress semantics into fixed-dimensional vectors, losing
   fine-grained lexical precision on financial tables, coupon percentages (e.g.,
   '0.875% Notes due 2025'), and precise dates. By coupling dense semantic
   search with exact BM25 keyword matching, we ensure critical numerical anchors
   are not buried or lost.

2. Ingestion & Advanced Chunking:
   - Chunk size upgraded from Phase 1 naive baseline (1000/100) to 1500 chars with
     200 overlap.
   - Semantic boundary and table-aware separators preserve financial schedules,
     preventing table headers from being severed from numerical rows.

3. Dual-Retrieval Stream Setup:
   - Stream A (Dense Retriever): Chroma vector store with text-embedding-004.
     Retrieves top k=10 semantically relevant candidate documents.
   - Stream B (Sparse Retriever): BM25Retriever over the exact same document
     pool. Retrieves top k=10 lexically matching candidate documents.

4. Reciprocal Rank Fusion (RRF) via LangChain EnsembleRetriever:
   - Merges candidate lists using rank reciprocal scoring:
       RRF_Score(d) = sum_{r in {dense, sparse}} ( weight_r / (k_rrf + rank_r(d)) )
     where k_rrf = 60, weights = [0.5, 0.5].
   - Documents matching BOTH dense and sparse rank at the very top.
   - Slices top-5 candidates for zero-shot prompt-stuffing generation.

5. Generation & Evaluation Pipeline:
   - Exact zero-shot negative constraint prompt container from Phase 1.
   - Generator: Gemini 2.5 Flash (temperature=0.0).
   - Evaluation: Ragas metrics (Context Recall, Context Precision, Faithfulness)
     measured against golden_testset_50.json.
   - Comparative analytics: Side-by-side Pandas DataFrame comparing Phase 1 vs Phase 2.
================================================================================
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

# LangChain Document loaders & Splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Embeddings & Vector Stores
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

# Dual Retrieval & Reciprocal Rank Fusion (RRF)
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Google GenAI SDK (Vertex AI)
from google import genai
from google.genai import types

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Phase2HybridRAG")


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class HybridRAGConfig:
    project_id: str = "project-d6b77157-fe8f-48f4-afd"
    location: str = "us-central1"
    pdf_paths: List[str] = field(default_factory=lambda: [
        "data/_10-K-2025-As-Filed.pdf",
        "data/10-Q4-2024-As-Filed.pdf"
    ])
    # Advanced chunking: larger window preserving financial tables
    chunk_size: int = 1500
    chunk_overlap: int = 200
    embedding_model: str = "text-embedding-004"
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.0

    # Dual-Retrieval specifications
    dense_top_k: int = 10
    sparse_top_k: int = 10
    ensemble_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])
    rrf_c_constant: int = 60
    final_top_k: int = 5  # Top-K documents fed to generator LLM

    # Storage paths
    chroma_db_dir: str = "./chroma_db"
    collection_name: str = "sec_10k_phase2_hybrid"
    testset_path: str = "golden_testset_50.json"
    phase1_results_path: str = "phase1_baseline_results.csv"
    output_csv_path: str = "phase2_hybrid_results.csv"
    comparison_csv_path: str = "phase2_vs_phase1_comparison.csv"
    batch_embed_size: int = 50
    rate_limit_delay_seconds: float = 0.5


# ==============================================================================
# 1. Ingestion Pipeline & Advanced Chunking Strategy
# ==============================================================================
class AdvancedDocumentIngestionPipeline:
    """
    Ingests SEC 10-K filings using advanced table-aware & semantic boundary splitting.
    
    Why this upgrade:
    Financial filings contain complex tabular schedules (segment net sales, notes,
    balance sheets) spanning 1,200 to 1,500 characters. Phase 1's 1000-char naive splitting
    amputated table headers from row values. Upgrading to chunk_size=1500 with 200 overlap
    and structured separators preserves table row continuity.
    """
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Semantic boundary separators prioritising section and table boundaries
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n\n",               # Multi-line section headers
                "\n\n",                 # Paragraph boundaries
                "\n(?=[A-Z0-9\$\(])",   # Table row or itemized bullet starts
                "\n",                   # Line breaks
                " ",                    # Word boundaries
                ""                      # Character fallback
            ],
            is_separator_regex=True
        )

    def process_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        all_chunks: List[Document] = []
        global_chunk_idx = 0

        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                logger.warning(f"File not found: {pdf_path}. Skipping.")
                continue

            logger.info(f"Loading PDF document: {pdf_path}...")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            logger.info(f"Loaded {len(pages)} pages from {pdf_path}.")

            splits = self.splitter.split_documents(pages)
            logger.info(f"Split {pdf_path} into {len(splits)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap}).")

            for split in splits:
                text = split.page_content.strip()
                if not text:
                    continue
                # Attach consistent chunk identifiers and structural metadata
                metadata = dict(split.metadata)
                metadata["chunk_id"] = f"chunk_{global_chunk_idx:05d}"
                metadata["char_length"] = len(text)
                
                doc = Document(page_content=text, metadata=metadata)
                all_chunks.append(doc)
                global_chunk_idx += 1

        logger.info(f"Advanced Ingestion complete. Total indexed chunks: {len(all_chunks)}")
        return all_chunks


# ==============================================================================
# 2. Dense Vector Embeddings (Vertex AI Adapter)
# ==============================================================================
class VertexDenseEmbedder(Embeddings):
    """
    Production-grade LangChain Embeddings adapter wrapping Google GenAI SDK (Vertex AI).
    Implements batched embedding with retries and exponential backoff.
    """
    def __init__(self, client: genai.Client, model_name: str = "text-embedding-004", batch_size: int = 50):
        self.client = client
        self.model_name = model_name
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
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
            time.sleep(0.05)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        retries = 3
        while retries > 0:
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=[text]
                )
                return response.embeddings[0].values
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise e
                time.sleep(1.0)
        return []


# ==============================================================================
# 3. Dual-Retrieval Setup & Reciprocal Rank Fusion (RRF)
# ==============================================================================
class HybridRetrieverSystem:
    """
    Manages the dual-retrieval pipeline and Reciprocal Rank Fusion.
    
    Architecture:
    -------------
    [Input Query]
         │
         ├───> [Stream A: Dense Retriever (Chroma k=10)] ───> Top 10 Dense Chunks
         │                                                            │
         └───> [Stream B: Sparse Retriever (BM25 k=10)]  ───> Top 10 Sparse Chunks
                                                                      │
                                          ┌───────────────────────────┘
                                          ▼
                         [EnsembleRetriever (RRF, weights=[0.5, 0.5])]
                                          │
                                          ▼
                            [Top-5 Fused Chunks to LLM]
    """
    def __init__(
        self,
        config: HybridRAGConfig,
        embedder: VertexDenseEmbedder,
        chunks: List[Document],
        reindex: bool = False
    ):
        self.config = config
        self.embedder = embedder
        self.chunks = chunks
        self.reindex = reindex

        # Initialize Stream A & Stream B
        self._init_dense_retriever()
        self._init_sparse_retriever()
        self._init_ensemble_retriever()

    def _init_dense_retriever(self) -> None:
        """
        ========================================================================
        STREAM A: DENSE RETRIEVER
        Dense semantic retrieval using Chroma vector store and text-embedding-004.
        Retrieves top k=10 documents based on cosine vector similarity.
        ========================================================================
        """
        logger.info(f"Initializing Dense Vector Store (ChromaDB at {self.config.chroma_db_dir})...")
        
        # Check if collection already exists
        import chromadb
        client = chromadb.PersistentClient(path=self.config.chroma_db_dir)
        existing_collections = [c.name for c in client.list_collections()]

        if self.reindex and self.config.collection_name in existing_collections:
            logger.info(f"Deleting existing Chroma collection: {self.config.collection_name}")
            client.delete_collection(self.config.collection_name)
            existing_collections.remove(self.config.collection_name)

        if self.config.collection_name not in existing_collections or client.get_collection(self.config.collection_name).count() == 0:
            logger.info(f"Creating new Chroma collection '{self.config.collection_name}' with {len(self.chunks)} chunks...")
            # Create Chroma vector store in batches to monitor progress
            self.chroma_store = Chroma(
                collection_name=self.config.collection_name,
                embedding_function=self.embedder,
                persist_directory=self.config.chroma_db_dir,
                collection_metadata={"hnsw:space": "cosine"}
            )
            # Add documents in batches
            batch_size = 50
            with tqdm(total=len(self.chunks), desc="Indexing Chroma (Dense Stream)", unit="doc") as pbar:
                for i in range(0, len(self.chunks), batch_size):
                    batch_docs = self.chunks[i:i + batch_size]
                    self.chroma_store.add_documents(batch_docs)
                    pbar.update(len(batch_docs))
            logger.info(f"Dense vector indexing complete. Stored {self.chroma_store._collection.count()} chunks.")
        else:
            logger.info(f"Reusing existing Chroma collection '{self.config.collection_name}' ({client.get_collection(self.config.collection_name).count()} chunks).")
            self.chroma_store = Chroma(
                collection_name=self.config.collection_name,
                embedding_function=self.embedder,
                persist_directory=self.config.chroma_db_dir,
                collection_metadata={"hnsw:space": "cosine"}
            )

        # Dense stream retrieves top k=10
        self.dense_retriever = self.chroma_store.as_retriever(
            search_kwargs={"k": self.config.dense_top_k}
        )
        logger.info(f"Stream A (Dense Retriever) configured with k={self.config.dense_top_k}.")

    def _init_sparse_retriever(self) -> None:
        """
        ========================================================================
        STREAM B: SPARSE RETRIEVER
        Exact lexical keyword retrieval using BM25Okapi over the exact same
        pool of document chunks. Retrieves top k=10 documents based on TF-IDF.
        ========================================================================
        """
        logger.info("Initializing Sparse Retriever (BM25Okapi over identical document chunks)...")
        # Initialize BM25 from the exact same list of Document objects
        self.sparse_retriever = BM25Retriever.from_documents(self.chunks)
        self.sparse_retriever.k = self.config.sparse_top_k
        logger.info(f"Stream B (Sparse Retriever) configured with k={self.config.sparse_top_k}.")

    def _init_ensemble_retriever(self) -> None:
        """
        ========================================================================
        FUSION POINT: RECIPROCAL RANK FUSION (RRF)
        LangChain EnsembleRetriever merges candidates from Stream A and Stream B.
        Weights: [0.5, 0.5] (equal weighting between dense and sparse).
        c constant: 60 (standard RRF smoothing parameter).
        
        Formula:
          RRF_Score(d) = 0.5 / (60 + rank_dense(d)) + 0.5 / (60 + rank_sparse(d))
        
        Documents appearing in both streams receive a massive score boost,
        ensuring high Context Precision and eliminating "Lost in the Middle".
        ========================================================================
        """
        logger.info(
            f"Configuring EnsembleRetriever with weights={self.config.ensemble_weights}, "
            f"c={self.config.rrf_c_constant}, id_key='chunk_id'..."
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=self.config.ensemble_weights,
            c=self.config.rrf_c_constant,
            id_key="chunk_id"
        )

    def retrieve(self, query: str) -> List[Document]:
        """
        Executes dual retrieval + RRF fusion and slices the top-5 documents
        to feed to the generator LLM.
        """
        fused_docs = self.ensemble_retriever.invoke(query)
        # Slices exactly the top final_top_k (top-5) documents
        return fused_docs[:self.config.final_top_k]


# ==============================================================================
# 4. Generation Chain (Strict Zero-Shot Prompt Stuffing Container)
# ==============================================================================
class HybridRAGChain:
    """
    Prompt stuffing container strictly adhering to Phase 1 constraints:
    - Exactly top-5 fused chunks from EnsembleRetriever.
    - Chunks concatenated separated by '---'.
    - Temperature=0.0 zero-shot generation with strict negative fallback constraint.
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

    def __init__(self, client: genai.Client, retriever_system: HybridRetrieverSystem, config: HybridRAGConfig):
        self.client = client
        self.retriever_system = retriever_system
        self.config = config

    def retrieve(self, question: str) -> List[Document]:
        return self.retriever_system.retrieve(question)

    def generate(self, question: str, retrieved_docs: List[Document]) -> str:
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]
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

    def run(self, question: str) -> Tuple[str, List[Document]]:
        docs = self.retrieve(question)
        answer = self.generate(question, docs)
        return answer, docs


# ==============================================================================
# 5. Evaluation Harness: Ragas & Production LLM-as-a-Judge
# ==============================================================================
class LLMJudgeEvaluator:
    """
    Self-contained, production-grade LLM-as-a-judge evaluation harness.
    Implements the mathematical rubrics of Ragas metrics:
    1. Context Recall: Fraction of ground-truth factual statements attributed to context.
    2. Context Precision: Mean Average Precision (MAP) of relevant chunks in retrieved top-K.
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
# 6. Comparative Analytics: Phase 1 Baseline vs Phase 2 Hybrid
# ==============================================================================
def generate_comparative_report(
    df_hybrid: pd.DataFrame,
    phase1_csv_path: str,
    output_comparison_csv: str
) -> pd.DataFrame:
    """
    Computes side-by-side metric comparison between Phase 1 Naive Baseline
    and Phase 2 Hybrid Architecture across question categories.
    """
    metrics = ["context_recall", "context_precision", "faithfulness"]
    
    # Check if Phase 1 baseline exists
    if not os.path.exists(phase1_csv_path):
        logger.warning(f"Phase 1 results file not found at {phase1_csv_path}. Displaying Phase 2 summary only.")
        summary_v2 = df_hybrid.groupby("category")[metrics].mean().round(4)
        print("\n" + "=" * 80)
        print("PHASE 2 HYBRID SEARCH EVALUATION REPORT")
        print("=" * 80)
        print(summary_v2.to_string())
        return summary_v2

    df_p1 = pd.read_csv(phase1_csv_path)

    # Compute category-level means
    p1_cat = df_p1.groupby("category")[metrics].mean()
    p2_cat = df_hybrid.groupby("category")[metrics].mean()

    # Compute overall means
    p1_overall = df_p1[metrics].mean()
    p2_overall = df_hybrid[metrics].mean()

    p1_cat.loc["OVERALL"] = p1_overall
    p2_cat.loc["OVERALL"] = p2_overall

    # Build comparison rows for categories present in Phase 2 + OVERALL
    categories_to_compare = [c for c in p1_cat.index if c in p2_cat.index]
    if "OVERALL" in p2_cat.index and "OVERALL" not in categories_to_compare:
        categories_to_compare.append("OVERALL")

    comparison_rows = []
    for cat in categories_to_compare:
        for m in metrics:
            val_p1 = p1_cat.loc[cat, m]
            val_p2 = p2_cat.loc[cat, m]
            delta = val_p2 - val_p1
            rel_change = (delta / val_p1 * 100) if val_p1 > 0 else 0.0
            comparison_rows.append({
                "Category": cat,
                "Metric": m,
                "Phase 1 Naive": round(val_p1, 4),
                "Phase 2 Hybrid": round(val_p2, 4),
                "Delta (Abs)": round(delta, 4),
                "Improvement (%)": f"{rel_change:+.2f}%"
            })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_comparison_csv, index=False)
    logger.info(f"Comparison report saved to: {output_comparison_csv}")

    # Display formatted tables
    print("\n" + "=" * 90)
    print("      RAG ARCHITECTURE UPGRADE EVALUATION REPORT: PHASE 1 vs. PHASE 2")
    print("=" * 90)
    print(comparison_df.to_string(index=False))
    print("=" * 90 + "\n")

    return comparison_df


# ==============================================================================
# 7. Main Execution Harness
# ==============================================================================
def run_evaluation(config: HybridRAGConfig, reindex: bool = False, max_samples: Optional[int] = None):
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

    # 1. Advanced Ingestion & Chunking
    ingestion = AdvancedDocumentIngestionPipeline(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    chunks = ingestion.process_pdfs(config.pdf_paths)
    if not chunks:
        logger.error("No chunks extracted from PDF files! Exiting.")
        sys.exit(1)

    # 2. Hybrid Dual-Retrieval System (Chroma + BM25 + EnsembleRetriever)
    retriever_system = HybridRetrieverSystem(
        config=config,
        embedder=embedder,
        chunks=chunks,
        reindex=reindex
    )

    # 3. Load golden test set
    if not os.path.exists(config.testset_path):
        logger.error(f"Evaluation dataset not found at {config.testset_path}!")
        sys.exit(1)

    with open(config.testset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if max_samples:
        test_data = test_data[:max_samples]

    logger.info(f"Loaded {len(test_data)} evaluation test cases from {config.testset_path}.")

    rag_chain = HybridRAGChain(client=client, retriever_system=retriever_system, config=config)
    evaluator = LLMJudgeEvaluator(client=client, judge_model=config.llm_model)

    records: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("STARTING PHASE 2 HYBRID SEARCH EVALUATION HARNESS")
    print("=" * 80 + "\n")

    eval_pbar = tqdm(enumerate(test_data, start=1), total=len(test_data), desc="Evaluating Hybrid RAG", unit="sample")
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

        # 1. Retrieve top-5 fused documents via EnsembleRetriever
        retrieved_docs = rag_chain.retrieve(question)
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]
        retrieved_chunk_ids = [doc.metadata.get("chunk_id", "unknown") for doc in retrieved_docs]

        # 2. Generate answer with strict negative constraint container
        generated_answer = rag_chain.generate(question, retrieved_docs)

        # 3. Evaluate Ragas Metrics
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
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_contexts": json.dumps(retrieved_contexts),
            "num_chunks_retrieved": len(retrieved_contexts)
        }
        records.append(record)

        eval_pbar.set_postfix({
            "Cat": category[:10],
            "Recall": f"{context_recall:.2f}",
            "Prec": f"{context_precision:.2f}",
            "Faith": f"{faithfulness:.2f}"
        })

        time.sleep(config.rate_limit_delay_seconds)

    # Convert to DataFrame
    df_hybrid = pd.DataFrame(records)

    # Save granular traces
    df_hybrid.to_csv(config.output_csv_path, index=False)
    logger.info(f"Saved granular row-level traces to: {config.output_csv_path}")

    # Generate comparative analysis against Phase 1
    generate_comparative_report(
        df_hybrid=df_hybrid,
        phase1_csv_path=config.phase1_results_path,
        output_comparison_csv=config.comparison_csv_path
    )

    return df_hybrid


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Hybrid Search (Dense + Sparse RRF) RAG & Evaluation Harness for SEC 10-K Filings."
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
        "--phase1-results",
        default="phase1_baseline_results.csv",
        help="Path to Phase 1 baseline results CSV for comparative analysis."
    )
    parser.add_argument(
        "--output-csv",
        default="phase2_hybrid_results.csv",
        help="Output CSV path for granular evaluation traces."
    )
    parser.add_argument(
        "--comparison-csv",
        default="phase2_vs_phase1_comparison.csv",
        help="Output CSV path for Phase 1 vs Phase 2 comparative analysis."
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
        help="Optional limit on number of test set questions to evaluate (for dry-run/testing)."
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

    config = HybridRAGConfig(
        project_id=args.project_id,
        location=args.location,
        pdf_paths=args.pdf_paths,
        testset_path=args.testset,
        phase1_results_path=args.phase1_results,
        output_csv_path=args.output_csv,
        comparison_csv_path=args.comparison_csv
    )

    run_evaluation(config, reindex=args.reindex, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
