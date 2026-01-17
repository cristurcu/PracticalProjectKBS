"""
Configuration settings for MedRAG replication study.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CORPUS_DIR = DATA_DIR / "corpus"
BENCHMARK_DIR = DATA_DIR / "benchmark"
INDEX_DIR = DATA_DIR / "indices"
SNIPPETS_DIR = DATA_DIR / "snippets"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, CORPUS_DIR, BENCHMARK_DIR, INDEX_DIR, SNIPPETS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model settings
LLM_CONFIG = {
    "model": "gpt-3.5-turbo",
    "temperature": 0,
    "max_tokens": 500,
}

# Retrieval settings
RETRIEVAL_CONFIG = {
    "k_values": [4, 8, 16, 32],
    "rrf_k": 60,  # RRF smoothing parameter
    "context_length": 3500,  # Max tokens for context
}

# Corpus configuration
CORPUS_CONFIG = {
    "pubmed": {
        "name": "PubMed",
        "hf_path": "MedRAG/pubmed",
        "description": "PubMed abstracts",
    },
    "statpearls": {
        "name": "StatPearls",
        "hf_path": "MedRAG/statpearls",
        "description": "StatPearls clinical articles",
    },
}

# Dataset configuration
DATASET_CONFIG = {
    "medqa": {
        "name": "MedQA-US",
        "subset_size": 300,  # Reduced for replication
        "answer_type": "mcq",  # A, B, C, D
    },
    "pubmedqa": {
        "name": "PubMedQA",
        "subset_size": 500,
        "answer_type": "binary",  # yes, no, maybe
    },
}

# Experiment settings
EXPERIMENTS = {
    "baseline": {
        "name": "Baseline (CoT, No RAG)",
        "rag": False,
        "retriever": None,
        "corpus": None,
    },
    "bm25_pubmed": {
        "name": "BM25 + PubMed",
        "rag": True,
        "retriever": "bm25",
        "corpus": "pubmed",
    },
    "bm25_statpearls": {
        "name": "BM25 + StatPearls",
        "rag": True,
        "retriever": "bm25",
        "corpus": "statpearls",
    },
    "medcpt_pubmed": {
        "name": "MedCPT + PubMed",
        "rag": True,
        "retriever": "medcpt",
        "corpus": "pubmed",
    },
    "medcpt_statpearls": {
        "name": "MedCPT + StatPearls",
        "rag": True,
        "retriever": "medcpt",
        "corpus": "statpearls",
    },
    "hybrid": {
        "name": "Hybrid (RRF: BM25 + MedCPT)",
        "rag": True,
        "retriever": "hybrid",
        "corpus": "pubmed",
    },
}
