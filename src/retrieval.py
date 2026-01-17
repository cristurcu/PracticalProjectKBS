"""
Retrieval system for MedRAG replication.
Implements BM25, dense retrieval (MedCPT), and hybrid (RRF) methods.
"""
import json
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

# IMPORTANT: Import sentence_transformers BEFORE faiss to prevent segfaults on macOS
# See: https://github.com/huggingface/sentence-transformers/issues/2291
from sentence_transformers import SentenceTransformer
import faiss

from .config import CORPUS_DIR, INDEX_DIR, RETRIEVAL_CONFIG


@dataclass
class RetrievedDoc:
    """Represents a retrieved document/snippet."""
    doc_id: str
    title: str
    content: str
    score: float
    rank: int


class BaseRetriever(ABC):
    """Base class for retrievers."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 32) -> List[RetrievedDoc]:
        """Retrieve top-k documents for a query."""
        pass


class BM25Retriever(BaseRetriever):
    """BM25 lexical retriever using rank_bm25."""

    def __init__(self, corpus_name: str):
        self.corpus_name = corpus_name
        self.corpus_path = CORPUS_DIR / corpus_name
        self.index = None
        self.documents = []
        self.doc_ids = []

    def load_corpus(self) -> None:
        """Load corpus documents for BM25 indexing."""
        from rank_bm25 import BM25Okapi
        import re

        print(f"Loading corpus: {self.corpus_name}")

        corpus_file = self.corpus_path / "corpus.jsonl"
        if not corpus_file.exists():
            raise FileNotFoundError(
                f"Corpus not found at {corpus_file}. "
                f"Please download it first using download_corpus()."
            )

        self.documents = []
        self.doc_ids = []

        with open(corpus_file, 'r') as f:
            for line in tqdm(f, desc="Loading documents"):
                doc = json.loads(line)
                self.doc_ids.append(doc.get("id", doc.get("_id", "")))
                # Combine title and content
                text = f"{doc.get('title', '')} {doc.get('content', doc.get('text', ''))}"
                self.documents.append(doc)

        # Tokenize documents for BM25
        print("Building BM25 index...")
        tokenized_docs = [self._tokenize(
            f"{d.get('title', '')} {d.get('content', d.get('text', ''))}"
        ) for d in tqdm(self.documents, desc="Tokenizing")]

        self.index = BM25Okapi(tokenized_docs)
        print(f"BM25 index built with {len(self.documents)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        return re.findall(r'\w+', text.lower())

    def retrieve(self, query: str, k: int = 32) -> List[RetrievedDoc]:
        """Retrieve top-k documents using BM25."""
        if self.index is None:
            self.load_corpus()

        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_indices):
            doc = self.documents[idx]
            results.append(RetrievedDoc(
                doc_id=self.doc_ids[idx],
                title=doc.get("title", ""),
                content=doc.get("content", doc.get("text", "")),
                score=float(scores[idx]),
                rank=rank + 1,
            ))

        return results


class DenseRetriever(BaseRetriever):
    """Dense retriever using sentence transformers (e.g., MedCPT)."""

    def __init__(self, corpus_name: str, model_name: str = "ncbi/MedCPT-Query-Encoder"):
        self.corpus_name = corpus_name
        self.model_name = model_name
        self.corpus_path = CORPUS_DIR / corpus_name
        self.index = None
        self.documents = []
        self.doc_ids = []
        self.model = None

    def load_model(self) -> None:
        """Load the embedding model."""
        print(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

    def load_corpus_and_index(self) -> None:
        """Load corpus and build/load FAISS index."""
        
        if self.model is None:
            self.load_model()

        print(f"Loading corpus: {self.corpus_name}")

        corpus_file = self.corpus_path / "corpus.jsonl"
        index_file = INDEX_DIR / f"{self.corpus_name}_medcpt.index"
        ids_file = INDEX_DIR / f"{self.corpus_name}_medcpt_ids.json"

        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus not found at {corpus_file}")

        # Load documents
        self.documents = []
        self.doc_ids = []

        with open(corpus_file, 'r') as f:
            for line in f:
                doc = json.loads(line)
                self.doc_ids.append(doc.get("id", doc.get("_id", "")))
                self.documents.append(doc)

        # Check for pre-built index
        if index_file.exists() and ids_file.exists():
            print("Loading pre-built FAISS index...")
            self.index = faiss.read_index(str(index_file))
            with open(ids_file, 'r') as f:
                self.doc_ids = json.load(f)
            print(f"Loaded index with {self.index.ntotal} vectors")
        else:
            print("Building FAISS index (this may take a while)...")
            self._build_index(index_file, ids_file)

    def _build_index(self, index_file: Path, ids_file: Path) -> None:
        """Build FAISS index from corpus."""
        
        # Encode documents in batches
        batch_size = 128
        all_embeddings = []

        texts = [
            f"{d.get('title', '')} {d.get('content', d.get('text', ''))}"
            for d in self.documents
        ]

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding documents"):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.append(embeddings)

        all_embeddings = np.vstack(all_embeddings).astype('float32')

        # Build FAISS index
        dimension = all_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        faiss.normalize_L2(all_embeddings)  # Normalize for cosine similarity
        self.index.add(all_embeddings)

        # Save index
        faiss.write_index(self.index, str(index_file))
        with open(ids_file, 'w') as f:
            json.dump(self.doc_ids, f)

        print(f"Index saved to {index_file}")

    def retrieve(self, query: str, k: int = 32) -> List[RetrievedDoc]:
        """Retrieve top-k documents using dense retrieval."""
        
        if self.index is None:
            self.load_corpus_and_index()

        # Encode query
        query_embedding = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append(RetrievedDoc(
                    doc_id=self.doc_ids[idx],
                    title=doc.get("title", ""),
                    content=doc.get("content", doc.get("text", "")),
                    score=float(score),
                    rank=rank + 1,
                ))

        return results


class HybridRetriever(BaseRetriever):
    """Hybrid retriever using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, corpus_name: str, rrf_k: int = 60):
        self.corpus_name = corpus_name
        self.rrf_k = rrf_k
        self.bm25_retriever = BM25Retriever(corpus_name)
        self.dense_retriever = DenseRetriever(corpus_name)

    def retrieve(self, query: str, k: int = 32) -> List[RetrievedDoc]:
        """Retrieve using RRF fusion of BM25 and dense retrieval."""
        # Get results from both retrievers (get more than k to have good fusion)
        bm25_results = self.bm25_retriever.retrieve(query, k=k * 2)
        dense_results = self.dense_retriever.retrieve(query, k=k * 2)

        # Apply RRF
        rrf_scores = {}
        doc_info = {}

        # Process BM25 results
        for doc in bm25_results:
            rrf_score = 1.0 / (self.rrf_k + doc.rank)
            rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0) + rrf_score
            doc_info[doc.doc_id] = doc

        # Process dense results
        for doc in dense_results:
            rrf_score = 1.0 / (self.rrf_k + doc.rank)
            rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0) + rrf_score
            if doc.doc_id not in doc_info:
                doc_info[doc.doc_id] = doc

        # Sort by RRF score and get top-k
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            original_doc = doc_info[doc_id]
            results.append(RetrievedDoc(
                doc_id=doc_id,
                title=original_doc.title,
                content=original_doc.content,
                score=score,
                rank=rank + 1,
            ))

        return results


class PrecomputedRetriever(BaseRetriever):
    """Uses pre-computed snippets instead of live retrieval."""

    def __init__(self, snippets: Dict[str, List[Dict]]):
        """
        Args:
            snippets: Dict mapping question_id to list of snippet dicts
        """
        self.snippets = snippets

    def retrieve(self, query: str, k: int = 32,
                 question_id: Optional[str] = None) -> List[RetrievedDoc]:
        """Get pre-computed snippets for a question."""
        if question_id is None or question_id not in self.snippets:
            return []

        snippets = self.snippets[question_id][:k]

        return [
            RetrievedDoc(
                doc_id=s.get("id", str(i)),
                title=s.get("title", ""),
                content=s.get("content", s.get("text", "")),
                score=s.get("score", 1.0 - i * 0.01),
                rank=i + 1,
            )
            for i, s in enumerate(snippets)
        ]


def get_retriever(retriever_name: str, corpus_name: str,
                  precomputed_snippets: Optional[Dict] = None) -> BaseRetriever:
    """Factory function to get the appropriate retriever."""
    if precomputed_snippets:
        return PrecomputedRetriever(precomputed_snippets)

    retriever_name = retriever_name.lower()

    if retriever_name == "bm25":
        return BM25Retriever(corpus_name)
    elif retriever_name in ["medcpt", "dense"]:
        return DenseRetriever(corpus_name)
    elif retriever_name in ["hybrid", "rrf"]:
        return HybridRetriever(corpus_name)
    else:
        raise ValueError(f"Unknown retriever: {retriever_name}")
