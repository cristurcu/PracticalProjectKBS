"""
Corpus downloader for MedRAG replication.
Downloads PubMed and StatPearls corpora from HuggingFace.
"""
import json
import os
import subprocess
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from .config import CORPUS_DIR


def check_git_lfs() -> bool:
    """Check if git-lfs is installed."""
    try:
        result = subprocess.run(
            ["git", "lfs", "version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_corpus_hf(corpus_name: str, max_docs: Optional[int] = None) -> Path:
    """
    Download corpus from HuggingFace using the datasets library.

    Args:
        corpus_name: Name of the corpus (pubmed, statpearls)
        max_docs: Optional limit on number of documents (for testing)

    Returns:
        Path to the downloaded corpus
    """
    from datasets import load_dataset

    corpus_path = CORPUS_DIR / corpus_name
    corpus_file = corpus_path / "corpus.jsonl"

    if corpus_file.exists():
        print(f"Corpus already exists at {corpus_file}")
        return corpus_path

    corpus_path.mkdir(parents=True, exist_ok=True)

    # Map corpus names to HuggingFace dataset paths
    hf_paths = {
        "pubmed": "MedRAG/pubmed",
        "statpearls": "MedRAG/statpearls",
    }

    if corpus_name not in hf_paths:
        raise ValueError(f"Unknown corpus: {corpus_name}. Available: {list(hf_paths.keys())}")

    print(f"Downloading {corpus_name} corpus from HuggingFace...")
    print("This may take a while depending on corpus size...")

    try:
        # Load dataset from HuggingFace
        dataset = load_dataset(hf_paths[corpus_name], split="train")

        # Apply limit if specified
        if max_docs:
            dataset = dataset.select(range(min(max_docs, len(dataset))))

        print(f"Writing {len(dataset)} documents to {corpus_file}")

        with open(corpus_file, 'w') as f:
            for item in tqdm(dataset, desc="Writing corpus"):
                doc = {
                    "id": item.get("id", item.get("_id", "")),
                    "title": item.get("title", ""),
                    "content": item.get("content", item.get("text", "")),
                }
                f.write(json.dumps(doc) + "\n")

        print(f"Corpus saved to {corpus_file}")
        return corpus_path

    except Exception as e:
        print(f"Error downloading corpus: {e}")
        raise


def download_corpus_git(corpus_name: str) -> Path:
    """
    Download corpus using git clone from HuggingFace.

    Args:
        corpus_name: Name of the corpus (pubmed, statpearls)

    Returns:
        Path to the downloaded corpus
    """
    corpus_path = CORPUS_DIR / corpus_name

    if corpus_path.exists() and (corpus_path / "corpus.jsonl").exists():
        print(f"Corpus already exists at {corpus_path}")
        return corpus_path

    if not check_git_lfs():
        print("Warning: git-lfs not installed. Some files may not download correctly.")
        print("Install with: brew install git-lfs (macOS) or apt install git-lfs (Linux)")

    hf_urls = {
        "pubmed": "https://huggingface.co/datasets/MedRAG/pubmed",
        "statpearls": "https://huggingface.co/datasets/MedRAG/statpearls",
    }

    if corpus_name not in hf_urls:
        raise ValueError(f"Unknown corpus: {corpus_name}")

    print(f"Cloning {corpus_name} corpus from HuggingFace...")

    try:
        subprocess.run(
            ["git", "clone", hf_urls[corpus_name], str(corpus_path)],
            check=True,
        )
        print(f"Corpus downloaded to {corpus_path}")
        return corpus_path

    except subprocess.CalledProcessError as e:
        print(f"Error cloning corpus: {e}")
        raise


def create_sample_corpus(corpus_name: str, num_docs: int = 1000) -> Path:
    """
    Create a sample corpus for testing without downloading full datasets.

    Args:
        corpus_name: Name of the corpus
        num_docs: Number of sample documents to create

    Returns:
        Path to the sample corpus
    """
    corpus_path = CORPUS_DIR / corpus_name
    corpus_file = corpus_path / "corpus.jsonl"

    corpus_path.mkdir(parents=True, exist_ok=True)

    if corpus_file.exists():
        print(f"Corpus already exists at {corpus_file}")
        return corpus_path

    print(f"Creating sample corpus with {num_docs} documents...")

    sample_docs = []

    if corpus_name == "pubmed":
        # Sample PubMed-style abstracts
        topics = [
            "diabetes mellitus", "hypertension", "cancer therapy",
            "cardiovascular disease", "infectious disease", "immunology",
            "neurology", "pharmacology", "surgery", "pediatrics",
        ]
        for i in range(num_docs):
            topic = topics[i % len(topics)]
            sample_docs.append({
                "id": f"pubmed_{i}",
                "title": f"Study on {topic} treatment approaches #{i}",
                "content": f"This study investigates the effects of various treatments "
                          f"for {topic}. We analyzed data from clinical trials and found "
                          f"significant improvements in patient outcomes. The methodology "
                          f"included randomized controlled trials with proper controls.",
            })

    elif corpus_name == "statpearls":
        # Sample StatPearls-style clinical articles
        conditions = [
            "Myocardial Infarction", "Pneumonia", "Appendicitis",
            "Stroke", "Sepsis", "Diabetes Complications",
            "Kidney Disease", "Liver Cirrhosis", "Anemia", "Arthritis",
        ]
        for i in range(num_docs):
            condition = conditions[i % len(conditions)]
            sample_docs.append({
                "id": f"statpearls_{i}",
                "title": f"{condition} - Clinical Overview",
                "content": f"{condition} is a common clinical condition that requires "
                          f"prompt diagnosis and treatment. Key symptoms include specific "
                          f"clinical manifestations. Diagnosis involves laboratory tests "
                          f"and imaging. Treatment options include pharmacological and "
                          f"non-pharmacological approaches.",
            })

    with open(corpus_file, 'w') as f:
        for doc in sample_docs:
            f.write(json.dumps(doc) + "\n")

    print(f"Sample corpus saved to {corpus_file}")
    return corpus_path


def download_all_corpora(use_samples: bool = False, sample_size: int = 1000) -> None:
    """
    Download all required corpora.

    Args:
        use_samples: If True, create sample corpora instead of downloading
        sample_size: Number of documents for sample corpora
    """
    corpora = ["pubmed", "statpearls"]

    for corpus_name in corpora:
        print(f"\n{'='*50}")
        print(f"Processing corpus: {corpus_name}")
        print(f"{'='*50}")

        try:
            if use_samples:
                create_sample_corpus(corpus_name, sample_size)
            else:
                download_corpus_hf(corpus_name)
        except Exception as e:
            print(f"Failed to download {corpus_name}: {e}")
            print("Creating sample corpus as fallback...")
            create_sample_corpus(corpus_name, sample_size)
