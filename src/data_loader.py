"""
Data loader for benchmark datasets.
Handles loading and preprocessing of MedQA and PubMedQA datasets.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests
from tqdm import tqdm

from .config import BENCHMARK_DIR, DATASET_CONFIG


@dataclass
class Question:
    """Represents a single question from the benchmark."""
    id: str
    question: str
    options: Dict[str, str]  # For MedQA: {"A": "...", "B": "...", ...}
    answer: str  # Correct answer key
    dataset: str
    context: Optional[str] = None  # For PubMedQA


class BenchmarkDataset:
    """Loads and manages benchmark datasets."""

    BENCHMARK_URL = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"

    def __init__(self):
        self.benchmark_path = BENCHMARK_DIR / "benchmark.json"
        self.data = None

    def download_benchmark(self) -> None:
        """Download the benchmark file if not present."""
        if self.benchmark_path.exists():
            print(f"Benchmark file already exists at {self.benchmark_path}")
            return

        print("Downloading benchmark data...")
        response = requests.get(self.BENCHMARK_URL, timeout=60)
        response.raise_for_status()

        with open(self.benchmark_path, 'w') as f:
            json.dump(response.json(), f, indent=2)
        print(f"Benchmark saved to {self.benchmark_path}")

    def load(self) -> None:
        """Load the benchmark data."""
        if not self.benchmark_path.exists():
            self.download_benchmark()

        with open(self.benchmark_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded benchmark with datasets: {list(self.data.keys())}")

    def get_dataset(self, dataset_name: str, subset_size: Optional[int] = None,
                    seed: int = 42) -> List[Question]:
        """
        Get questions from a specific dataset.

        Args:
            dataset_name: Name of the dataset (medqa, pubmedqa, etc.)
            subset_size: Optional number of questions to sample
            seed: Random seed for reproducibility

        Returns:
            List of Question objects
        """
        if self.data is None:
            self.load()

        # Map our config names to benchmark keys
        dataset_key_map = {
            "medqa": "medqa",
            "pubmedqa": "pubmedqa",
        }

        key = dataset_key_map.get(dataset_name.lower(), dataset_name.lower())

        if key not in self.data:
            raise ValueError(f"Dataset '{key}' not found. Available: {list(self.data.keys())}")

        raw_data = self.data[key]
        questions = []

        for q_id, item in raw_data.items():
            # Handle different dataset formats
            if dataset_name.lower() == "pubmedqa":
                question = self._parse_pubmedqa(q_id, item)
            else:
                question = self._parse_medqa(q_id, item, dataset_name)

            if question:
                questions.append(question)

        # Sample if subset_size is specified
        if subset_size and len(questions) > subset_size:
            random.seed(seed)
            questions = random.sample(questions, subset_size)

        print(f"Loaded {len(questions)} questions from {dataset_name}")
        return questions

    def _parse_medqa(self, q_id: str, item: Dict, dataset_name: str) -> Optional[Question]:
        """Parse a MedQA format question."""
        try:
            return Question(
                id=q_id,
                question=item.get("question", ""),
                options=item.get("options", {}),
                answer=item.get("answer", ""),
                dataset=dataset_name,
            )
        except Exception as e:
            print(f"Error parsing question {q_id}: {e}")
            return None

    def _parse_pubmedqa(self, q_id: str, item: Dict) -> Optional[Question]:
        """Parse a PubMedQA format question."""
        try:
            # PubMedQA has yes/no/maybe answers
            answer_map = {"yes": "A", "no": "B", "maybe": "C"}
            raw_answer = item.get("answer", "").lower()

            return Question(
                id=q_id,
                question=item.get("question", ""),
                options={"A": "yes", "B": "no", "C": "maybe"},
                answer=answer_map.get(raw_answer, raw_answer),
                dataset="pubmedqa",
                context=item.get("context", None),  # PubMedQA includes context
            )
        except Exception as e:
            print(f"Error parsing PubMedQA question {q_id}: {e}")
            return None


def load_precomputed_snippets(dataset: str, retriever: str, corpus: str,
                               k: int = 32) -> Dict[str, List[Dict]]:
    """
    Load pre-computed retrieved snippets.

    The original MedRAG provides pre-computed top-k snippets for each question.
    This function loads them from the snippets directory.

    Args:
        dataset: Dataset name (medqa, pubmedqa)
        retriever: Retriever name (bm25, medcpt)
        corpus: Corpus name (pubmed, statpearls)
        k: Number of snippets

    Returns:
        Dict mapping question_id to list of snippets
    """
    from .config import SNIPPETS_DIR

    filename = f"{dataset}_{retriever}_{corpus}_top{k}.json"
    filepath = SNIPPETS_DIR / filename

    if not filepath.exists():
        print(f"Pre-computed snippets not found: {filepath}")
        return {}

    with open(filepath, 'r') as f:
        return json.load(f)
