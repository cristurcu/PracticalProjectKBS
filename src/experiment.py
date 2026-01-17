"""
Experiment runner for MedRAG replication study.
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

from .config import RESULTS_DIR, EXPERIMENTS, RETRIEVAL_CONFIG, DATASET_CONFIG
from .data_loader import BenchmarkDataset, Question
from .retrieval import get_retriever, BaseRetriever, RetrievedDoc
from .llm import LLMInterface


@dataclass
class ExperimentResult:
    """Results from a single question."""
    question_id: str
    question: str
    correct_answer: str
    predicted_answer: str
    is_correct: bool
    raw_response: str
    metadata: Dict[str, Any]


@dataclass
class ExperimentSummary:
    """Summary of an experiment run."""
    experiment_name: str
    dataset: str
    retriever: Optional[str]
    corpus: Optional[str]
    k: Optional[int]
    total_questions: int
    correct: int
    accuracy: float
    timestamp: str
    duration_seconds: float


class ExperimentRunner:
    """Runs experiments for MedRAG replication."""

    def __init__(self, api_key: str = None):
        self.llm = LLMInterface(api_key=api_key)
        self.benchmark = BenchmarkDataset()

    def run_experiment(
        self,
        experiment_name: str,
        dataset_name: str,
        retriever: Optional[BaseRetriever] = None,
        k: int = None,
        subset_size: Optional[int] = None,
        save_results: bool = True,
    ) -> ExperimentSummary:
        """
        Run a single experiment configuration.

        Args:
            experiment_name: Name of the experiment
            dataset_name: Dataset to use (medqa, pubmedqa)
            retriever: Optional retriever for RAG experiments
            k: Number of snippets to retrieve
            subset_size: Number of questions to use
            save_results: Whether to save detailed results

        Returns:
            ExperimentSummary with accuracy and metadata
        """
        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment_name}")
        print(f"Dataset: {dataset_name}, k={k}")
        print(f"{'='*60}\n")

        start_time = time.time()

        # Load dataset
        if subset_size is None:
            subset_size = DATASET_CONFIG.get(dataset_name, {}).get("subset_size")

        questions = self.benchmark.get_dataset(dataset_name, subset_size=subset_size)

        # Determine dataset type
        dataset_type = "pubmedqa" if dataset_name.lower() == "pubmedqa" else "mcq"

        results = []
        correct_count = 0

        for question in tqdm(questions, desc=f"Processing {dataset_name}"):
            # Retrieve documents if using RAG
            documents = None
            if retriever is not None and k is not None:
                try:
                    documents = retriever.retrieve(
                        question.question,
                        k=k,
                        question_id=question.id if hasattr(retriever, 'snippets') else None
                    )
                except TypeError:
                    # Fallback for retrievers that don't support question_id
                    documents = retriever.retrieve(question.question, k=k)

            # Get answer from LLM
            predicted, raw_response, metadata = self.llm.answer_question(
                question=question.question,
                options=question.options,
                documents=documents,
                dataset_type=dataset_type,
                context=question.context,
            )

            # Check correctness
            # Handle different answer formats
            correct_answer = question.answer
            if dataset_type == "pubmedqa":
                # PubMedQA: correct_answer is "A"/"B"/"C", predicted is "yes"/"no"/"maybe"
                # Map correct_answer back to yes/no/maybe for comparison
                answer_map = {"A": "yes", "B": "no", "C": "maybe"}
                correct_text = answer_map.get(correct_answer.upper(), correct_answer.lower())
                is_correct = predicted.lower() == correct_text.lower()
            else:
                # For MCQ, compare letters
                is_correct = predicted.upper() == correct_answer.upper()

            if is_correct:
                correct_count += 1

            results.append(ExperimentResult(
                question_id=question.id,
                question=question.question,
                correct_answer=correct_answer,
                predicted_answer=predicted,
                is_correct=is_correct,
                raw_response=raw_response,
                metadata=metadata,
            ))

        duration = time.time() - start_time
        accuracy = correct_count / len(questions) if questions else 0

        summary = ExperimentSummary(
            experiment_name=experiment_name,
            dataset=dataset_name,
            retriever=retriever.__class__.__name__ if retriever else None,
            corpus=getattr(retriever, 'corpus_name', None) if retriever else None,
            k=k,
            total_questions=len(questions),
            correct=correct_count,
            accuracy=accuracy,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
        )

        print(f"\n{'='*60}")
        print(f"Results: {correct_count}/{len(questions)} correct ({accuracy:.1%})")
        print(f"Duration: {duration:.1f} seconds")
        print(f"{'='*60}\n")

        # Save results
        if save_results:
            self._save_results(experiment_name, dataset_name, k, results, summary)

        return summary

    def _save_results(
        self,
        experiment_name: str,
        dataset_name: str,
        k: Optional[int],
        results: List[ExperimentResult],
        summary: ExperimentSummary,
    ) -> None:
        """Save experiment results to files."""
        # Create experiment directory
        exp_dir = RESULTS_DIR / experiment_name.replace(" ", "_").lower()
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        k_suffix = f"_k{k}" if k else ""
        results_file = exp_dir / f"{dataset_name}{k_suffix}_results.json"

        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)

        # Save summary
        summary_file = exp_dir / f"{dataset_name}{k_suffix}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2)

        print(f"Results saved to {exp_dir}")

    def run_baseline(self, datasets: List[str] = None) -> Dict[str, ExperimentSummary]:
        """Run baseline experiments (no RAG, CoT only)."""
        if datasets is None:
            datasets = ["medqa", "pubmedqa"]

        summaries = {}
        for dataset in datasets:
            summary = self.run_experiment(
                experiment_name="baseline_cot",
                dataset_name=dataset,
                retriever=None,
                k=None,
            )
            summaries[dataset] = summary

        return summaries

    def run_rag_experiments(
        self,
        datasets: List[str] = None,
        retrievers: List[str] = None,
        corpora: List[str] = None,
        k_values: List[int] = None,
    ) -> Dict[str, ExperimentSummary]:
        """Run RAG experiments with various configurations."""
        if datasets is None:
            datasets = ["medqa", "pubmedqa"]
        if retrievers is None:
            retrievers = ["bm25"]
        if corpora is None:
            corpora = ["pubmed"]
        if k_values is None:
            k_values = RETRIEVAL_CONFIG["k_values"]

        summaries = {}

        for dataset in datasets:
            for retriever_name in retrievers:
                for corpus in corpora:
                    for k in k_values:
                        exp_name = f"{retriever_name}_{corpus}_k{k}"

                        try:
                            retriever = get_retriever(retriever_name, corpus)

                            summary = self.run_experiment(
                                experiment_name=exp_name,
                                dataset_name=dataset,
                                retriever=retriever,
                                k=k,
                            )

                            key = f"{dataset}_{exp_name}"
                            summaries[key] = summary

                        except Exception as e:
                            print(f"Error running {exp_name} on {dataset}: {e}")

        return summaries


def run_all_experiments(api_key: str = None) -> Dict[str, ExperimentSummary]:
    """Run all experiments for the replication study."""
    runner = ExperimentRunner(api_key=api_key)

    all_summaries = {}

    # Experiment 1: Baseline (No RAG)
    print("\n" + "="*80)
    print("EXPERIMENT 1: Baseline (CoT, No RAG)")
    print("="*80)
    baseline_summaries = runner.run_baseline()
    all_summaries.update({f"baseline_{k}": v for k, v in baseline_summaries.items()})

    # Experiment 2: BM25 Retrieval
    print("\n" + "="*80)
    print("EXPERIMENT 2: BM25 Retrieval")
    print("="*80)
    bm25_summaries = runner.run_rag_experiments(
        retrievers=["bm25"],
        corpora=["pubmed", "statpearls"],
        k_values=[4, 8, 16, 32],
    )
    all_summaries.update(bm25_summaries)

    # Save overall summary
    summary_file = RESULTS_DIR / "all_experiments_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({k: asdict(v) for k, v in all_summaries.items()}, f, indent=2)

    return all_summaries
