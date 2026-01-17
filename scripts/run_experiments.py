#!/usr/bin/env python3
"""
Main script to run all MedRAG replication experiments.

Usage:
    python scripts/run_experiments.py --all
    python scripts/run_experiments.py --baseline
    python scripts/run_experiments.py --bm25 --dataset medqa
    python scripts/run_experiments.py --medcpt --k 8 16
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OPENAI_API_KEY
from src.data_loader import BenchmarkDataset
from src.corpus_downloader import download_all_corpora, create_sample_corpus
from src.experiment import ExperimentRunner, run_all_experiments
from src.retrieval import get_retriever, BM25Retriever, DenseRetriever, HybridRetriever
from src.analysis import generate_results_summary, save_final_report


def setup_data(use_samples: bool = False):
    """Download benchmark and corpus data."""
    print("\n" + "="*80)
    print("SETTING UP DATA")
    print("="*80 + "\n")

    # Download benchmark
    benchmark = BenchmarkDataset()
    benchmark.download_benchmark()

    # Download corpora
    if use_samples:
        print("\nCreating sample corpora for testing...")
        create_sample_corpus("pubmed", num_docs=5000)
        create_sample_corpus("statpearls", num_docs=2000)
    else:
        print("\nDownloading corpora (this may take a while)...")
        download_all_corpora(use_samples=False)


def run_baseline_experiments(runner: ExperimentRunner, datasets: list):
    """Run baseline (no RAG) experiments."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: BASELINE (CoT, No RAG)")
    print("="*80 + "\n")

    summaries = {}
    for dataset in datasets:
        summary = runner.run_experiment(
            experiment_name="baseline_cot",
            dataset_name=dataset,
            retriever=None,
            k=None,
        )
        summaries[f"baseline_{dataset}"] = summary

    return summaries


def run_bm25_experiments(runner: ExperimentRunner, datasets: list,
                         corpora: list, k_values: list):
    """Run BM25 retrieval experiments."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: BM25 RETRIEVAL")
    print("="*80 + "\n")

    summaries = {}
    for corpus in corpora:
        print(f"\nInitializing BM25 retriever for {corpus}...")
        try:
            retriever = BM25Retriever(corpus)
            retriever.load_corpus()

            for dataset in datasets:
                for k in k_values:
                    exp_name = f"bm25_{corpus}"
                    summary = runner.run_experiment(
                        experiment_name=exp_name,
                        dataset_name=dataset,
                        retriever=retriever,
                        k=k,
                    )
                    summaries[f"{dataset}_{exp_name}_k{k}"] = summary

        except Exception as e:
            print(f"Error with BM25 on {corpus}: {e}")

    return summaries


def run_medcpt_experiments(runner: ExperimentRunner, datasets: list,
                           corpora: list, k_values: list):
    """Run MedCPT (dense retrieval) experiments."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: MedCPT RETRIEVAL")
    print("="*80 + "\n")

    summaries = {}
    for corpus in corpora:
        print(f"\nInitializing MedCPT retriever for {corpus}...")
        try:
            retriever = DenseRetriever(corpus)
            retriever.load_corpus_and_index()

            for dataset in datasets:
                for k in k_values:
                    exp_name = f"medcpt_{corpus}"
                    summary = runner.run_experiment(
                        experiment_name=exp_name,
                        dataset_name=dataset,
                        retriever=retriever,
                        k=k,
                    )
                    summaries[f"{dataset}_{exp_name}_k{k}"] = summary

        except Exception as e:
            print(f"Error with MedCPT on {corpus}: {e}")

    return summaries


def run_hybrid_experiments(runner: ExperimentRunner, datasets: list,
                           corpora: list, k: int = 32):
    """Run hybrid RRF experiments."""
    print("\n" + "="*80)
    print("EXPERIMENT 4: HYBRID (RRF) RETRIEVAL")
    print("="*80 + "\n")

    summaries = {}
    for corpus in corpora:
        print(f"\nInitializing Hybrid retriever for {corpus}...")
        try:
            retriever = HybridRetriever(corpus)

            for dataset in datasets:
                exp_name = f"hybrid_{corpus}"
                summary = runner.run_experiment(
                    experiment_name=exp_name,
                    dataset_name=dataset,
                    retriever=retriever,
                    k=k,
                )
                summaries[f"{dataset}_{exp_name}_k{k}"] = summary

        except Exception as e:
            print(f"Error with Hybrid on {corpus}: {e}")

    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Run MedRAG replication experiments"
    )

    # Experiment selection
    parser.add_argument("--all", action="store_true",
                        help="Run all experiments")
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline (no RAG) experiments")
    parser.add_argument("--bm25", action="store_true",
                        help="Run BM25 retrieval experiments")
    parser.add_argument("--medcpt", action="store_true",
                        help="Run MedCPT retrieval experiments")
    parser.add_argument("--hybrid", action="store_true",
                        help="Run hybrid RRF experiments")

    # Configuration
    parser.add_argument("--dataset", nargs="+", default=["medqa", "pubmedqa"],
                        help="Datasets to use")
    parser.add_argument("--corpus", nargs="+", default=["pubmed", "statpearls"],
                        help="Corpora to use")
    parser.add_argument("--k", nargs="+", type=int, default=[4, 8, 16, 32],
                        help="Number of snippets to retrieve")

    # Setup options
    parser.add_argument("--setup", action="store_true",
                        help="Download data before running experiments")
    parser.add_argument("--use-samples", action="store_true",
                        help="Use sample corpora instead of full downloads")

    # Output
    parser.add_argument("--report", action="store_true",
                        help="Generate final report after experiments")

    args = parser.parse_args()

    # Check API key
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set.")
        print("Set it in .env file or as environment variable.")
        sys.exit(1)

    # Setup data if requested
    if args.setup:
        setup_data(use_samples=args.use_samples)

    # Initialize runner
    runner = ExperimentRunner()

    all_summaries = {}

    # Run experiments based on flags
    if args.all:
        args.baseline = True
        args.bm25 = True
        args.medcpt = True
        args.hybrid = True

    if args.baseline:
        summaries = run_baseline_experiments(runner, args.dataset)
        all_summaries.update(summaries)

    if args.bm25:
        summaries = run_bm25_experiments(
            runner, args.dataset, args.corpus, args.k
        )
        all_summaries.update(summaries)

    if args.medcpt:
        summaries = run_medcpt_experiments(
            runner, args.dataset, args.corpus, args.k
        )
        all_summaries.update(summaries)

    if args.hybrid:
        summaries = run_hybrid_experiments(
            runner, args.dataset, args.corpus, k=32
        )
        all_summaries.update(summaries)

    # Generate report
    if args.report or all_summaries:
        print("\n" + "="*80)
        print("GENERATING REPORT")
        print("="*80 + "\n")
        print(generate_results_summary())
        save_final_report()

    if not any([args.baseline, args.bm25, args.medcpt, args.hybrid]):
        parser.print_help()


if __name__ == "__main__":
    main()
