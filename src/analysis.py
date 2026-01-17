"""
Analysis and visualization for MedRAG replication study.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import pandas as pd

from .config import RESULTS_DIR


def load_all_summaries() -> pd.DataFrame:
    """Load all experiment summaries into a DataFrame."""
    summaries = []

    # Look for summary files in all subdirectories
    for summary_file in RESULTS_DIR.glob("**/*_summary.json"):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            summaries.append(summary)

    if not summaries:
        print("No experiment summaries found.")
        return pd.DataFrame()

    df = pd.DataFrame(summaries)
    return df


def load_experiment_results(experiment_name: str, dataset: str,
                            k: Optional[int] = None) -> List[Dict]:
    """Load detailed results for a specific experiment."""
    exp_dir = RESULTS_DIR / experiment_name.replace(" ", "_").lower()

    k_suffix = f"_k{k}" if k else ""
    results_file = exp_dir / f"{dataset}{k_suffix}_results.json"

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return []

    with open(results_file, 'r') as f:
        return json.load(f)


def generate_comparison_table(summaries_df: pd.DataFrame) -> str:
    """Generate a markdown table comparing experiment results."""
    if summaries_df.empty:
        return "No results to display."

    # Pivot table for better comparison
    table_data = []

    for _, row in summaries_df.iterrows():
        table_data.append({
            "Experiment": row.get("experiment_name", "Unknown"),
            "Dataset": row.get("dataset", "Unknown"),
            "Retriever": row.get("retriever", "None"),
            "Corpus": row.get("corpus", "N/A"),
            "k": row.get("k", "N/A"),
            "Accuracy": f"{row.get('accuracy', 0):.1%}",
            "N": row.get("total_questions", 0),
        })

    df_table = pd.DataFrame(table_data)

    # Sort by dataset, then by accuracy
    df_table = df_table.sort_values(["Dataset", "Accuracy"], ascending=[True, False])

    return df_table.to_markdown(index=False)


def generate_results_summary() -> str:
    """Generate a comprehensive results summary."""
    summaries_df = load_all_summaries()

    if summaries_df.empty:
        return "No experiment results found."

    report = []
    report.append("# MedRAG Replication Study Results\n")
    report.append(f"Total experiments: {len(summaries_df)}\n")

    # Group by dataset
    for dataset in summaries_df["dataset"].unique():
        report.append(f"\n## {dataset.upper()}\n")

        dataset_df = summaries_df[summaries_df["dataset"] == dataset]

        # Best result
        best = dataset_df.loc[dataset_df["accuracy"].idxmax()]
        report.append(f"Best accuracy: {best['accuracy']:.1%} ({best['experiment_name']})\n")

        # Table for this dataset
        table_data = []
        for _, row in dataset_df.iterrows():
            k_str = str(row.get("k", "")) if pd.notna(row.get("k")) else "N/A"
            table_data.append({
                "Experiment": row.get("experiment_name", ""),
                "k": k_str,
                "Accuracy": f"{row.get('accuracy', 0):.1%}",
                "Correct": f"{row.get('correct', 0)}/{row.get('total_questions', 0)}",
            })

        df_table = pd.DataFrame(table_data)
        df_table = df_table.sort_values("Accuracy", ascending=False)
        report.append(df_table.to_markdown(index=False))
        report.append("\n")

    return "\n".join(report)


def compare_with_original(our_results: pd.DataFrame) -> str:
    """
    Compare our replication results with the original MedRAG paper.

    Original paper results (from Table 2, GPT-3.5-turbo):
    - MedQA: Baseline 50.6%, Best RAG ~53%
    - PubMedQA: Baseline 71.0%, Best RAG ~74%
    """
    original_results = {
        "medqa": {
            "baseline": 0.506,
            "best_rag": 0.53,
            "best_config": "MedCPT + Textbooks",
        },
        "pubmedqa": {
            "baseline": 0.710,
            "best_rag": 0.74,
            "best_config": "MedCPT + PubMed",
        },
    }

    report = []
    report.append("# Comparison with Original MedRAG Paper\n")
    report.append("Note: Original uses full datasets; our replication uses subsets.\n")

    for dataset in ["medqa", "pubmedqa"]:
        if dataset not in our_results["dataset"].values:
            continue

        report.append(f"\n## {dataset.upper()}\n")

        dataset_df = our_results[our_results["dataset"] == dataset]

        # Our baseline
        baseline_df = dataset_df[dataset_df["retriever"].isna()]
        our_baseline = baseline_df["accuracy"].iloc[0] if not baseline_df.empty else None

        # Our best RAG
        rag_df = dataset_df[dataset_df["retriever"].notna()]
        our_best_rag = rag_df["accuracy"].max() if not rag_df.empty else None

        orig = original_results[dataset]

        report.append("| Metric | Original Paper | Our Replication | Difference |")
        report.append("|--------|----------------|-----------------|------------|")

        if our_baseline is not None:
            diff = our_baseline - orig["baseline"]
            report.append(f"| Baseline | {orig['baseline']:.1%} | {our_baseline:.1%} | {diff:+.1%} |")

        if our_best_rag is not None:
            diff = our_best_rag - orig["best_rag"]
            report.append(f"| Best RAG | {orig['best_rag']:.1%} | {our_best_rag:.1%} | {diff:+.1%} |")

        report.append(f"\nOriginal best config: {orig['best_config']}")

    return "\n".join(report)


def error_analysis(experiment_name: str, dataset: str,
                   k: Optional[int] = None) -> Dict[str, Any]:
    """Analyze errors from a specific experiment."""
    results = load_experiment_results(experiment_name, dataset, k)

    if not results:
        return {}

    errors = [r for r in results if not r.get("is_correct", True)]
    correct = [r for r in results if r.get("is_correct", False)]

    analysis = {
        "total": len(results),
        "correct": len(correct),
        "errors": len(errors),
        "accuracy": len(correct) / len(results) if results else 0,
        "error_examples": errors[:5],  # First 5 errors for inspection
    }

    # Analyze prediction distribution
    predictions = [r.get("predicted_answer", "") for r in results]
    from collections import Counter
    analysis["prediction_distribution"] = dict(Counter(predictions))

    # Analyze empty/invalid predictions
    empty_preds = [r for r in results if not r.get("predicted_answer")]
    analysis["empty_predictions"] = len(empty_preds)

    return analysis


def save_final_report(output_path: Optional[Path] = None) -> None:
    """Generate and save the final analysis report."""
    if output_path is None:
        output_path = RESULTS_DIR / "final_report.md"

    summaries_df = load_all_summaries()

    report = []
    report.append(generate_results_summary())
    report.append("\n---\n")
    report.append(compare_with_original(summaries_df))

    with open(output_path, 'w') as f:
        f.write("\n".join(report))

    print(f"Report saved to {output_path}")
