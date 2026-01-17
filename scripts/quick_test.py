#!/usr/bin/env python3
"""
Quick test script to verify the implementation works.
Runs a minimal experiment with a few questions.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_data_loading():
    """Test benchmark data loading."""
    print("\n1. Testing data loading...")
    from src.data_loader import BenchmarkDataset

    benchmark = BenchmarkDataset()
    benchmark.download_benchmark()

    # Load a few questions from MedQA
    medqa_questions = benchmark.get_dataset("medqa", subset_size=5)
    print(f"   Loaded {len(medqa_questions)} MedQA questions")

    if medqa_questions:
        q = medqa_questions[0]
        print(f"   Sample question: {q.question[:100]}...")
        print(f"   Options: {list(q.options.keys())}")
        print(f"   Answer: {q.answer}")

    # Load a few questions from PubMedQA
    pubmedqa_questions = benchmark.get_dataset("pubmedqa", subset_size=5)
    print(f"   Loaded {len(pubmedqa_questions)} PubMedQA questions")

    return True


def test_llm_interface():
    """Test LLM interface with a simple question."""
    print("\n2. Testing LLM interface...")
    from src.config import OPENAI_API_KEY
    from src.llm import LLMInterface

    if not OPENAI_API_KEY:
        print("   Skipping LLM test - no API key set")
        return False

    llm = LLMInterface()

    # Test with a simple medical question
    question = "What is the most common cause of community-acquired pneumonia in adults?"
    options = {
        "A": "Staphylococcus aureus",
        "B": "Streptococcus pneumoniae",
        "C": "Haemophilus influenzae",
        "D": "Klebsiella pneumoniae",
    }

    predicted, response, metadata = llm.answer_question(
        question=question,
        options=options,
        documents=None,
        dataset_type="mcq",
    )

    print(f"   Question: {question[:80]}...")
    print(f"   Predicted: {predicted}")
    print(f"   Expected: B (Streptococcus pneumoniae)")
    print(f"   Tokens used: {metadata.get('prompt_tokens', 'N/A')}")

    return True


def test_sample_corpus():
    """Test corpus creation."""
    print("\n3. Testing sample corpus creation...")
    from src.corpus_downloader import create_sample_corpus

    create_sample_corpus("pubmed", num_docs=100)
    create_sample_corpus("statpearls", num_docs=100)

    print("   Sample corpora created successfully")
    return True


def test_bm25_retrieval():
    """Test BM25 retrieval."""
    print("\n4. Testing BM25 retrieval...")
    from src.retrieval import BM25Retriever

    # Make sure sample corpus exists
    from src.corpus_downloader import create_sample_corpus
    create_sample_corpus("pubmed", num_docs=100)

    retriever = BM25Retriever("pubmed")
    retriever.load_corpus()

    results = retriever.retrieve("diabetes treatment options", k=5)
    print(f"   Retrieved {len(results)} documents")

    if results:
        print(f"   Top result: {results[0].title[:50]}...")
        print(f"   Score: {results[0].score:.4f}")

    return True


def test_baseline_experiment():
    """Test running a baseline experiment with 3 questions."""
    print("\n5. Testing baseline experiment (3 questions)...")
    from src.config import OPENAI_API_KEY

    if not OPENAI_API_KEY:
        print("   Skipping experiment test - no API key set")
        return False

    from src.experiment import ExperimentRunner

    runner = ExperimentRunner()
    summary = runner.run_experiment(
        experiment_name="test_baseline",
        dataset_name="medqa",
        retriever=None,
        k=None,
        subset_size=3,
    )

    print(f"   Accuracy: {summary.accuracy:.1%}")
    print(f"   Correct: {summary.correct}/{summary.total_questions}")

    return True


def main():
    print("="*60)
    print("MedRAG Replication - Quick Test Suite")
    print("="*60)

    tests = [
        ("Data Loading", test_data_loading),
        ("Sample Corpus", test_sample_corpus),
        ("BM25 Retrieval", test_bm25_retrieval),
        ("LLM Interface", test_llm_interface),
        ("Baseline Experiment", test_baseline_experiment),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"   ERROR: {e}")
            results[name] = False

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
