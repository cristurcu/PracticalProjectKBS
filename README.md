# MedRAG Replication Study

A partial replication of **"Benchmarking Retrieval-Augmented Generation for Medicine"** (Xiong et al., ACL 2024) for an academic technical report.

## Paper Reference

- **Original Paper**: [MedRAG at ACL 2024 Findings](https://aclanthology.org/2024.findings-acl.372/)
- **Original Repository**: [Teddy-XiongGZ/MedRAG](https://github.com/Teddy-XiongGZ/MedRAG)
- **Benchmark**: [MIRAGE Benchmark](https://github.com/Teddy-XiongGZ/MIRAGE)

```bibtex
@inproceedings{xiong2024benchmarking,
  title={Benchmarking Retrieval-Augmented Generation for Medicine},
  author={Xiong, Guangzhi and Jin, Qiao and Lu, Zhiyong and Zhang, Aidong},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2024},
  year={2024}
}
```

---

## Overview

This project implements a reduced-scope replication of MedRAG to evaluate Retrieval-Augmented Generation (RAG) for medical question answering. The original paper benchmarks various RAG configurations across multiple medical QA datasets; this replication focuses on a subset to validate the core findings.

### Scope Comparison

| Aspect | Original MedRAG | This Replication |
|--------|-----------------|------------------|
| **Datasets** | 5 (MedQA, MedMCQA, PubMedQA, MMLU-Med, BioASQ) | 1 (MedQA) |
| **Questions** | 7,663 total | 300 (MedQA subset) |
| **LLMs** | 6 (GPT-4, GPT-3.5, Mixtral, LLaMA, etc.) | 1 (GPT-3.5-turbo) |
| **Retrievers** | 4 (BM25, Contriever, SPECTER, MedCPT) | 1 (BM25) |
| **Corpora** | 5 (PubMed, StatPearls, Textbooks, Wikipedia, MedCorp) | 2 (PubMed, StatPearls - sample) |
| **Snippet counts (k)** | 1, 2, 4, 8, 16, 32, 64 | 4, 8, 16, 32 |

---

## Results

### MedQA Performance (300 questions)

| Configuration | k | Accuracy | vs Baseline |
|--------------|---|----------|-------------|
| Baseline (CoT, no RAG) | - | 66.0% | - |
| BM25 + StatPearls | 4 | 66.0% | +0.0% |
| BM25 + StatPearls | 8 | 67.0% | +1.0% |
| **BM25 + StatPearls** | **16** | **67.3%** | **+1.3%** |
| BM25 + StatPearls | 32 | 65.0% | -1.0% |
| BM25 + PubMed | 4 | 65.7% | -0.3% |
| BM25 + PubMed | 8 | 63.7% | -2.3% |
| BM25 + PubMed | 16 | 64.0% | -2.0% |
| BM25 + PubMed | 32 | 63.0% | -3.0% |

### Comparison with Original Paper

| Metric | Original Paper | This Replication | Difference |
|--------|----------------|------------------|------------|
| Baseline (GPT-3.5) | 50.6% | 66.0% | +15.4% |
| Best RAG | 53.0% | 67.3% | +14.3% |
| RAG Improvement | +2.4% | +1.3% | -1.1% |

**Note**: Higher baseline likely due to GPT-3.5-turbo improvements since the original paper (2024).

### Key Findings

1. **RAG provides modest improvement** (+1.3% with optimal configuration)
2. **Corpus choice matters**: StatPearls (clinical articles) outperforms PubMed (abstracts) for MedQA
3. **Optimal snippet count exists**: k=16 works best; more snippets (k=32) can hurt performance
4. **Confirms paper's trend**: RAG helps, but gains are incremental for medical QA

---

## Implementation Details

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MedRAG Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  1. Question Input                                          │
│         ↓                                                   │
│  2. Retrieval (BM25)                                        │
│     - Query corpus (PubMed or StatPearls)                   │
│     - Return top-k relevant snippets                        │
│         ↓                                                   │
│  3. Context Augmentation                                    │
│     - Format retrieved documents                            │
│     - Truncate to fit context window                        │
│         ↓                                                   │
│  4. LLM Generation (GPT-3.5-turbo)                         │
│     - Chain-of-thought prompting                            │
│     - JSON-formatted response                               │
│         ↓                                                   │
│  5. Answer Extraction                                       │
│     - Parse predicted answer (A/B/C/D)                      │
│     - Compare with ground truth                             │
└─────────────────────────────────────────────────────────────┘
```

### Components

#### 1. Data Loading (`src/data_loader.py`)
- Loads MIRAGE benchmark from the original MedRAG repository
- Supports MedQA (multiple choice) and PubMedQA (yes/no/maybe) formats
- Configurable subset sampling for reduced experiments

#### 2. Retrieval System (`src/retrieval.py`)
- **BM25Retriever**: Lexical retrieval using Okapi BM25 algorithm
- **DenseRetriever**: Neural retrieval using MedCPT embeddings (not used due to dependency issues)
- **HybridRetriever**: Reciprocal Rank Fusion of BM25 + Dense (not used)
- **PrecomputedRetriever**: For using pre-computed snippet files

#### 3. LLM Interface (`src/llm.py`)
- OpenAI GPT-3.5-turbo integration
- Chain-of-thought prompting with JSON output format
- Token counting and context truncation
- Rate limiting and retry logic

#### 4. Prompt Templates (`src/templates.py`)
Based on the original MedRAG paper:

**Baseline (CoT) Prompt**:
```
Given the following medical question, think step-by-step and select the correct answer.

## Question
{question}

## Options
A. {option_a}
B. {option_b}
...

Provide your response in JSON format:
{"step_by_step_thinking": "...", "answer_choice": "A/B/C/D"}
```

**RAG Prompt**:
```
Given the following medical question and relevant documents, think step-by-step...

## Relevant Documents
Document [1] (Title: ...)
{content}
...

## Question
{question}

## Options
...
```

#### 5. Experiment Runner (`src/experiment.py`)
- Orchestrates end-to-end experiments
- Tracks accuracy, timing, and per-question results
- Saves detailed JSON results for analysis

#### 6. Analysis (`src/analysis.py`)
- Generates summary tables and comparison reports
- Markdown output for easy inclusion in reports

---

## Project Structure

```
medrag-replication/
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration and paths
│   ├── data_loader.py       # Benchmark data loading
│   ├── retrieval.py         # BM25, Dense, Hybrid retrievers
│   ├── llm.py               # OpenAI GPT interface
│   ├── templates.py         # Prompt templates
│   ├── experiment.py        # Experiment orchestration
│   ├── analysis.py          # Results analysis
│   └── corpus_downloader.py # Corpus management
├── scripts/
│   ├── run_experiments.py   # Main CLI for experiments
│   └── quick_test.py        # Validation script
├── data/
│   ├── benchmark/           # MIRAGE benchmark JSON
│   ├── corpus/              # Medical corpora (sample)
│   │   ├── pubmed/
│   │   └── statpearls/
│   └── indices/             # FAISS indices (if used)
├── results/                 # Experiment outputs
│   ├── baseline_cot/
│   ├── bm25_pubmed/
│   └── bm25_statpearls/
├── pyproject.toml           # Poetry dependencies
├── poetry.lock
├── .env.example
└── README.md
```

---

## Installation

### Requirements
- Python 3.10+ (required for faiss-cpu)
- Poetry (dependency management)
- OpenAI API key

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd medrag-replication

# Install dependencies with Poetry
poetry install

# Configure environment
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your-key-here
```

### Dependencies

Key packages:
- `openai` - LLM API
- `rank-bm25` - BM25 retrieval
- `sentence-transformers` - Dense embeddings (MedCPT)
- `faiss-cpu` - Vector similarity search
- `datasets` - HuggingFace data loading
- `pandas`, `numpy` - Data processing
- `jinja2` - Prompt templating

---

## Usage

### Quick Validation
```bash
poetry run python scripts/quick_test.py
```

### Run Experiments

```bash
# Baseline only (no RAG)
poetry run python scripts/run_experiments.py --baseline --dataset medqa

# BM25 with all k values
poetry run python scripts/run_experiments.py --bm25 --dataset medqa --corpus pubmed statpearls

# Specific k values
poetry run python scripts/run_experiments.py --bm25 --dataset medqa --k 8 16

# Setup sample corpora first
poetry run python scripts/run_experiments.py --setup --use-samples

# Generate report
poetry run python scripts/run_experiments.py --report
```

### Command Options

| Flag | Description |
|------|-------------|
| `--baseline` | Run baseline (no RAG) experiments |
| `--bm25` | Run BM25 retrieval experiments |
| `--medcpt` | Run MedCPT dense retrieval (requires compatible NumPy) |
| `--hybrid` | Run hybrid RRF experiments |
| `--dataset` | Datasets to use: `medqa`, `pubmedqa` |
| `--corpus` | Corpora to use: `pubmed`, `statpearls` |
| `--k` | Snippet counts: e.g., `4 8 16 32` |
| `--setup` | Download/create data before running |
| `--use-samples` | Use sample corpora (100 docs) instead of full |
| `--report` | Generate final report |

---

## Limitations

### Technical Limitations
1. **MedCPT not tested**: NumPy 2.x / PyTorch 2.2.x / faiss-cpu version conflict on macOS x86_64 prevented dense retrieval experiments
2. **Sample corpora**: Used 100-document samples instead of full corpora (millions of documents)
3. **Subset of questions**: 300 MedQA questions instead of full 1,273

### Dataset Limitations
1. **PubMedQA excluded**: The MIRAGE benchmark file lacks the abstract context required for meaningful PubMedQA evaluation (model defaulted to "maybe" for 75% of questions)
2. **Single LLM**: Only GPT-3.5-turbo tested (original paper tested 6 models)

### Findings Limitations
1. **Higher baseline than paper**: GPT-3.5-turbo has improved since the original paper, making direct comparison difficult
2. **Random subset**: Results may vary with different random samples

---

## API Costs

Approximate costs using GPT-3.5-turbo:
- ~$0.50-1.00 per 100 questions
- Full replication (~300 questions × 9 configs): ~$15-20

---

## Development

```bash
# Run tests
poetry run pytest

# Format code
poetry run black src/ scripts/

# Lint code
poetry run ruff check src/ scripts/
```

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

- Original MedRAG paper authors (Xiong et al.)
- OpenAI for GPT-3.5-turbo API
- HuggingFace for datasets and model hosting
