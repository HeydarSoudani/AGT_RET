# AGT_RET - Agentic Retrieval Project

A retrieval and report generation system for TREC RAG 2024 and other information retrieval tasks.

## Project Structure

```
AGT_RET/
├── c0_corpus_dataset_preparation/  # Corpus and dataset preparation scripts
│   └── download_trec_rag_2024.py  # Download corpus, topics, and qrels
├── c1_retrieval_agents/            # Retrieval agent implementations
├── c2_report_generation/           # Report generation components
├── corpus_datasets/                # Downloaded corpora and datasets
│   └── trec_rag_2024/             # TREC RAG 2024 data
│       ├── src/                   # Source files (original downloads)
│       │   ├── topics.rag24.test.txt
│       │   ├── qrels.2024.txt
│       │   └── msmarco_v2.1_doc_segmented/
│       └── processed/             # Processed files
│           └── topics.rag24.test.jsonl
├── scripts/                        # Shell scripts for automation
│   └── download_trec_rag_2024.sh  # Shell wrapper for download script
├── utils/                          # Utility modules
│   └── trec_rag_2024_loader.py    # Data loading utilities
├── run_outputs/                    # Experiment outputs
├── script_logging/                 # Script logs
└── requirements.txt                # Python dependencies
```

## TREC RAG 2024 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download TREC RAG 2024 Data

#### Option A: Using Download Script (Recommended)

```bash
# Download all data (corpus, topics, and qrels)
./scripts/download_trec_rag_2024.sh --all

# Or download specific components:
./scripts/download_trec_rag_2024.sh --corpus  # Corpus only
./scripts/download_trec_rag_2024.sh --topics  # Topics only (auto-downloads and converts to JSONL)
./scripts/download_trec_rag_2024.sh --qrels   # Qrels only

# With custom directories:
./scripts/download_trec_rag_2024.sh --topics --topics-dir /custom/path
./scripts/download_trec_rag_2024.sh --all --base-dir /custom/path

# Show help
./scripts/download_trec_rag_2024.sh --help
```

You can also call the Python script directly (the shell script is just a simple wrapper):
```bash
python c0_corpus_dataset_preparation/download_trec_rag_2024.py --topics
```

#### Option B: Using ir_datasets Directly

```python
import ir_datasets

# Load the dataset
dataset = ir_datasets.load('msmarco-v2.1-doc-segmented/trec-rag-2024')

# Iterate over corpus
for doc in dataset.docs_iter():
    print(f"Doc ID: {doc.doc_id}")
    print(f"Title: {doc.title}")
    print(f"Body: {doc.body[:100]}...")

# Iterate over topics/queries
for query in dataset.queries_iter():
    print(f"Query {query.query_id}: {query.text}")

# Iterate over qrels
for qrel in dataset.qrels_iter():
    print(f"{qrel.query_id} {qrel.doc_id} {qrel.relevance}")
```

### 3. Using the Data Loader

```python
from pathlib import Path
from utils.trec_rag_2024_loader import TRECRag2024Loader

# Initialize loader
loader = TRECRag2024Loader(use_ir_datasets=True)

# Load queries
queries = list(loader.load_queries())
print(f"Loaded {len(queries)} queries")

# Load qrels
qrels = list(loader.load_qrels())
print(f"Loaded {len(qrels)} qrels")

# Load corpus (note: this is very large!)
for i, doc in enumerate(loader.load_corpus()):
    print(f"Doc {i}: {doc.title}")
    if i >= 10:  # Preview first 10 docs
        break

# Save to files
loader.save_queries_to_file(queries, Path("corpus_datasets/trec_rag_2024/topics/queries.json"))
loader.save_qrels_to_file(qrels, Path("corpus_datasets/trec_rag_2024/qrels/qrels.txt"))
```

## TREC RAG 2024 Overview

### Corpus
- **Name**: MS MARCO V2.1 Segmented Document Corpus
- **Size**: ~124 million segments from ~12 million documents
- **Source**: Common Crawl web documents
- **Format**: JSONL with fields: doc_id, url, title, headings, body

### Topics/Queries
- Real-world information needs requiring retrieval-augmented generation
- Available through ir_datasets or TREC website

### Qrels (Relevance Judgments)
- Human-judged relevance assessments
- Format: `query_id iteration doc_id relevance`
- Relevance scale: 0 (not relevant) to 3 (perfectly relevant)

## Quick Start Example

```python
# 1. Load data
from utils.trec_rag_2024_loader import TRECRag2024Loader
import logging

logging.basicConfig(level=logging.INFO)
loader = TRECRag2024Loader()

# 2. Get queries and qrels
queries = list(loader.load_queries())
qrels = list(loader.load_qrels())

print(f"Loaded {len(queries)} queries and {len(qrels)} qrels")

# 3. Example query
query = queries[0]
print(f"\nExample query:")
print(f"ID: {query.query_id}")
print(f"Text: {query.query_text}")

# 4. Get qrels for this query
query_qrels = [q for q in qrels if q.query_id == query.query_id]
print(f"\nFound {len(query_qrels)} relevant documents for query {query.query_id}")
```

## Evaluation

### Using pytrec_eval

```python
import pytrec_eval
from utils.trec_rag_2024_loader import TRECRag2024Loader

# Load qrels
loader = TRECRag2024Loader()
qrels = list(loader.load_qrels())
qrels_dict = loader.get_qrels_dict(qrels)

# Your retrieval run (example format)
run = {
    "query_1": {
        "doc_1": 1.5,
        "doc_2": 1.2,
        "doc_3": 0.9
    }
}

# Evaluate
evaluator = pytrec_eval.RelevanceEvaluator(
    qrels_dict,
    {'ndcg', 'map', 'recall', 'P'}
)
results = evaluator.evaluate(run)

# Print results
for query_id, metrics in results.items():
    print(f"\nQuery {query_id}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
```

## Resources

- **TREC RAG 2024**: https://trec-rag.github.io/
- **MS MARCO**: https://microsoft.github.io/msmarco/
- **ir_datasets**: https://ir-datasets.com/
- **PyTerrier**: https://pyterrier.readthedocs.io/
- **pytrec_eval**: https://github.com/cvangysel/pytrec_eval

## Development

### Running Scripts

The shell script is a simple wrapper that calls the Python script. Use either:

```bash
# Using shell wrapper (easier to type)
./scripts/download_trec_rag_2024.sh --topics

# Or call Python directly
python c0_corpus_dataset_preparation/download_trec_rag_2024.py --topics

# Both accept the same arguments:
--all                 # Download everything
--corpus              # Corpus only
--topics              # Topics only (auto-downloads and converts to JSONL)
--qrels               # Qrels only
--base-dir PATH       # Custom base directory
--topics-dir PATH     # Custom topics directory
--help                # Show all options
```

### Logging

All scripts log to the `script_logging/` directory. Check logs for detailed information about downloads and processing.

## License

This project structure is for research and educational purposes.

## Contributing

Feel free to add your retrieval agents and report generation components to the respective directories.
