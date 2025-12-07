# Baseline Retrievers

This directory contains code for running baseline retrieval models and evaluating their performance.

## Overview

The retrieval pipeline consists of:
1. **Loading queries** from JSONL files
2. **Initializing retrieval models** (BM25, Dense Retrievers, Rerankers)
3. **Retrieving documents** for each query
4. **Evaluating results** using standard IR metrics
5. **Saving results** incrementally to output files

## File Structure

```
c1_baseline_retrievers/
├── run_retrieval.py           # Main script to run retrieval
├── src/
│   ├── retrieval_models.py    # Retrieval model classes
│   ├── local_retrievers.py    # Base retriever implementations
│   ├── evaluation_metrics.py  # Evaluation metrics (P@K, R@K, NDCG@K, MAP, MRR, etc.)
│   └── index_builder.py       # Index building utilities
└── README.md                  # This file
```

## Retrieval Models

Supported retrieval models:
- **BM25**: Sparse retrieval using Lucene/Pyserini
- **Dense Retrievers**:
  - `contriever`: Facebook Contriever
  - `dpr`: Dense Passage Retrieval
  - `e5`: E5 embeddings
  - `bge`: BGE embeddings
- **Rerankers**:
  - `rerank_l6`: Cross-encoder reranking (6 layers)
  - `rerank_l12`: Cross-encoder reranking (12 layers)

## Usage

### Basic Command

```bash
python run_retrieval.py \
    --query_file <path_to_queries.jsonl> \
    --corpus_path <path_to_corpus.jsonl> \
    --qrels_file <path_to_qrels.txt> \
    --output_file <path_to_output.jsonl> \
    --retriever_name <retriever_name> \
    --index_dir <path_to_index_directory> \
    --retrieval_topk 100
```

### Example: BM25 Retrieval

```bash
python run_retrieval.py \
    --query_file corpus_datasets/trec_rag_2024/processed/topics.rag24.test.jsonl \
    --corpus_path corpus_datasets/enwiki_20251001.jsonl \
    --qrels_file corpus_datasets/trec_rag_2024/src/2024-retrieval-conditions-qrels.txt \
    --output_file run_outputs/retrieval_results_bm25.jsonl \
    --retriever_name bm25 \
    --index_dir /projects/0/prjs0834/heydars/INDICES \
    --retrieval_topk 100 \
    --bm25_k1 0.9 \
    --bm25_b 0.4 \
    --device 0 \
    --seed 42
```

### Example: Dense Retrieval (Contriever)

```bash
python run_retrieval.py \
    --query_file corpus_datasets/trec_rag_2024/processed/topics.rag24.test.jsonl \
    --corpus_path corpus_datasets/enwiki_20251001.jsonl \
    --qrels_file corpus_datasets/trec_rag_2024/src/2024-retrieval-conditions-qrels.txt \
    --output_file run_outputs/retrieval_results_contriever.jsonl \
    --retriever_name contriever \
    --index_dir /projects/0/prjs0834/heydars/INDICES \
    --retrieval_topk 100 \
    --retrieval_query_max_length 256 \
    --retrieval_batch_size 32 \
    --retrieval_use_fp16 \
    --faiss_gpu \
    --device 0 \
    --seed 42
```

### Example: Reranking

```bash
python run_retrieval.py \
    --query_file corpus_datasets/trec_rag_2024/processed/topics.rag24.test.jsonl \
    --corpus_path corpus_datasets/enwiki_20251001.jsonl \
    --qrels_file corpus_datasets/trec_rag_2024/src/2024-retrieval-conditions-qrels.txt \
    --output_file run_outputs/retrieval_results_rerank.jsonl \
    --retriever_name rerank_l6 \
    --index_dir /projects/0/prjs0834/heydars/INDICES \
    --retrieval_topk 100 \
    --device 0 \
    --seed 42
```

## Arguments

### Required Arguments

- `--query_file`: Path to query file (JSONL format with 'qid' and 'question' fields)
- `--corpus_path`: Path to corpus file (JSONL format)
- `--output_file`: Path to output file for retrieval results
- `--retriever_name`: Name of the retriever (bm25, contriever, dpr, e5, bge, rerank_l6, rerank_l12)
- `--index_dir`: Directory containing pre-built indices

### Optional Arguments

- `--qrels_file`: Path to qrels file for evaluation (TREC format) - if not provided, only retrieval is performed
- `--retrieval_topk`: Number of documents to retrieve per query (default: 100)
- `--device`: CUDA device ID (default: 0)
- `--seed`: Random seed for reproducibility (default: 42)

### BM25-Specific Arguments

- `--bm25_k1`: BM25 k1 parameter (default: 0.9)
- `--bm25_b`: BM25 b parameter (default: 0.4)

### Dense Retriever Arguments

- `--retrieval_query_max_length`: Maximum query length (default: 256)
- `--retrieval_use_fp16`: Use FP16 for inference (flag)
- `--retrieval_batch_size`: Batch size for encoding (default: 32)
- `--faiss_gpu`: Use GPU for FAISS index (flag)

### Evaluation Arguments

- `--eval_k_values`: K values for evaluation metrics (default: [1, 3, 5, 10, 20, 100])

## Input File Formats

### Query File (JSONL)

Each line should be a JSON object with:
```json
{"qid": "2024-145979", "question": "what is vicarious trauma and how can it be coped with?"}
```

### Corpus File (JSONL)

Each line should be a JSON object with document content. The exact format depends on your corpus structure.

### Qrels File (TREC Format)

Space-separated format: `qid iteration docid relevance`
```
2024-105741 1 msmarco_v2.1_doc_00_125364462#6_229054655 0
2024-105741 1 msmarco_v2.1_doc_00_1534870566#5_2687183111 1
```

## Output Format

### Retrieval Results (JSONL)

Each line contains:
```json
{
  "qid": "2024-145979",
  "query": "what is vicarious trauma and how can it be coped with?",
  "retrieved_doc_ids": ["doc1", "doc2", ...],
  "num_retrieved": 100,
  "metrics": {
    "MRR": 0.5,
    "AP": 0.45,
    "P@1": 1.0,
    "P@5": 0.8,
    "NDCG@10": 0.75,
    ...
  }
}
```

### Aggregated Metrics (JSON)

Saved to `<output_file>_metrics.json`:
```json
{
  "MRR": 0.45,
  "AP": 0.42,
  "P@1": 0.38,
  "P@5": 0.52,
  "NDCG@10": 0.58,
  ...
}
```

## Evaluation Metrics

The following metrics are computed:

- **MRR (Mean Reciprocal Rank)**: Reciprocal rank of first relevant document
- **AP (Average Precision)**: Average of precision values at relevant document positions
- **P@K (Precision@K)**: Fraction of top-K documents that are relevant
- **R@K (Recall@K)**: Fraction of relevant documents in top-K
- **F1@K**: Harmonic mean of P@K and R@K
- **NDCG@K (Normalized Discounted Cumulative Gain)**: Ranking quality metric
- **Hit@K**: Binary indicator if any relevant document in top-K

## Running with SLURM

Use the provided shell script:

```bash
sbatch scripts/run_retrieval.sh
```

Edit the script to customize:
- Resource allocation (GPUs, memory, time)
- Input/output paths
- Retriever settings

## Notes

- Results are written **incrementally** (one query at a time) to avoid losing progress if the job fails
- For dense retrievers, ensure you have enough GPU memory (use `--retrieval_use_fp16` to reduce memory usage)
- For rerankers, the first-stage retrieval is BM25 with 1000 candidates, then reranked to top-K
- The corpus structure may vary; adjust the document ID extraction logic in `run_retrieval.py` if needed (lines 84-91)
