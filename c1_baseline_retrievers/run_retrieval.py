#!/usr/bin/env python3
import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.general_utils import set_seed

# Import retrieval models and evaluation metrics
from src.retrieval_models import SingleStageRetrievalModel
from src.evaluation_metrics import compute_all_metrics, aggregate_metrics, load_qrels


def load_queries(query_file: str):
    """
    Load queries from JSONL file.

    Args:
        query_file: Path to query file in JSONL format

    Returns:
        List of query dictionaries with 'qid' and 'question' fields
    """
    queries = []
    with open(query_file, 'r') as f:
        for line in f:
            query = json.loads(line.strip())
            queries.append(query)
    return queries


def run_retrieval(args):
    print("=" * 80)
    print("Starting Retrieval Process")
    print("=" * 80)

    # === Read input data =====================
    print("\n[1/4] Loading queries...")
    queries = load_queries(args.query_file)
    print(f"Loaded {len(queries)} queries from {args.query_file}")

    # Load qrels if provided for evaluation
    qrels = None
    if args.qrels_file and os.path.exists(args.qrels_file):
        print(f"\n[2/4] Loading qrels from {args.qrels_file}...")
        qrels = load_qrels(args.qrels_file)
        print(f"Loaded qrels for {len(qrels)} queries")
    else:
        print("\n[2/4] No qrels file provided, skipping evaluation")

    # === Initialize retrieval model ==========
    print(f"\n[3/4] Initializing retrieval model: {args.retriever_name}...")
    retrieval_model = SingleStageRetrievalModel(args.device, args)
    print(f"Retrieval model initialized successfully")

    # === Retrieval ===========================
    print(f"\n[4/4] Running retrieval for {len(queries)} queries...")
    print(f"Retrieving top-{args.retrieval_topk} documents per query")

    # Prepare output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Open output file for writing results incrementally
    all_metrics = []

    with open(args.output_file, 'w') as out_f:
        for query_data in tqdm(queries, desc="Retrieving"):
            qid = query_data['qid']
            query_text = query_data['question']

            # Retrieve documents
            retrieved_results = retrieval_model.retrieve(query_text, top_k=args.retrieval_topk)

            # Extract document IDs from retrieved results
            # Assumes results are dictionaries with 'id' or 'docid' field
            # Adjust based on your corpus structure
            retrieved_doc_ids = []
            for doc in retrieved_results:
                if isinstance(doc, dict):
                    # Try different possible ID field names
                    doc_id = doc.get('id') or doc.get('docid') or doc.get('_id') or doc.get('doc_id')
                    if doc_id:
                        retrieved_doc_ids.append(str(doc_id))
                else:
                    # If doc is not a dict, assume it's the ID itself
                    retrieved_doc_ids.append(str(doc))

            # Prepare result entry
            result_entry = {
                'qid': qid,
                'query': query_text,
                'retrieved_doc_ids': retrieved_doc_ids,
                'num_retrieved': len(retrieved_doc_ids)
            }

            # === Evaluation ==========================
            # Compute metrics if qrels available
            if qrels and qid in qrels:
                relevant_docs = qrels[qid]
                # Filter to only positive relevance (relevance > 0)
                relevant_docs_positive = {doc_id: rel for doc_id, rel in relevant_docs.items() if rel > 0}

                if len(relevant_docs_positive) > 0:
                    metrics = compute_all_metrics(
                        retrieved_doc_ids,
                        relevant_docs_positive,
                        k_values=args.eval_k_values
                    )
                    result_entry['metrics'] = metrics
                    all_metrics.append(metrics)

            # Write result to file (one JSON object per line)
            out_f.write(json.dumps(result_entry) + '\n')
            out_f.flush()  # Ensure it's written immediately

    print(f"\nRetrieval results saved to: {args.output_file}")

    # === Aggregate and Save Metrics ==========================
    if len(all_metrics) > 0:
        print("\n" + "=" * 80)
        print("Computing Aggregate Metrics")
        print("=" * 80)

        aggregated_metrics = aggregate_metrics(all_metrics)

        # Print metrics
        print("\nAggregated Evaluation Metrics:")
        print("-" * 80)
        for metric_name, value in sorted(aggregated_metrics.items()):
            print(f"{metric_name:15s}: {value:.4f}")

        # Save aggregated metrics
        metrics_output_file = args.output_file.replace('.jsonl', '_metrics.json')
        with open(metrics_output_file, 'w') as f:
            json.dump(aggregated_metrics, f, indent=2)
        print(f"\nAggregated metrics saved to: {metrics_output_file}")

    print("\n" + "=" * 80)
    print("Retrieval Complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval on a query set")

    # === Input/Output arguments ===
    parser.add_argument('--query_file', type=str, required=True,
                        help='Path to query file (JSONL format with qid and question fields)')
    parser.add_argument('--corpus_path', type=str, required=True,
                        help='Path to corpus file (JSONL format)')
    parser.add_argument('--qrels_file', type=str, default=None,
                        help='Path to qrels file for evaluation (TREC format)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to output file for retrieval results')

    # === Retrieval model arguments ===
    parser.add_argument('--retriever_name', type=str, required=True,
                        choices=['bm25', 'contriever', 'dpr', 'e5', 'bge', 'rerank_l6', 'rerank_l12'],
                        help='Name of the retriever to use')
    parser.add_argument('--index_dir', type=str, required=True,
                        help='Directory containing pre-built indices')
    parser.add_argument('--retrieval_topk', type=int, default=100,
                        help='Number of documents to retrieve per query')

    # === BM25-specific arguments ===
    parser.add_argument('--bm25_k1', type=float, default=0.9,
                        help='BM25 k1 parameter')
    parser.add_argument('--bm25_b', type=float, default=0.4,
                        help='BM25 b parameter')

    # === Dense retriever arguments ===
    parser.add_argument('--retrieval_query_max_length', type=int, default=256,
                        help='Maximum query length for dense retrievers')
    parser.add_argument('--retrieval_use_fp16', action='store_true',
                        help='Use FP16 for dense retriever inference')
    parser.add_argument('--retrieval_batch_size', type=int, default=32,
                        help='Batch size for dense retriever encoding')
    parser.add_argument('--faiss_gpu', action='store_true',
                        help='Use GPU for FAISS index')

    # === Evaluation arguments ===
    parser.add_argument('--eval_k_values', type=int, nargs='+', default=[1, 3, 5, 10, 20, 100],
                        help='K values for evaluation metrics (e.g., P@K, NDCG@K)')

    # === General arguments ===
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # === Define CUDA device =======
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPUs detected.")

    print(f"\nRunning on device: {args.device}")
    print(f"Retriever: {args.retriever_name}")
    print(f"Top-K: {args.retrieval_topk}")
    print(f"Random seed: {args.seed}\n")

    # === Run ========================
    set_seed(args.seed)
    run_retrieval(args)