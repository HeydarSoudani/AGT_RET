#!/usr/bin/env python3
"""
Evaluation metrics for information retrieval tasks.
Includes: Precision, Recall, F1, MRR, MAP, NDCG, Hit Rate
"""
import numpy as np
from typing import List, Dict, Set, Union


def precision_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
    """
    Precision@K: Fraction of retrieved documents that are relevant.

    Args:
        retrieved_docs: List of retrieved document IDs (ordered by rank)
        relevant_docs: Set of relevant document IDs
        k: Cut-off rank

    Returns:
        Precision@K score
    """
    if k <= 0 or len(retrieved_docs) == 0:
        return 0.0

    retrieved_at_k = retrieved_docs[:k]
    relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant_docs)
    return relevant_retrieved / k


def recall_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
    """
    Recall@K: Fraction of relevant documents that are retrieved.

    Args:
        retrieved_docs: List of retrieved document IDs (ordered by rank)
        relevant_docs: Set of relevant document IDs
        k: Cut-off rank

    Returns:
        Recall@K score
    """
    if len(relevant_docs) == 0 or len(retrieved_docs) == 0:
        return 0.0

    retrieved_at_k = retrieved_docs[:k]
    relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant_docs)
    return relevant_retrieved / len(relevant_docs)


def f1_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
    """
    F1@K: Harmonic mean of Precision@K and Recall@K.

    Args:
        retrieved_docs: List of retrieved document IDs (ordered by rank)
        relevant_docs: Set of relevant document IDs
        k: Cut-off rank

    Returns:
        F1@K score
    """
    prec = precision_at_k(retrieved_docs, relevant_docs, k)
    rec = recall_at_k(retrieved_docs, relevant_docs, k)

    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def mean_reciprocal_rank(retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
    """
    MRR: Mean Reciprocal Rank - reciprocal of the rank of the first relevant document.

    Args:
        retrieved_docs: List of retrieved document IDs (ordered by rank)
        relevant_docs: Set of relevant document IDs

    Returns:
        MRR score (1/rank of first relevant doc, or 0 if none found)
    """
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant_docs:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
    """
    Average Precision: Average of precision values at each relevant document position.

    Args:
        retrieved_docs: List of retrieved document IDs (ordered by rank)
        relevant_docs: Set of relevant document IDs

    Returns:
        Average Precision score
    """
    if len(relevant_docs) == 0:
        return 0.0

    num_relevant = 0
    sum_precision = 0.0

    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant_docs:
            num_relevant += 1
            precision_at_rank = num_relevant / rank
            sum_precision += precision_at_rank

    if num_relevant == 0:
        return 0.0

    return sum_precision / len(relevant_docs)


def mean_average_precision(all_retrieved: List[List[str]], all_relevant: List[Set[str]]) -> float:
    """
    MAP: Mean Average Precision across multiple queries.

    Args:
        all_retrieved: List of retrieved document lists (one per query)
        all_relevant: List of relevant document sets (one per query)

    Returns:
        MAP score
    """
    if len(all_retrieved) == 0:
        return 0.0

    ap_scores = [average_precision(retr, rel) for retr, rel in zip(all_retrieved, all_relevant)]
    return np.mean(ap_scores)


def dcg_at_k(retrieved_docs: List[str], relevant_docs: Union[Set[str], Dict[str, int]], k: int) -> float:
    """
    DCG@K: Discounted Cumulative Gain.

    Args:
        retrieved_docs: List of retrieved document IDs (ordered by rank)
        relevant_docs: Set of relevant document IDs or Dict mapping doc_id to relevance score
        k: Cut-off rank

    Returns:
        DCG@K score
    """
    if k <= 0 or len(retrieved_docs) == 0:
        return 0.0

    dcg = 0.0
    for rank, doc in enumerate(retrieved_docs[:k], start=1):
        if isinstance(relevant_docs, dict):
            relevance = relevant_docs.get(doc, 0)
        else:
            relevance = 1 if doc in relevant_docs else 0

        dcg += (2 ** relevance - 1) / np.log2(rank + 1)

    return dcg


def ndcg_at_k(retrieved_docs: List[str], relevant_docs: Union[Set[str], Dict[str, int]], k: int) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain.

    Args:
        retrieved_docs: List of retrieved document IDs (ordered by rank)
        relevant_docs: Set of relevant document IDs or Dict mapping doc_id to relevance score
        k: Cut-off rank

    Returns:
        NDCG@K score
    """
    dcg = dcg_at_k(retrieved_docs, relevant_docs, k)

    # Compute ideal DCG (IDCG)
    if isinstance(relevant_docs, dict):
        ideal_docs = sorted(relevant_docs.keys(), key=lambda x: relevant_docs[x], reverse=True)
    else:
        ideal_docs = list(relevant_docs)

    idcg = dcg_at_k(ideal_docs, relevant_docs, k)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def hit_rate_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
    """
    Hit Rate@K (Success@K): Binary indicator of whether any relevant doc is in top-k.

    Args:
        retrieved_docs: List of retrieved document IDs (ordered by rank)
        relevant_docs: Set of relevant document IDs
        k: Cut-off rank

    Returns:
        1.0 if at least one relevant doc in top-k, else 0.0
    """
    if k <= 0 or len(retrieved_docs) == 0 or len(relevant_docs) == 0:
        return 0.0

    retrieved_at_k = set(retrieved_docs[:k])
    return 1.0 if len(retrieved_at_k & relevant_docs) > 0 else 0.0


def compute_all_metrics(retrieved_docs: List[str],
                        relevant_docs: Union[Set[str], Dict[str, int]],
                        k_values: List[int] = [1, 3, 5, 10, 20, 100]) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a single query.

    Args:
        retrieved_docs: List of retrieved document IDs (ordered by rank)
        relevant_docs: Set of relevant document IDs or Dict mapping doc_id to relevance score
        k_values: List of k values for which to compute metrics

    Returns:
        Dictionary of metric names to scores
    """
    metrics = {}

    # Convert dict to set for some metrics
    if isinstance(relevant_docs, dict):
        relevant_set = set(relevant_docs.keys())
    else:
        relevant_set = relevant_docs

    # MRR and AP don't depend on k
    metrics['MRR'] = mean_reciprocal_rank(retrieved_docs, relevant_set)
    metrics['AP'] = average_precision(retrieved_docs, relevant_set)

    # Compute metrics at different k values
    for k in k_values:
        metrics[f'P@{k}'] = precision_at_k(retrieved_docs, relevant_set, k)
        metrics[f'R@{k}'] = recall_at_k(retrieved_docs, relevant_set, k)
        metrics[f'F1@{k}'] = f1_at_k(retrieved_docs, relevant_set, k)
        metrics[f'NDCG@{k}'] = ndcg_at_k(retrieved_docs, relevant_docs, k)
        metrics[f'Hit@{k}'] = hit_rate_at_k(retrieved_docs, relevant_set, k)

    return metrics


def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple queries by computing means.

    Args:
        all_metrics: List of metric dictionaries (one per query)

    Returns:
        Dictionary with averaged metrics
    """
    if len(all_metrics) == 0:
        return {}

    aggregated = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        aggregated[metric_name] = np.mean(values)

    return aggregated


def load_qrels(qrels_file: str) -> Dict[str, Dict[str, int]]:
    """
    Load TREC-style qrels file.

    Format: qid iteration docid relevance
    Example: 2024-145979 1 msmarco_v2.1_doc_00_125364462#6_229054655 0

    Args:
        qrels_file: Path to qrels file

    Returns:
        Dictionary mapping qid to dict of docid -> relevance score
    """
    qrels = {}

    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            qid = parts[0]
            docid = parts[2]
            relevance = int(parts[3])

            if qid not in qrels:
                qrels[qid] = {}

            qrels[qid][docid] = relevance

    return qrels
