#!/usr/bin/env python3
"""
Utility functions for loading TREC RAG 2024 data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Document:
    """Document from TREC RAG 2024 corpus."""
    doc_id: str
    url: str
    title: str
    headings: str
    body: str

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "title": self.title,
            "headings": self.headings,
            "body": self.body
        }


@dataclass
class Query:
    """Query/Topic from TREC RAG 2024."""
    query_id: str
    query_text: str
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "metadata": self.metadata or {}
        }


@dataclass
class Qrel:
    """Relevance judgment from TREC RAG 2024."""
    query_id: str
    doc_id: str
    relevance: int
    iteration: str = "0"

    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "doc_id": self.doc_id,
            "relevance": self.relevance,
            "iteration": self.iteration
        }


class TRECRag2024Loader:
    """Loader for TREC RAG 2024 data using ir_datasets."""

    def __init__(self, use_ir_datasets: bool = True):
        """
        Initialize the loader.

        Args:
            use_ir_datasets: Whether to use ir_datasets library (recommended)
        """
        self.use_ir_datasets = use_ir_datasets
        self.logger = logging.getLogger(__name__)

        if use_ir_datasets:
            try:
                import ir_datasets
                self.ir_datasets = ir_datasets
                self.logger.info("ir_datasets library loaded successfully")
            except ImportError:
                self.logger.warning(
                    "ir_datasets not found. Install with: pip install ir-datasets"
                )
                self.use_ir_datasets = False

    def load_corpus(self, dataset_id: str = 'msmarco-v2.1-doc-segmented') -> Iterator[Document]:
        """
        Load corpus documents.

        Args:
            dataset_id: Dataset identifier for ir_datasets

        Yields:
            Document objects
        """
        if not self.use_ir_datasets:
            raise RuntimeError("ir_datasets is required. Install with: pip install ir-datasets")

        self.logger.info(f"Loading corpus: {dataset_id}")
        dataset = self.ir_datasets.load(dataset_id)

        for doc in dataset.docs_iter():
            yield Document(
                doc_id=doc.doc_id,
                url=doc.url,
                title=doc.title,
                headings=doc.headings,
                body=doc.body
            )

    def load_queries(self, dataset_id: str = 'msmarco-v2.1-doc-segmented/trec-rag-2024') -> Iterator[Query]:
        """
        Load queries/topics.

        Args:
            dataset_id: Dataset identifier for ir_datasets

        Yields:
            Query objects
        """
        if not self.use_ir_datasets:
            raise RuntimeError("ir_datasets is required. Install with: pip install ir-datasets")

        self.logger.info(f"Loading queries: {dataset_id}")
        dataset = self.ir_datasets.load(dataset_id)

        for query in dataset.queries_iter():
            yield Query(
                query_id=query.query_id,
                query_text=query.text
            )

    def load_qrels(self, dataset_id: str = 'msmarco-v2.1-doc-segmented/trec-rag-2024') -> Iterator[Qrel]:
        """
        Load qrels (relevance judgments).

        Args:
            dataset_id: Dataset identifier for ir_datasets

        Yields:
            Qrel objects
        """
        if not self.use_ir_datasets:
            raise RuntimeError("ir_datasets is required. Install with: pip install ir-datasets")

        self.logger.info(f"Loading qrels: {dataset_id}")
        dataset = self.ir_datasets.load(dataset_id)

        for qrel in dataset.qrels_iter():
            yield Qrel(
                query_id=qrel.query_id,
                doc_id=qrel.doc_id,
                relevance=qrel.relevance,
                iteration=str(qrel.iteration)
            )

    def load_queries_from_file(self, filepath: Path) -> List[Query]:
        """
        Load queries from a JSON file.

        Args:
            filepath: Path to queries JSON file

        Returns:
            List of Query objects
        """
        self.logger.info(f"Loading queries from file: {filepath}")
        queries = []

        with open(filepath, 'r') as f:
            data = json.load(f)

        for item in data:
            queries.append(Query(
                query_id=item.get('query_id', item.get('id', '')),
                query_text=item.get('query', item.get('text', '')),
                metadata=item.get('metadata')
            ))

        self.logger.info(f"Loaded {len(queries)} queries")
        return queries

    def load_qrels_from_file(self, filepath: Path) -> List[Qrel]:
        """
        Load qrels from a TREC format file.

        Args:
            filepath: Path to qrels file

        Returns:
            List of Qrel objects
        """
        self.logger.info(f"Loading qrels from file: {filepath}")
        qrels = []

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) >= 4:
                    qrels.append(Qrel(
                        query_id=parts[0],
                        iteration=parts[1],
                        doc_id=parts[2],
                        relevance=int(parts[3])
                    ))

        self.logger.info(f"Loaded {len(qrels)} qrels")
        return qrels

    def save_queries_to_file(self, queries: List[Query], filepath: Path):
        """
        Save queries to a JSON file.

        Args:
            queries: List of Query objects
            filepath: Output file path
        """
        self.logger.info(f"Saving {len(queries)} queries to: {filepath}")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = [q.to_dict() for q in queries]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info("Queries saved successfully")

    def save_qrels_to_file(self, qrels: List[Qrel], filepath: Path):
        """
        Save qrels to a TREC format file.

        Args:
            qrels: List of Qrel objects
            filepath: Output file path
        """
        self.logger.info(f"Saving {len(qrels)} qrels to: {filepath}")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            for qrel in qrels:
                f.write(f"{qrel.query_id} {qrel.iteration} {qrel.doc_id} {qrel.relevance}\n")

        self.logger.info("Qrels saved successfully")

    def get_qrels_dict(self, qrels: List[Qrel]) -> Dict[str, Dict[str, int]]:
        """
        Convert qrels list to nested dictionary format for evaluation.

        Args:
            qrels: List of Qrel objects

        Returns:
            Dictionary: {query_id: {doc_id: relevance}}
        """
        qrels_dict = {}
        for qrel in qrels:
            if qrel.query_id not in qrels_dict:
                qrels_dict[qrel.query_id] = {}
            qrels_dict[qrel.query_id][qrel.doc_id] = qrel.relevance

        return qrels_dict


def example_usage():
    """Example usage of the TREC RAG 2024 loader."""
    logging.basicConfig(level=logging.INFO)

    # Initialize loader
    loader = TRECRag2024Loader(use_ir_datasets=True)

    # Load queries
    print("\n=== Loading Queries ===")
    queries = []
    for i, query in enumerate(loader.load_queries()):
        queries.append(query)
        print(f"Query {query.query_id}: {query.query_text}")
        if i >= 4:  # Show first 5 queries
            break

    # Load qrels
    print("\n=== Loading Qrels ===")
    qrels = []
    for i, qrel in enumerate(loader.load_qrels()):
        qrels.append(qrel)
        print(f"Query: {qrel.query_id}, Doc: {qrel.doc_id}, Relevance: {qrel.relevance}")
        if i >= 9:  # Show first 10 qrels
            break

    # Load corpus (be careful - this is large!)
    print("\n=== Loading Corpus (first 3 documents) ===")
    for i, doc in enumerate(loader.load_corpus()):
        print(f"Doc ID: {doc.doc_id}")
        print(f"Title: {doc.title}")
        print(f"Body preview: {doc.body[:100]}...")
        print()
        if i >= 2:  # Show first 3 documents
            break

    # Get qrels as dictionary
    print("\n=== Qrels Dictionary Format ===")
    qrels_dict = loader.get_qrels_dict(qrels)
    for query_id in list(qrels_dict.keys())[:2]:  # Show first 2 queries
        print(f"Query {query_id}: {len(qrels_dict[query_id])} relevant docs")


if __name__ == "__main__":
    example_usage()
