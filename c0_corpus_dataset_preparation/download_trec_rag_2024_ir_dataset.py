#!/usr/bin/env python3
"""
Script to download TREC RAG 2024 data using ir_datasets library.
This provides an alternative method to download corpus, topics, and qrels.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional


def setup_logging(log_dir: Path):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "download_trec_rag_2024_ir_dataset.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def download_with_ir_datasets(base_dir: Path, download_corpus: bool = True,
                               download_topics: bool = True,
                               download_qrels: bool = True):
    """
    Download TREC RAG 2024 data using ir_datasets library.

    Args:
        base_dir: Base directory for TREC RAG 2024 (will use base_dir/src)
        download_corpus: Whether to download corpus
        download_topics: Whether to download topics
        download_qrels: Whether to download qrels
    """
    logger = logging.getLogger(__name__)

    try:
        import ir_datasets
    except ImportError:
        logger.error("ir_datasets is not installed!")
        logger.info("Please install it using: pip install ir-datasets")
        return

    # Use src directory at base level
    src_dir = base_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("TREC RAG 2024 Data Download using ir_datasets")
    logger.info("=" * 80)
    logger.info("")

    # Load the dataset
    dataset_id = 'msmarco-v2.1-doc-segmented/trec-rag-2024'
    logger.info(f"Loading dataset: {dataset_id}")
    logger.info("")

    try:
        dataset = ir_datasets.load(dataset_id)
        logger.info(f"✓ Dataset loaded successfully")
        logger.info("")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("")
        logger.info("Trying base dataset: msmarco-v2.1-doc-segmented")
        try:
            dataset = ir_datasets.load('msmarco-v2.1-doc-segmented')
            logger.info(f"✓ Base dataset loaded successfully")
            logger.info("")
        except Exception as e2:
            logger.error(f"Failed to load base dataset: {e2}")
            return

    # Download Topics
    if download_topics:
        logger.info("-" * 80)
        logger.info("Downloading Topics")
        logger.info("-" * 80)

        topics_txt_path = src_dir / "topics.rag24.test.ir_datasets.txt"

        try:
            logger.info("Fetching topics from ir_datasets...")
            topics_count = 0

            with open(topics_txt_path, 'w', encoding='utf-8') as f:
                for query in dataset.queries_iter():
                    # Write in TSV format: query_id\tquery_text
                    f.write(f"{query.query_id}\t{query.text}\n")
                    topics_count += 1

            logger.info(f"✓ Downloaded {topics_count} topics")
            logger.info(f"✓ Saved to: {topics_txt_path}")
            logger.info("")

            # Show sample topics
            logger.info("Sample topics (first 3):")
            with open(topics_txt_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        logger.info(f"  {parts[0]}: {parts[1][:80]}...")
            logger.info("")

        except Exception as e:
            logger.error(f"Failed to download topics: {e}")
            logger.info("")

    # Download Qrels
    if download_qrels:
        logger.info("-" * 80)
        logger.info("Downloading Qrels")
        logger.info("-" * 80)

        qrels_path = src_dir / "qrels.rag24.test.ir_datasets.txt"

        try:
            logger.info("Fetching qrels from ir_datasets...")
            qrels_count = 0
            queries_set = set()

            with open(qrels_path, 'w', encoding='utf-8') as f:
                for qrel in dataset.qrels_iter():
                    # Write in TREC format: query_id iteration doc_id relevance
                    f.write(f"{qrel.query_id} {qrel.iteration} {qrel.doc_id} {qrel.relevance}\n")
                    qrels_count += 1
                    queries_set.add(qrel.query_id)

            logger.info(f"✓ Downloaded qrels")
            logger.info(f"  - Total judgments: {qrels_count}")
            logger.info(f"  - Unique queries: {len(queries_set)}")
            logger.info(f"✓ Saved to: {qrels_path}")
            logger.info("")

            # Show sample qrels
            logger.info("Sample qrels (first 5 lines):")
            with open(qrels_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    logger.info(f"  {line.strip()}")
            logger.info("")

        except Exception as e:
            logger.error(f"Failed to download qrels: {e}")
            logger.info("")

    # Download Corpus
    if download_corpus:
        logger.info("-" * 80)
        logger.info("Downloading Corpus")
        logger.info("-" * 80)

        corpus_jsonl_path = src_dir / "corpus.msmarco_v2.1_doc_segmented.ir_datasets.jsonl"

        try:
            logger.info("Fetching corpus from ir_datasets...")
            logger.info("WARNING: This will take a very long time and create a large file!")
            logger.info("")

            docs_count = 0
            batch_size = 10000

            with open(corpus_jsonl_path, 'w', encoding='utf-8') as f:
                for doc in dataset.docs_iter():
                    # Create JSON object with document fields
                    doc_obj = {
                        "doc_id": doc.doc_id,
                        "url": doc.url if hasattr(doc, 'url') else "",
                        "title": doc.title if hasattr(doc, 'title') else "",
                        "headings": doc.headings if hasattr(doc, 'headings') else "",
                        "body": doc.body if hasattr(doc, 'body') else ""
                    }

                    # Write as JSONL
                    f.write(json.dumps(doc_obj) + '\n')
                    docs_count += 1

                    # Log progress
                    if docs_count % batch_size == 0:
                        docs_millions = docs_count / 1_000_000
                        print(f"\r  Processed: {docs_millions:.2f}M documents", end='', flush=True)

            print()  # New line after progress
            logger.info(f"✓ Downloaded {docs_count:,} documents")
            logger.info(f"✓ Saved to: {corpus_jsonl_path}")
            logger.info("")

            # Show sample documents
            logger.info("Sample documents (first 2):")
            with open(corpus_jsonl_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 2:
                        break
                    doc = json.loads(line)
                    logger.info(f"  Doc ID: {doc['doc_id']}")
                    logger.info(f"  Title: {doc['title'][:80]}...")
                    logger.info(f"  Body: {doc['body'][:100]}...")
                    logger.info("")

        except Exception as e:
            logger.error(f"Failed to download corpus: {e}")
            logger.info("")

    logger.info("=" * 80)
    logger.info("Download Complete!")
    logger.info("=" * 80)


def convert_topics_txt_to_jsonl(topics_txt_path: Path, topics_jsonl_path: Path):
    """
    Convert topics from TSV format to JSONL format.

    Args:
        topics_txt_path: Path to input TSV file (query_id\tquery_text)
        topics_jsonl_path: Path to output JSONL file
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Converting Topics TSV to JSONL")
    logger.info("=" * 80)
    logger.info("")

    if not topics_txt_path.exists():
        logger.error(f"Topics file not found: {topics_txt_path}")
        return

    try:
        topics_count = 0

        with open(topics_txt_path, 'r', encoding='utf-8') as f_in, \
             open(topics_jsonl_path, 'w', encoding='utf-8') as f_out:

            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                # Parse TSV format: topic_id\tquestion
                parts = line.split('\t')
                if len(parts) >= 2:
                    topic_id = parts[0]
                    question = parts[1]

                    # Create JSON object
                    topic_obj = {
                        "qid": topic_id,
                        "question": question
                    }

                    # Write as JSONL
                    f_out.write(json.dumps(topic_obj) + '\n')
                    topics_count += 1

        logger.info(f"✓ Converted {topics_count} topics to JSONL format")
        logger.info(f"✓ Input:  {topics_txt_path}")
        logger.info(f"✓ Output: {topics_jsonl_path}")
        logger.info("")

        # Show sample topics
        logger.info("Sample topics (first 3):")
        with open(topics_jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                topic = json.loads(line)
                logger.info(f"  {topic['qid']}: {topic['question'][:80]}...")
        logger.info("")

        logger.info("✓ Conversion complete!")
        logger.info("")

    except Exception as e:
        logger.error(f"Failed to convert topics: {e}")
        logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description="Download TREC RAG 2024 data using ir_datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all data with default paths
  python download_trec_rag_2024_ir_dataset.py --all

  # Download only topics and qrels (skip corpus)
  python download_trec_rag_2024_ir_dataset.py --topics --qrels

  # Download topics and convert to JSONL
  python download_trec_rag_2024_ir_dataset.py --topics --convert-topics

  # Download all with custom base directory
  python download_trec_rag_2024_ir_dataset.py --all --base-dir /custom/path

  # Convert existing topics file to JSONL
  python download_trec_rag_2024_ir_dataset.py --convert-topics --topics-txt-file topics.txt --topics-jsonl-file topics.jsonl
        """
    )

    # Component selection
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all components (corpus, topics, and qrels)"
    )
    parser.add_argument(
        "--corpus",
        action="store_true",
        help="Download corpus only"
    )
    parser.add_argument(
        "--topics",
        action="store_true",
        help="Download topics only"
    )
    parser.add_argument(
        "--qrels",
        action="store_true",
        help="Download qrels only"
    )

    # Conversion options
    parser.add_argument(
        "--convert-topics",
        action="store_true",
        help="Convert topics TSV to JSONL format"
    )
    parser.add_argument(
        "--topics-txt-file",
        type=str,
        help="Path to topics TSV file for conversion"
    )
    parser.add_argument(
        "--topics-jsonl-file",
        type=str,
        help="Path to output topics JSONL file"
    )

    # Directory configuration
    parser.add_argument(
        "--base-dir",
        type=str,
        default="corpus_datasets/trec_rag_2024",
        help="Base directory for all data (default: corpus_datasets/trec_rag_2024)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="script_logging",
        help="Directory for log files (default: script_logging)"
    )

    args = parser.parse_args()

    # If no component specified, default to all
    if not (args.all or args.corpus or args.topics or args.qrels or args.convert_topics):
        args.all = True

    # Setup logging
    log_dir = Path(args.log_dir)
    setup_logging(log_dir)
    logger = logging.getLogger(__name__)

    # Print header with component info
    logger.info("=" * 80)
    logger.info("TREC RAG 2024 Data Download Script (ir_datasets)")
    logger.info("=" * 80)
    logger.info("")

    base_dir = Path(args.base_dir)
    src_dir = base_dir / "src"
    processed_dir = base_dir / "processed"

    # Download components using ir_datasets
    if args.all or args.corpus or args.topics or args.qrels:
        download_corpus = args.all or args.corpus
        download_topics = args.all or args.topics
        download_qrels = args.all or args.qrels

        components = []
        if download_corpus:
            components.append("Corpus")
        if download_topics:
            components.append("Topics")
        if download_qrels:
            components.append("Qrels")

        logger.info(f"Downloading: {', '.join(components)}")
        logger.info("")

        download_with_ir_datasets(
            base_dir=base_dir,
            download_corpus=download_corpus,
            download_topics=download_topics,
            download_qrels=download_qrels
        )

    # Convert topics to JSONL
    if args.convert_topics:
        processed_dir.mkdir(parents=True, exist_ok=True)

        if args.topics_txt_file and args.topics_jsonl_file:
            topics_txt_path = Path(args.topics_txt_file)
            topics_jsonl_path = Path(args.topics_jsonl_file)
        else:
            # Use default paths
            topics_txt_path = src_dir / "topics.rag24.test.ir_datasets.txt"
            topics_jsonl_path = processed_dir / "topics.rag24.test.ir_datasets.jsonl"

        convert_topics_txt_to_jsonl(topics_txt_path, topics_jsonl_path)

    # Final summary
    logger.info("=" * 80)
    logger.info("Process Complete!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next Steps:")
    logger.info("1. Check the downloaded files in the src directory")
    logger.info("2. If topics were downloaded, consider converting to JSONL: --convert-topics")
    logger.info("3. Use the data for your retrieval experiments")
    logger.info("")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

# Example usage:
# python c0_corpus_dataset_preparation/download_trec_rag_2024_ir_dataset.py --topics --qrels
# python c0_corpus_dataset_preparation/download_trec_rag_2024_ir_dataset.py --all
# python c0_corpus_dataset_preparation/download_trec_rag_2024_ir_dataset.py --convert-topics
