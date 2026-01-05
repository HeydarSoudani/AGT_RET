#!/usr/bin/env python3
"""
Script for corpus analysis operations including subsampling and document counting.
"""

import os
import sys
import json
import argparse
import logging
import random
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: Path):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "corpus_analysis.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def corpus_subsampling(
    input_path: str,
    output_path: str,
    sample_size: Optional[int] = None,
    sample_ratio: Optional[float] = None,
    seed: int = 42
):
    """
    Subsample documents from a corpus JSONL file.

    Args:
        input_path: Path to input corpus JSONL file
        output_path: Path to output subsampled JSONL file
        sample_size: Number of documents to sample (absolute number)
        sample_ratio: Ratio of documents to sample (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Number of documents sampled
    """
    logger = logging.getLogger(__name__)

    input_path = Path(input_path)
    output_path = Path(output_path)

    # Validate input file exists
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Corpus Subsampling")
    logger.info("=" * 80)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Random seed: {seed}")
    logger.info("")

    # Set random seed
    random.seed(seed)

    # First pass: count total documents
    logger.info("Counting total documents in corpus...")
    total_docs = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                total_docs += 1

    logger.info(f"Total documents in corpus: {total_docs:,}")
    logger.info("")

    # Determine sample size
    if sample_size is not None:
        target_sample_size = min(sample_size, total_docs)
        logger.info(f"Sampling by absolute size: {target_sample_size:,} documents")
    elif sample_ratio is not None:
        if not 0.0 <= sample_ratio <= 1.0:
            logger.error("Sample ratio must be between 0.0 and 1.0")
            raise ValueError("Sample ratio must be between 0.0 and 1.0")
        target_sample_size = int(total_docs * sample_ratio)
        logger.info(f"Sampling by ratio: {sample_ratio:.2%} = {target_sample_size:,} documents")
    else:
        logger.error("Either sample_size or sample_ratio must be specified")
        raise ValueError("Either sample_size or sample_ratio must be specified")

    logger.info("")

    # Generate random indices to sample
    logger.info("Generating random sample indices...")
    sample_indices = set(random.sample(range(total_docs), target_sample_size))
    logger.info(f"Generated {len(sample_indices):,} sample indices")
    logger.info("")

    # Second pass: write sampled documents
    logger.info("Writing sampled documents...")
    sampled_count = 0
    current_idx = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            if current_idx in sample_indices:
                f_out.write(line + '\n')
                sampled_count += 1

                # Progress update
                if sampled_count % 10000 == 0:
                    logger.info(f"  Sampled {sampled_count:,} / {target_sample_size:,} documents...")

            current_idx += 1

    logger.info(f"✓ Successfully sampled {sampled_count:,} documents")
    logger.info(f"✓ Output saved to: {output_path}")
    logger.info("")

    return sampled_count


def count_corpus_docs(input_path: str) -> int:
    """
    Count the number of documents in a corpus JSONL file.

    Args:
        input_path: Path to corpus JSONL file

    Returns:
        Number of documents in the corpus
    """
    logger = logging.getLogger(__name__)

    input_path = Path(input_path)

    # Validate input file exists
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("=" * 80)
    logger.info("Counting Corpus Documents")
    logger.info("=" * 80)
    logger.info(f"Input: {input_path}")
    logger.info("")

    logger.info("Counting documents...")
    doc_count = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc_count += 1

            # Progress update for large files
            if doc_count % 1000000 == 0:
                logger.info(f"  Counted {doc_count:,} documents so far...")

    logger.info("")
    logger.info(f"✓ Total documents: {doc_count:,}")
    logger.info("")

    return doc_count


def main():
    parser = argparse.ArgumentParser(
        description="Corpus analysis operations: subsampling and document counting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Count documents in corpus
  python corpus_analysis.py count --input /path/to/corpus.jsonl

  # Subsample by absolute number
  python corpus_analysis.py subsample --input /path/to/corpus.jsonl --output /path/to/output.jsonl --sample-size 10000

  # Subsample by ratio
  python corpus_analysis.py subsample --input /path/to/corpus.jsonl --output /path/to/output.jsonl --sample-ratio 0.1

  # Use default paths
  python corpus_analysis.py subsample --sample-size 10000
  python corpus_analysis.py count
        """
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True

    # Subsampling command
    subsample_parser = subparsers.add_parser('subsample', help='Subsample documents from corpus')
    subsample_parser.add_argument(
        '--input',
        type=str,
        default='/projects/0/prjs0834/heydars/AGT_RET/src/msmarco_v2.1_doc_segmented.jsonl',
        help='Input corpus JSONL file (default: /projects/0/prjs0834/heydars/AGT_RET/src/msmarco_v2.1_doc_segmented.jsonl)'
    )
    subsample_parser.add_argument(
        '--output',
        type=str,
        default='corpus_datasets/trec_rag_2024/processed/corpus_subsample.jsonl',
        help='Output subsampled JSONL file (default: corpus_datasets/trec_rag_2024/processed/corpus_subsample.jsonl)'
    )

    # Sample size options (mutually exclusive)
    sample_group = subsample_parser.add_mutually_exclusive_group(required=True)
    sample_group.add_argument(
        '--sample-size',
        type=int,
        help='Number of documents to sample (absolute)'
    )
    sample_group.add_argument(
        '--sample-ratio',
        type=float,
        help='Ratio of documents to sample (0.0 to 1.0)'
    )

    subsample_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    # Count command
    count_parser = subparsers.add_parser('count', help='Count documents in corpus')
    count_parser.add_argument(
        '--input',
        type=str,
        default='/projects/0/prjs0834/heydars/AGT_RET/src/msmarco_v2.1_doc_segmented.jsonl',
        help='Input corpus JSONL file (default: /projects/0/prjs0834/heydars/AGT_RET/src/msmarco_v2.1_doc_segmented.jsonl)'
    )

    # Logging configuration
    parser.add_argument(
        '--log-dir',
        type=str,
        default='script_logging',
        help='Directory for log files (default: script_logging)'
    )

    args = parser.parse_args()

    # Setup logging
    log_dir = Path(args.log_dir)
    setup_logging(log_dir)
    logger = logging.getLogger(__name__)

    # Execute command
    try:
        if args.command == 'subsample':
            corpus_subsampling(
                input_path=args.input,
                output_path=args.output,
                sample_size=args.sample_size,
                sample_ratio=args.sample_ratio,
                seed=args.seed
            )
        elif args.command == 'count':
            count_corpus_docs(input_path=args.input)

        logger.info("=" * 80)
        logger.info("Operation completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
