#!/usr/bin/env python3
"""
Script to download TREC RAG 2024 data (corpus, topics, and qrels).
"""

import os
import sys
import json
import argparse
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List


def setup_logging(log_dir: Path):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "download_trec_rag_2024.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def download_corpus(base_dir: Path):
    """
    Download TREC RAG 2024 corpus.

    The TREC RAG 2024 track uses the MS MARCO V2.1 document corpus.
    This corpus contains web documents from the Common Crawl.

    Args:
        base_dir: Base directory for TREC RAG 2024 (will use base_dir/src)
    """
    logger = logging.getLogger(__name__)

    # Use src directory at base level
    src_dir = base_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("TREC RAG 2024 Corpus Download")
    logger.info("=" * 80)

    # MS MARCO V2.1 corpus download URL
    corpus_url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2.1_doc_segmented.tar"
    corpus_tar_path = src_dir / "msmarco_v2.1_doc_segmented.tar"

    logger.info("MS MARCO V2.1 Segmented Document Corpus")
    logger.info("TREC RAG 2024 official corpus")
    logger.info("")
    logger.info(f"URL: {corpus_url}")
    logger.info(f"Size: ~30 GB compressed")
    logger.info("")

    logger.info("Attempting to download MS MARCO V2.1 corpus...")
    logger.info("WARNING: This is a large file (~30 GB). Download may take a long time.")
    logger.info("")

    try:
        # Download the corpus tar file
        logger.info("Downloading corpus file...")
        logger.info("This will take a while due to the file size...")

        # Use urllib with progress indication
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                downloaded_gb = downloaded / (1024**3)
                total_gb = total_size / (1024**3)
                if block_num % 100 == 0:  # Update every 100 blocks to avoid spam
                    print(f"\r  Progress: {percent:.1f}% ({downloaded_gb:.2f} GB / {total_gb:.2f} GB)", end='', flush=True)

        urllib.request.urlretrieve(corpus_url, corpus_tar_path, reporthook=download_progress)
        logger.info(f"✓ Downloaded to: {corpus_tar_path}")
        logger.info("")

        # Extract the tar file
        logger.info("Extracting corpus archive...")
        logger.info("This will also take a while...")
        import subprocess
        result = subprocess.run(
            ["tar", "-xvf", str(corpus_tar_path), "-C", str(src_dir)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.info(f"✓ Extracted to: {src_dir}/")
            logger.info("")

            # Get corpus statistics
            logger.info("Corpus extracted successfully!")
            logger.info("")
        else:
            logger.error(f"Failed to extract corpus: {result.stderr}")
            logger.info("")

        logger.info("✓ Successfully downloaded and extracted TREC RAG 2024 corpus!")
        logger.info("")

    except Exception as e:
        logger.error(f"Failed to download corpus: {e}")
        logger.info("")
        logger.info("Alternative Options:")
        logger.info("")

        logger.info("Option 1: Using ir_datasets (Recommended)")
        logger.info("-" * 80)
        logger.info("Install: pip install ir-datasets")
        logger.info("")
        logger.info("Usage in Python:")
        logger.info("  import ir_datasets")
        logger.info("  dataset = ir_datasets.load('msmarco-v2.1-doc-segmented')")
        logger.info("  for doc in dataset.docs_iter():")
        logger.info("      # doc.doc_id, doc.url, doc.title, doc.headings, doc.body")
        logger.info("")

        logger.info("Option 2: Manual Download with wget")
        logger.info("-" * 80)
        logger.info(f"wget -P {src_dir} {corpus_url}")
        logger.info(f"tar -xvf {corpus_tar_path} -C {src_dir}")
        logger.info("")

        logger.info("Option 3: Using PyTerrier")
        logger.info("-" * 80)
        logger.info("Install: pip install python-terrier")
        logger.info("")
        logger.info("Usage in Python:")
        logger.info("  import pyterrier as pt")
        logger.info("  pt.init()")
        logger.info("  dataset = pt.get_dataset('irds:msmarco-v2.1-doc-segmented')")
        logger.info("")



def download_topics(base_dir: Path):
    """
    Download TREC RAG 2024 topics and convert to JSONL format.

    Args:
        base_dir: Base directory for TREC RAG 2024 (will use base_dir/src and base_dir/processed)
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("TREC RAG 2024 Topics Download")
    logger.info("=" * 80)

    # Use src and processed directories at base level
    src_dir = base_dir / "src"
    processed_dir = base_dir / "processed"
    src_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Official TREC RAG 2024 topics URL
    topics_url = "https://trec.nist.gov/data/rag/topics.rag24.test.txt"
    topics_txt_path = src_dir / "topics.rag24.test.txt"
    topics_jsonl_path = processed_dir / "topics.rag24.test.jsonl"

    logger.info("Attempting to download TREC RAG 2024 topics...")
    logger.info(f"URL: {topics_url}")
    logger.info("")

    try:
        # Download the TSV file
        logger.info("Downloading topics file...")
        urllib.request.urlretrieve(topics_url, topics_txt_path)
        logger.info(f"✓ Downloaded to: {topics_txt_path}")
        logger.info("")

        # Convert TSV to JSONL
        logger.info("Converting TSV to JSONL format...")
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
        logger.info(f"✓ Saved to: {topics_jsonl_path}")
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

        logger.info("✓ Successfully downloaded and converted TREC RAG 2024 topics!")
        logger.info("")

    except Exception as e:
        logger.error(f"Failed to download topics: {e}")

def download_qrels(base_dir: Path):
    """
    Download TREC RAG 2024 qrels (relevance judgments).
    Downloads both umbrella and retrieval-conditions qrels files.

    Args:
        base_dir: Base directory for TREC RAG 2024 (will use base_dir/src)
    """
    logger = logging.getLogger(__name__)

    # Use src directory at base level
    src_dir = base_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("TREC RAG 2024 Qrels Download")
    logger.info("=" * 80)

    # Define both qrels files to download
    qrels_files = [
        {
            "name": "Umbrella Qrels (All)",
            "url": "https://trec-rag.github.io/assets/txt/qrels.rag24.test-umbrela-all.txt",
            "filename": "qrels.rag24.test-umbrela-all.txt",
            "description": "Combined qrels from all retrieval conditions"
        },
        {
            "name": "Retrieval Conditions Qrels",
            "url": "https://trec.nist.gov/data/rag/2024-retrieval-conditions-qrels.txt",
            "filename": "2024-retrieval-conditions-qrels.txt",
            "description": "Condition-specific qrels"
        }
    ]

    for qrels_info in qrels_files:
        logger.info("-" * 80)
        logger.info(f"Downloading: {qrels_info['name']}")
        logger.info(f"Description: {qrels_info['description']}")
        logger.info(f"URL: {qrels_info['url']}")
        logger.info("")

        qrels_path = src_dir / qrels_info['filename']

        try:
            # Download the qrels file
            logger.info("Downloading qrels file...")
            urllib.request.urlretrieve(qrels_info['url'], qrels_path)
            logger.info(f"✓ Downloaded to: {qrels_path}")
            logger.info("")

            # Read and display statistics
            logger.info("Analyzing qrels file...")
            qrels_count = 0
            queries_set = set()

            with open(qrels_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        queries_set.add(parts[0])
                        qrels_count += 1

            logger.info(f"✓ Qrels statistics:")
            logger.info(f"  - Total judgments: {qrels_count}")
            logger.info(f"  - Unique queries: {len(queries_set)}")
            logger.info("")

            # Show sample qrels
            logger.info("Sample qrels (first 5 lines):")
            with open(qrels_path, 'r', encoding='utf-8') as f:
                count = 0
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    logger.info(f"  {line}")
                    count += 1
                    if count >= 5:
                        break
            logger.info("")

            logger.info(f"✓ Successfully downloaded {qrels_info['name']}!")
            logger.info("")

        except Exception as e:
            logger.error(f"Failed to download {qrels_info['name']}: {e}")
            logger.info("")



def main():
    parser = argparse.ArgumentParser(
        description="Download TREC RAG 2024 data (corpus, topics, and qrels)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all data with default paths
  python download_trec_rag_2024.py --all

  # Download only corpus
  python download_trec_rag_2024.py --corpus

  # Download only topics
  python download_trec_rag_2024.py --topics

  # Download only qrels
  python download_trec_rag_2024.py --qrels

  # Download all with custom base directory
  python download_trec_rag_2024.py --all --base-dir /custom/path

  # Download specific components with custom paths
  python download_trec_rag_2024.py --corpus --corpus-dir /custom/corpus/path
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

    # Directory configuration
    parser.add_argument(
        "--base-dir",
        type=str,
        # default="corpus_datasets/trec_rag_2024",
        default="/projects/0/prjs0834/heydars/AGT_RET",
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
    if not (args.all or args.corpus or args.topics or args.qrels):
        args.all = True

    # Setup logging
    log_dir = Path(args.log_dir)
    setup_logging(log_dir)
    logger = logging.getLogger(__name__)

    # Print header with component info
    logger.info("=" * 80)
    logger.info("TREC RAG 2024 Data Download Script")
    logger.info("=" * 80)

    components = []
    if args.all:
        components = ["Corpus", "Topics", "Qrels"]
    else:
        if args.corpus:
            components.append("Corpus")
        if args.topics:
            components.append("Topics")
        if args.qrels:
            components.append("Qrels")

    logger.info(f"Downloading: {', '.join(components)}")
    logger.info("")

    # All components use the same base directory with src/ and processed/ subdirs
    base_dir = Path(args.base_dir)

    # Download components
    if args.all or args.corpus:
        download_corpus(base_dir)
        logger.info("✓ Corpus download instructions generated")
        logger.info(f"  Location: {base_dir}/src")
        logger.info("")

    if args.all or args.topics:
        download_topics(base_dir)
        logger.info("✓ Topics downloaded and processed")
        logger.info(f"  Source: {base_dir}/src")
        logger.info(f"  Processed: {base_dir}/processed")
        logger.info("")

    if args.all or args.qrels:
        download_qrels(base_dir)
        logger.info("✓ Qrels download instructions generated")
        logger.info(f"  Location: {base_dir}/src")
        logger.info("")

if __name__ == "__main__":
    main()

# python c0_corpus_dataset_preparation/download_trec_rag_2024.py --corpus