#!/bin/bash
################################################################################
# Simple wrapper to run TREC RAG 2024 download script
# Usage: ./scripts/download_trec_rag_2024.sh [OPTIONS]
################################################################################

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Run the Python script with all arguments passed through
python3 "$PROJECT_ROOT/c0_corpus_dataset_preparation/download_trec_rag_2024.py" "$@"
