#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --output=script_logging/retrieval_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

### === Set variables ==========================
query_file=corpus_datasets/trec_rag_2024/processed/topics.rag24.test.jsonl
corpus_file=corpus_datasets/enwiki_20251001.jsonl
qrels_file=corpus_datasets/trec_rag_2024/src/2024-retrieval-conditions-qrels.txt
index_dir=/projects/0/prjs0834/heydars/INDICES
output_file=run_outputs/retrieval_results_bm25.jsonl

retriever_name=bm25  # Options: bm25, contriever, dpr, e5, bge, rerank_l6, rerank_l12
retrieval_topk=100

### === Run Retrieval ==========================
python $HOME/AGT_RET/c1_baseline_retrievers/run_retrieval.py \
    --query_file $query_file \
    --corpus_path $corpus_file \
    --qrels_file $qrels_file \
    --output_file $output_file \
    --retriever_name $retriever_name \
    --index_dir $index_dir \
    --retrieval_topk $retrieval_topk \
    --device 0 \
    --seed 42

echo "Retrieval completed! Results saved to $output_file"
