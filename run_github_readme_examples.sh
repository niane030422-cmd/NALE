#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-data/MRAG_dataset.json}"
MULTI_DATASET="${MULTI_DATASET:-data/MRAG_mul.json}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-7B-Instruct}"
CONTRIEVER_PATH="${CONTRIEVER_PATH:-bm25}"
QUERY_LANG="${QUERY_LANG:-en}"
DOC_LANG="${DOC_LANG:-zh}"
DESCRIPTION="${DESCRIPTION:-lang_specific}"
MODE="${MODE:-golden}"
NOISE_RATE="${NOISE_RATE:-0.0}"
CHUNK_NUM="${CHUNK_NUM:-5}"
SEED="${SEED:-233}"

mkdir -p processed_data

python3 process_ete_document.py \
  --input "${MULTI_DATASET}" \
  --output "processed_data/ete_documents.json"

python3 eval/eval_mono.py \
  --modelname "${MODEL_NAME}" \
  --contriever_path "${CONTRIEVER_PATH}" \
  --dataset "${DATASET}" \
  --language "${QUERY_LANG}" \
  --description "${DESCRIPTION}" \
  --mode "${MODE}" \
  --noise_rate "${NOISE_RATE}" \
  --chunk_num "${CHUNK_NUM}" \
  --seed "${SEED}" \
  --retrieval_setting given_gt

python3 eval/eval_mono.py \
  --modelname "${MODEL_NAME}" \
  --contriever_path "${CONTRIEVER_PATH}" \
  --dataset "${DATASET}" \
  --language "${QUERY_LANG}" \
  --description "${DESCRIPTION}" \
  --mode "${MODE}" \
  --noise_rate "${NOISE_RATE}" \
  --chunk_num "${CHUNK_NUM}" \
  --seed "${SEED}" \
  --retrieval_setting end2end

python3 eval/eval_cross.py \
  --modelname "${MODEL_NAME}" \
  --contriever_path "${CONTRIEVER_PATH}" \
  --dataset "${DATASET}" \
  --language "${QUERY_LANG}" \
  --doc_lang "${DOC_LANG}" \
  --description "${DESCRIPTION}" \
  --mode "${MODE}" \
  --noise_rate "${NOISE_RATE}" \
  --chunk_num "${CHUNK_NUM}" \
  --seed "${SEED}" \
  --retrieval_setting given_gt \
  --answer_match_mode strict

python3 eval/eval_cross.py \
  --modelname "${MODEL_NAME}" \
  --contriever_path "${CONTRIEVER_PATH}" \
  --dataset "${DATASET}" \
  --language "${QUERY_LANG}" \
  --doc_lang "${DOC_LANG}" \
  --description "${DESCRIPTION}" \
  --mode "${MODE}" \
  --noise_rate "${NOISE_RATE}" \
  --chunk_num "${CHUNK_NUM}" \
  --seed "${SEED}" \
  --retrieval_setting given_gt \
  --answer_match_mode loosen

python3 eval/eval_cross.py \
  --modelname "${MODEL_NAME}" \
  --contriever_path "${CONTRIEVER_PATH}" \
  --dataset "${DATASET}" \
  --language "${QUERY_LANG}" \
  --doc_lang "${DOC_LANG}" \
  --description "${DESCRIPTION}" \
  --mode "${MODE}" \
  --noise_rate "${NOISE_RATE}" \
  --chunk_num "${CHUNK_NUM}" \
  --seed "${SEED}" \
  --retrieval_setting end2end \
  --answer_match_mode loosen

python3 eval/eval_multi.py \
  --modelname "${MODEL_NAME}" \
  --contriever_path "${CONTRIEVER_PATH}" \
  --dataset "${MULTI_DATASET}" \
  --language "${QUERY_LANG}" \
  --description "${DESCRIPTION}" \
  --mode "${MODE}" \
  --noise_rate "${NOISE_RATE}" \
  --chunk_num "${CHUNK_NUM}" \
  --seed "${SEED}" \
  --retrieval_setting given_gt

python3 eval/eval_multi.py \
  --modelname "${MODEL_NAME}" \
  --contriever_path "${CONTRIEVER_PATH}" \
  --dataset "${MULTI_DATASET}" \
  --language "${QUERY_LANG}" \
  --description "${DESCRIPTION}" \
  --mode "${MODE}" \
  --noise_rate "${NOISE_RATE}" \
  --chunk_num "${CHUNK_NUM}" \
  --seed "${SEED}" \
  --retrieval_setting end2end

python3 eval/eval_contriver.py \
  --dataset "${DATASET}" \
  --language "${QUERY_LANG}" \
  --doc_lang "${DOC_LANG}" \
  --contriever_path "${CONTRIEVER_PATH}" \
  --chunk_size 200 \
  --max_k 20 \
  --seed "${SEED}" \
  --save_path "results/retriever_recall_${QUERY_LANG}_${DOC_LANG}.jsonl"

python3 eval/eval_contriver_multi.py \
  --dataset "${MULTI_DATASET}" \
  --contriever_path "${CONTRIEVER_PATH}" \
  --chunk_size 200 \
  --top_k 15 \
  --seed "${SEED}" \
  --save_dir "contriver_evalresults/multilang_analysis"
