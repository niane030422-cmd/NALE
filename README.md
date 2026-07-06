# Futurepedia

Futurepedia is a multilingual retrieval-augmented language model (RALM) benchmark built from parallel texts in eight representative languages.

This repository provides the dataset and evaluation code for three tasks: monolingual knowledge extraction, cross-lingual knowledge transfer, and multilingual knowledge selection. It includes scripts for evaluating retrievers and LLMs in both ground-truth context and end-to-end RAG settings.

## Repository Layout

```text
data/
  MRAG_dataset.json                 # Futurepedia monolingual/cross-lingual data
  MRAG_mul.json                     # Futurepedia multilingual selection data
  mul_ans_all_replaced.json         # multilingual data with replaced answers
  rag_eval/                         # translated RAG evaluation files
eval/
  eval_mono.py                      # monolingual RAG evaluation
  eval_cross.py                     # cross-lingual RAG evaluation
  eval_multi.py                     # multilingual RAG evaluation
  eval_contriver.py                 # cross-lingual retriever recall
  eval_contriver_multi.py           # multilingual retriever analysis
instructions/                       # prompt templates
llm/                                # GPT, Aya, and Qwen wrappers
process_ete_document.py             # merge documents before end-to-end evaluation
run.sh                               # runnable command examples
```

## Setup

Install the Python dependencies required by the models and retrievers you plan to run:

```bash
pip install -r requirements.txt
```

For OpenAI-compatible GPT models:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional
```

For local Hugging Face models, set model roots if your checkpoints are not under the default repository paths:

```bash
export QWEN_MODEL_ROOT="/path/to/Qwen"
export AYA_MODEL_ROOT="/path/to/google"
```

Retriever paths can be either `bm25`, a local `mcontriever-msmarco` checkpoint path, or another path supported by `M3contriver`.

## Core Evaluation Settings

Most RAG scripts support two retrieval settings. In the code, this setting controls `force_first_chunk`:

```text
given_gt  # sets force_first_chunk=True, so the if force_first_chunk block runs and chunk 0 is forced into Top-K
end2end   # sets force_first_chunk=False, so the retriever Top-K is used directly
```

`eval_cross.py` also supports two answer matching modes:

```text
strict    # only checks the answer in the query language
loosen    # checks answers across languages for code-switching / cross-language matches
```

The older `eval_cross_strict.py` and `eval_cross_loosen.py` entry points are merged into `eval_cross.py`.

## Preprocess Documents

Use this to merge documents by language before end-to-end document processing:

```bash
python3 process_ete_document.py \
  --input data/MRAG_mul.json \
  --output processed_data/ete_documents.json
```

The script supports:

```text
doc: "..."
doc: {"en": "...", "zh": "...", ...}
docs: ["...", "..."]
```

It raises an error for files without document fields, such as answer-only files.

## Monolingual Evaluation

Ground-truth retrieval setting:

```bash
python3 eval/eval_mono.py \
  --modelname Qwen2.5-7B-Instruct \
  --contriever_path bm25 \
  --dataset data/MRAG_dataset.json \
  --language en \
  --description lang_specific \
  --mode golden \
  --noise_rate 0.0 \
  --chunk_num 5 \
  --seed 233 \
  --retrieval_setting given_gt
```

End-to-end retrieval setting:

```bash
python3 eval/eval_mono.py \
  --modelname Qwen2.5-7B-Instruct \
  --contriever_path bm25 \
  --dataset data/MRAG_dataset.json \
  --language en \
  --description lang_specific \
  --mode golden \
  --noise_rate 0.0 \
  --chunk_num 5 \
  --seed 233 \
  --retrieval_setting end2end
```

## Cross-Lingual Evaluation

Strict answer matching:

```bash
python3 eval/eval_cross.py \
  --modelname Qwen2.5-7B-Instruct \
  --contriever_path bm25 \
  --dataset data/MRAG_dataset.json \
  --language en \
  --doc_lang zh \
  --description lang_specific \
  --mode golden \
  --noise_rate 0.0 \
  --chunk_num 5 \
  --seed 233 \
  --retrieval_setting given_gt \
  --answer_match_mode strict
```

Loosen answer matching:

```bash
python3 eval/eval_cross.py \
  --modelname Qwen2.5-7B-Instruct \
  --contriever_path bm25 \
  --dataset data/MRAG_dataset.json \
  --language en \
  --doc_lang zh \
  --description lang_specific \
  --mode golden \
  --noise_rate 0.0 \
  --chunk_num 5 \
  --seed 233 \
  --retrieval_setting given_gt \
  --answer_match_mode loosen
```

End-to-end cross-lingual evaluation:

```bash
python3 eval/eval_cross.py \
  --modelname Qwen2.5-7B-Instruct \
  --contriever_path bm25 \
  --dataset data/MRAG_dataset.json \
  --language en \
  --doc_lang zh \
  --description lang_specific \
  --mode golden \
  --noise_rate 0.0 \
  --chunk_num 5 \
  --seed 233 \
  --retrieval_setting end2end \
  --answer_match_mode loosen
```

## Multilingual Evaluation

`eval_multi.py` supports both `given_gt` and `end2end`.

Ground-truth retrieval setting:

```bash
python3 eval/eval_multi.py \
  --modelname Qwen2.5-7B-Instruct \
  --contriever_path bm25 \
  --dataset data/MRAG_mul.json \
  --language en \
  --description lang_specific \
  --mode golden \
  --noise_rate 0.0 \
  --chunk_num 5 \
  --seed 233 \
  --retrieval_setting given_gt
```

End-to-end retrieval setting:

```bash
python3 eval/eval_multi.py \
  --modelname Qwen2.5-7B-Instruct \
  --contriever_path bm25 \
  --dataset data/MRAG_mul.json \
  --language en \
  --description lang_specific \
  --mode golden \
  --noise_rate 0.0 \
  --chunk_num 5 \
  --seed 233 \
  --retrieval_setting end2end
```

## Retriever Evaluation

`eval_contriver.py` is used for the retriever-only experiments in the paper. The batch script used for these runs has two modes:

```text
FIXED_K=""   # k-analysis mode: evaluate recall@1..20 for monolingual pairs only, where query language == document language
FIXED_K="5"  # cross-lingual mode: evaluate recall@5 for all query-language x document-language combinations
```

In both modes, the retriever is the outer loop so that all results for one retriever are completed before moving to the next retriever. The evaluated languages are:

```text
en ko zh fr es pt ar ja
```

The cross-lingual retriever recall run used in the paper corresponds to:

```bash
BASE="$(pwd)"
DATASET="${BASE}/data/MRAG_dataset.json"
SAVE_BASE="${BASE}/contriver_evalresults/cross"
FIXED_K="5"

contriever_paths=(
  "${BASE}/models/BAAI/bge-m3"
  "bm25"
  "${BASE}/models/facebook/mcontriever-msmarco"
)

languages=("en" "ko" "zh" "fr" "es" "pt" "ar" "ja")

for contriever_path in "${contriever_paths[@]}"; do
  contriever_name="$(basename "${contriever_path}")"
  for lang in "${languages[@]}"; do
    for doc_lang in "${languages[@]}"; do
      save_path="${SAVE_BASE}/${contriever_name}/${lang}_ql${lang}_dl${doc_lang}_recall.jsonl"
      mkdir -p "$(dirname "${save_path}")"

      CUDA_VISIBLE_DEVICES=0,1,2,3 python "${BASE}/eval/eval_contriver.py" \
        --dataset "${DATASET}" \
        --language "${lang}" \
        --doc_lang "${doc_lang}" \
        --contriever_path "${contriever_path}" \
        --chunk_size 200 \
        --fixed_k "${FIXED_K}" \
        --save_path "${save_path}"
    done
  done
done
```

For the k-value analysis, use the same script structure but set `FIXED_K=""`, set `SAVE_BASE="${BASE}/contriver_evalresults/check_k"`, and replace `--fixed_k "${FIXED_K}"` with `--max_k 20`. In this mode, only monolingual pairs are evaluated:

```text
query language == document language
```

Single cross-lingual retriever recall example:

```bash
python3 eval/eval_contriver.py \
  --dataset data/MRAG_dataset.json \
  --language en \
  --doc_lang zh \
  --contriever_path bm25 \
  --chunk_size 200 \
  --max_k 20 \
  --seed 233 \
  --save_path results/retriever_recall_en_zh.jsonl
```

Multilingual retriever analysis:

```bash
python3 eval/eval_contriver_multi.py \
  --dataset data/MRAG_mul.json \
  --contriever_path bm25 \
  --chunk_size 200 \
  --top_k 15 \
  --seed 233 \
  --save_dir contriver_evalresults/multilang_analysis
```

## One-Command Example

The repository includes a bash file with all major commands:

```bash
./run.sh
```

Override defaults with environment variables:

```bash
MODEL_NAME=Qwen2.5-7B-Instruct \
CONTRIEVER_PATH=bm25 \
QUERY_LANG=en \
DOC_LANG=zh \
CHUNK_NUM=5 \
./run.sh
```

## Outputs

Generation outputs are written under:

```text
results/mono/
results/cross_lingual/
results/multi_lingual/
results/retriever_recall_*.jsonl
contriver_evalresults/
```

Cross-lingual results are separated by answer matching mode and retrieval setting, for example:

```text
results/cross_lingual/MRAG_dataset._lang_specific/original_strict/
results/cross_lingual/MRAG_dataset._lang_specific/original_loosen/
results/cross_lingual/MRAG_dataset._lang_specific/original_loosen_end2end/
```

## Notes

- `max_tokens` and prompt truncation are set to `8192` in `llm/`.
- `given_gt` executes the `if force_first_chunk:` block and forces chunk 0 into Top-K.
- `end2end` skips the `if force_first_chunk:` block and uses the retriever Top-K directly.
