# code_mrag

This repository keeps the runnable evaluation code, metric aggregation scripts, and the JSON data files required by those workflows.

## Layout

- `eval/`: evaluation and inference scripts
- `metrics/`: metric aggregation scripts
- `data/`: input JSON files used by the evaluation scripts
- `instructions/`: prompt templates
- `llm/`: model wrappers
- `src/`: retriever and utility code

## Before Running

Run commands from the repository root:

```bash
cd code_mrag
```

Set API keys when using hosted LLMs:

```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=
export DASHSCOPE_API_KEY=your_key
```

If you use local models, point the code to your model directories:

```bash
export QWEN_MODEL_ROOT=/path/to/Qwen
export AYA_MODEL_ROOT=/path/to/google
export QWEN_LEGACY_MODEL_ROOT=/path/to/Qwen
```

Some scripts also expect a retriever model path such as:

```bash
/path/to/contriever
```

You can also pass `bm25` to `--contriever_path` for the sparse BM25 baseline, or a BGE-M3/M3-style model path for the `M3contriver` wrapper.

## Paper-Aligned Settings

The paper evaluates Futurepedia in three tasks:

- Monolingual Knowledge Extraction: query and document are in the same language.
- Cross-lingual Knowledge Transfer: query and document are in different languages.
- Multilingual Knowledge Selection: documents from all eight languages are provided together.

The LLM/RAG scripts support two context settings:

- `--retrieval_setting given_gt`: ground-truth context setting. Retrieval uses `query + ground-truth answer` and keeps chunk 0 in the Top-K context. This preserves the original LLM-only/oracle-context behavior.
- `--retrieval_setting end2end`: end-to-end RAG setting. Retrieval uses the query only and returns the actual retriever Top-K chunks. 

Use `--chunk_num 5` to match the paper's Top-5 context setting. The supported languages are `en`, `zh`, `fr`, `ja`, `ko`, `ar`, `es`, and `pt`.

## Eval

### 1. Monolingual Knowledge Extraction

Ground-truth context:

```bash
python3 eval/eval_mono.py \
  --modelname Qwen3-8B \
  --contriever_path /path/to/contriever \
  --dataset ./data/MRAG_dataset.json \
  --language en \
  --description lang_specific \
  --mode golden \
  --retrieval_setting given_gt \
  --chunk_num 5 \
  --seed 233
```

End-to-end RAG:

```bash
python3 eval/eval_mono.py \
  --modelname Qwen3-8B \
  --contriever_path /path/to/contriever \
  --dataset ./data/MRAG_dataset.json \
  --language en \
  --description lang_specific \
  --mode golden \
  --retrieval_setting end2end \
  --chunk_num 5 \
  --seed 233
```

Outputs go under `./results/mono/`. End-to-end runs are written to directories with the `_end2end` suffix.

### 2. Cross-Lingual Knowledge Transfer

Ground-truth context:

```bash
python3 eval/eval_cross.py \
  --modelname gpt-5.1 \
  --contriever_path /path/to/contriever \
  --dataset ./data/MRAG_dataset.json \
  --language en \
  --doc_lang zh \
  --description lang_specific \
  --mode golden \
  --retrieval_setting given_gt \
  --chunk_num 5 \
  --seed 233
```

End-to-end RAG:

```bash
python3 eval/eval_cross.py \
  --modelname gpt-5.1 \
  --contriever_path /path/to/contriever \
  --dataset ./data/MRAG_dataset.json \
  --language en \
  --doc_lang zh \
  --description lang_specific \
  --mode golden \
  --retrieval_setting end2end \
  --chunk_num 5 \
  --seed 233
```

Related variants:

- `eval/eval_cross_strict.py`
- `eval/eval_loosen.py`

Outputs go under `./results/cross_lingual/`. End-to-end runs are written to `original_loosen_end2end`.

### 3. Multilingual Knowledge Selection

Ground-truth context:

```bash
python3 eval/eval_multi.py \
  --modelname gpt-5.1 \
  --contriever_path /path/to/contriever \
  --dataset ./data/mul_ans_all_replaced.json \
  --language en \
  --description lang_specific \
  --mode golden \
  --retrieval_setting given_gt \
  --chunk_num 5 \
  --seed 233
```

End-to-end RAG:

```bash
python3 eval/eval_multi.py \
  --modelname gpt-5.1 \
  --contriever_path /path/to/contriever\
  --dataset ./data/mul_ans_all_replaced.json \
  --language en \
  --description lang_specific \
  --mode golden \
  --retrieval_setting end2end \
  --chunk_num 5 \
  --seed 233
```

Outputs go under `./results/multi_lingual/`. End-to-end runs are written to directories with the `_end2end` suffix. In `end2end` multilingual runs, the selected query language is used to retrieve from all document languages, matching a real single-query multilingual RAG pipeline.

### 4. Position / vote variants

Position-based:

```bash
python3 eval/eval_mul_pos.py \
  --modelname Qwen3-8B \
  --contriever_path /path/to/contriever \
  --dataset ./data/mul_ans_all_replaced.json \
  --language en \
  --description lang_specific \
  --mode pos \
  --pos 1 \
  --pos_lang en
```

Vote-based:

```bash
python3 eval/eval_mul_vote.py \
  --modelname Qwen3-8B \
  --contriever_path /path/to/contriever \
  --dataset ./data/mul_ans_all_replaced.json \
  --language en \
  --description lang_specific \
  --mode vote \
  --sample_languages_candidate '["ja","ko","ar","es","pt","zh","fr"]' \
  --sample_num 3
```

### 5. Retriever analysis

Single-language retriever recall:

```bash
python3 eval/eval_contriver.py \
  --dataset ./data/MRAG_dataset.json \
  --language en \
  --doc_lang en \
  --contriever_path /path/to/contriever \
  --chunk_size 200 \
  --max_k 20 \
  --save_path ./results/retriever_recall.jsonl
```

Multilingual retriever analysis:

```bash
python3 eval/eval_contriver_multi.py \
  --dataset ./data/mul_ans_all_replaced.json \
  --contriever_path /path/to/contriever \
  --chunk_size 200 \
  --top_k 15 \
  --save_dir ./contriver_evalresults/multilang_analysis
```

Improved mono evaluation for prepared chunk files:

```bash
python3 eval/eval_improve_mono.py \
  --modelname Qwen3-8B \
  --dataset ./data/rag_eval/translated_Arabic_English.json \
  --language en
```

## Metrics

These scripts read result files and write aggregated JSON summaries.

### 1. Monolingual metric aggregation

For ground-truth context, use `./results/mono`. For end-to-end RAG, the same recursive search works if `--results_root` still points to `./results/mono`; the aggregated files will include runs from both settings unless you point it to a narrower directory.

```bash
python3 metrics/compute_mono.py \
  --model_name Qwen3-8B \
  --contriver_path /path/to/contriever \
  --results_root ./results/mono \
  --out_dir ./avg_results/mono/Qwen3-8B
```

### 2. Cross-lingual metric aggregation

Ground-truth context:

```bash
python3 metrics/compute_avgcross.py \
  --results_root ./results/cross_lingual/MRAG_dataset._lang_specific/original_loosen \
  --out_dir ./avg_results/cross/original_loosen
```

End-to-end RAG:

```bash
python3 metrics/compute_avgcross.py \
  --results_root ./results/cross_lingual/MRAG_dataset._lang_specific/original_loosen_end2end \
  --out_dir ./avg_results/cross/original_loosen_end2end
```

There is also a helper shell script:

```bash
bash cross_avg.sh
```

### 3. Multilingual metric aggregation

Use `--results_root ./results/multi_lingual` for all runs, or point to a specific `_end2end` directory when you want to aggregate only the real RAG setting.

```bash
python3 metrics/compute_avgmulti.py \
  --results_root ./results/multi_lingual \
  --out_dir ./avg_results/multi \
  --contriever contriever
```

### 4. Position / vote metric summaries

```bash
python3 metrics/compute_multi_pos.py
python3 metrics/compute_multi_vote.py
```

These scripts use the default repository-relative result directories already defined in the code.

### 5. Final mono summary

```bash
python3 metrics/final_monocompute.py
```

This script reads aggregated mono result JSON files from `./avg_results/mono/` and writes a combined summary to `./llm_results/mono/`.

## Notes

- Output directories such as `results/`, `avg_results/`, `llm_results/`, and `contriver_evalresults/` are intentionally ignored by Git.
- `data/` is intentionally kept in the repository because the evaluation scripts need those JSON files.
- Some scripts still contain task-specific assumptions such as fixed model lists or output naming conventions. Adjust arguments or local constants as needed for your own runs.
