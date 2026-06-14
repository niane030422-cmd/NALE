# Public Release Sanitization Notes

This repository was cleaned for public release on 2026-06-14.

## Sensitive items cleaned

- `llm/gpt.py`: removed hard-coded API key; now reads `OPENAI_API_KEY`
- `llm/hf.py`: removed hard-coded API key; now reads `DASHSCOPE_API_KEY`

## Personal-path cleanup

The following code areas were rewritten to avoid personal absolute paths:

- `llm/hf.py`
- root-level evaluation and plotting scripts
- `scripts/`
- `scripts_tmp/`

The public version now uses repository-relative defaults or environment variables for external model locations.

## Packaging choices

- `data/` is intentionally kept in the repository because these JSON files are required for inference/evaluation.
- Plotting utilities and batch experiment shell scripts were removed from the public package to keep the release focused on runnable inference code.
- Evaluation scripts now live under `eval/`.
- Metric aggregation scripts now live under `metrics/`.

## Important follow-up

The old API keys were present in the working copy before sanitization. Treat them as exposed credentials and rotate or revoke them before publishing.
