#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
results_root="${REPO_ROOT}/results/cross_lingual/MRAG_dataset._lang_specific"
result=('golden' 'golden_loosen' 'original' 'original_loosen')
out_dir="${REPO_ROOT}/avg_results/cross"
python_script="${REPO_ROOT}/metrics/compute_avgcross.py"

for i in "${result[@]}"
do
    current_input="${results_root}/${i}"
    current_output="${out_dir}/${i}"
    mkdir -p "$current_output"

    python "$python_script" --results_root "$current_input" --out_dir "$current_output"

    echo "--------------------------------------"
done

echo "所有任务已完成！"
