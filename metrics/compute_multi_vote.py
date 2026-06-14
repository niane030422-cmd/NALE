import json

import os

import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:

    sys.path.insert(0, str(REPO_ROOT))

from public_repo import repo_path

def process_vote_results():

    base_dir = repo_path("results", "multi_lingual", "mul_ans_all_replaced_lang_specific", "vote", "Qwen3-8B")

    output_dir = repo_path("llm_results", "multi", "vote")

    if not os.path.exists(output_dir):

        os.makedirs(output_dir, exist_ok=True)

        print(f"创建输出目录: {output_dir}")

    summary_results = {}

    for k in range(1, 6):

        filename = f"prediction_sample{k}_cnum7_mul_ans_all_replaced_qlen_lmquery_lang_Qwen3-8B_noise0.0_chunk5_seed233_result.json"

        file_path = os.path.join(base_dir, filename)

        if not os.path.exists(file_path):

            continue

        try:

            with open(file_path, 'r', encoding='utf-8') as f:

                data = json.load(f)

            metrics = data.get("res_dict_acc_3gram", {})

            if metrics:

                all_values = list(metrics.values())

                average_score = sum(all_values) / len(all_values)

                en_score = metrics.get("en", 0)

                zh_score = metrics.get("zh", 0)

                summary_results[f"sample_{k}"] = {

                    "mean_all_languages": round(average_score, 4),

                    "en_score": en_score,

                    "zh_score": zh_score,

                    "k_value": k

                }

                print(f"成功处理 sample_{k}: 平均值={average_score:.2f}, en={en_score}")

            else:

                print(f"警告: 文件 {filename} 中未找到数据")

        except Exception as e:

            print(f"处理文件 {filename} 时出错: {str(e)}")

    output_filename = "qwen3_8b_en_vote_summary.json"

    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:

        json.dump(summary_results, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 汇总完成！数据已保存至: {output_path}")

if __name__ == "__main__":

    process_vote_results()
