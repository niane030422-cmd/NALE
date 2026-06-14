import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from public_repo import repo_path

def process_pos_sum_results():
    # 1. 定义路径
    input_base_dir = repo_path("results", "multi_lingual", "mul_ans_all_replaced_lang_specific", "pos", "Qwen3-8B")
    output_dir = repo_path("llm_results", "multi", "pos")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    summary_results = {}

    # 2. 循环处理 pos_1 到 pos_8
    for pos in range(1, 9):
        # 构造文件名
        filename = f"prediction_pos_{pos}_plen_mul_ans_all_replaced_qlen_lmquery_lang_Qwen3-8B_noise0.0_chunk5_seed233_result.json"
        file_path = os.path.join(input_base_dir, filename)

        if not os.path.exists(file_path):
            print(f"跳过不存在的文件: {filename}")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 获取目标字典
            metrics = data.get("res_dict_acc_3gram", {})
            
            if metrics:
                # 提取所有分值
                all_values = list(metrics.values())
                
                # 计算所有语种的平均值 (Mean of all languages)
                total_avg = sum(all_values) / len(all_values)
                
                # 计算排除 en 之外的所有语种的总和 (Sum of non-English languages)
                # 逻辑：把所有键不是 'en' 的数值加起来
                non_en_sum = sum(v for k, v in metrics.items() if k != 'en')

                # 存入该位置的结果
                summary_results[f"pos_{pos}"] = {
                    "all_lang_avg": round(total_avg, 4),    # 8个语种的平均值
                    "non_en_total_sum": round(non_en_sum, 4), # 除了en之外7个语种的总和
                    "en_score": metrics.get("en"),           # 单独列出en的分数
                    "raw_scores": metrics                    # 原始数据备份
                }
                print(f"成功处理 pos_{pos}: 总均值={total_avg:.2f}, 非英总和={non_en_sum:.2f}")
            else:
                print(f"警告: 文件 {filename} 中未找到数据")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")

    # 3. 保存汇总结果
    output_file = os.path.join(output_dir, "qwen3_8b_pos_sum_summary.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 处理完成！汇总文件已保存至: {output_file}")

if __name__ == "__main__":
    process_pos_sum_results()
