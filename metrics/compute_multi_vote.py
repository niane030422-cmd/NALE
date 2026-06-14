import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from public_repo import repo_path

def process_vote_results():
    # 1. 配置路径
    # 输入目录：vote 实验结果
    base_dir = repo_path("results", "multi_lingual", "mul_ans_all_replaced_lang_specific", "vote", "Qwen3-8B")
    # 输出目录：汇总数据存放处
    output_dir = repo_path("llm_results", "multi", "vote")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")

    # 用于存储汇总结果
    summary_results = {}

    # 2. 循环处理不同的 sample 值 (通常 k 投票实验包含 1~5 或 1~8)
    # 如果你的 sample 范围不同，可以修改 range(1, 6)
    for k in range(1, 6):
        # 构造文件名：注意这里 qlzh 代表 query 是中文，这在投票实验中很常见
        filename = f"prediction_sample{k}_cnum7_mul_ans_all_replaced_qlen_lmquery_lang_Qwen3-8B_noise0.0_chunk5_seed233_result.json"
        file_path = os.path.join(base_dir, filename)

        if not os.path.exists(file_path):
            # 如果 sample{k} 的文件不存在则跳过
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取 res_dict_acc_3gram 字典
            metrics = data.get("res_dict_acc_3gram", {})
            
            if metrics:
                # 计算该 k 值下所有语言的平均分
                all_values = list(metrics.values())
                average_score = sum(all_values) / len(all_values)
                
                # 提取 en 的分数 (分析英语偏见的关键指标)
                en_score = metrics.get("en", 0)
                # 提取被投票语种的分数 (例如这里 query 是 zh，可以顺便看 zh)
                zh_score = metrics.get("zh", 0)

                # 存入汇总字典
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

    # 3. 保存汇总结果
    output_filename = "qwen3_8b_en_vote_summary.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 汇总完成！数据已保存至: {output_path}")

if __name__ == "__main__":
    process_vote_results()
