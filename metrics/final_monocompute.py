import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from public_repo import repo_path

def calculate_stats():
    # 1. 定义输入文件路径（已新增 gpt-5.1）
    file_8b = repo_path("avg_results", "mono", "Qwen3-8B", "mcontriever-msmarco_Qwen3-8B.json")
    file_30b = repo_path("avg_results", "mono", "Qwen3-30B-A3B-Instruct-2507", "mcontriever-msmarco_Qwen3-30B-A3B-Instruct-2507.json")
    file_gpt5 = repo_path("avg_results", "mono", "gpt-5.1", "mcontriever-msmarco_gpt-5.1.json")
    
    files_to_process = {
        "Qwen3-8B": file_8b,
        "Qwen3-30B": file_30b,
        "gpt-5.1": file_gpt5  # 新增项
    }

    # 2. 定义输出路径并确保目录存在
    out_dir = repo_path("llm_results", "mono")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "mono_avg_and_variance.json")

    final_results = {}

    # 3. 遍历读取并计算
    for model_name, filepath in files_to_process.items():
        if not os.path.exists(filepath):
            print(f"❌ 文件不存在: {filepath}")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 提取数据
        recall_data = data.get("avg_character_3gram_recall", {})
        
        if not recall_data:
            print(f"⚠️ 在 {model_name} 中没有找到 'avg_character_3gram_recall' 字段")
            continue
            
        # 获取那 8 个语言的数值列表
        scores = list(recall_data.values())
        
        # 计算平均值和方差
        # 注意：这里计算的是原始浮点数（0~1之间）的均值和方差
        mean_val = np.mean(scores)
        variance_val = np.var(scores) # 总体方差
        
        # 存入结果字典
        final_results[model_name] = {
            "mean": float(mean_val),
            "variance": float(variance_val),
            "raw_scores": recall_data 
        }
        
        print(f"✅ {model_name} 计算完成:")
        print(f"   - 平均值 (Mean): {mean_val:.4f}")
        print(f"   - 方差 (Variance): {variance_val:.4f}\n")

    # 4. 保存为新的 JSON 文件
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"🎉 所有结果（含 GPT-5.1）已成功保存至: {out_file}")

if __name__ == "__main__":
    calculate_stats()
