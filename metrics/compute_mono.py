#!/usr/bin/env python3
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import glob
import argparse
from collections import defaultdict

from public_repo import repo_path

LANGUAGES = ["en", "ko", "zh", "fr", "es", "pt", "ar", "ja"]
SEEDS = [233, 42, 31415]

def contriever_name_from_path(cpath: str) -> str:
    cpath_l = cpath.lower()
    if "mcontriever-msmarco" in cpath_l:
        return "mcontriever-msmarco"
    if "bm25" in cpath_l:
        return "bm25"
    return "m3contriever"

def load_score(fp: str) -> float | None:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # 你给的字段名
        if "character 3-gram Recall" in obj:
            return float(obj["character 3-gram Recall"])
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="e.g. Qwen3-8B")
    parser.add_argument("--contriver_path", required=True, help="e.g. bm25 or /path/to/bge-m3 or /path/to/mcontriever-msmarco")
    parser.add_argument("--results_root", default=repo_path("results", "mono"), help="mono results root")
    parser.add_argument("--out_dir", default=repo_path("avg_results", "mono"), help="output dir")
    args = parser.parse_args()

    model = args.model_name
    contriever_name = contriever_name_from_path(args.contriver_path)

    # (lang -> list of scores)
    scores_by_lang = defaultdict(list)
    # for debug
    files_used = defaultdict(list)
    missing = []

    for lang in LANGUAGES:
        # 递归搜索：不依赖目录层级顺序
        # 文件名核心稳定部分：..._{contriever_name}_{lang}_{model}_..._seed{seed}_result.json
        pattern = os.path.join(
            args.results_root,
            "**",  # 递归搜索所有子目录
            args.model_name,
            contriever_name,
            f"*_{contriever_name}_{lang}_{model}_*_result.json"
        )
        matches = sorted(glob.glob(pattern, recursive=True))
        print(f"[DEBUG] Found {len(matches)} files for lang={lang}, pattern={pattern}")
        for fp in matches:
            score = load_score(fp)
            if score is not None:
                scores_by_lang[lang].append(score)
                files_used[lang].append(fp)
            else:
                missing.append(fp)
           

    # 计算均值
    avg_by_lang = {}
    detail = {}
    for lang in LANGUAGES:
        vals = scores_by_lang.get(lang, [])
        if vals:
            avg = sum(vals) / len(vals)
            avg_by_lang[lang] = avg
            detail[lang] = {
                "mean": avg,
                "count": len(vals),
                "values": vals,   # 如你不想保存每个值可删掉这一行
            }
        else:
            avg_by_lang[lang] = None
            detail[lang] = {"mean": None, "count": 0, "values": []}

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{contriever_name}_{model}.json")

    out_obj = {
        "model": model,
        "contriever": contriever_name,
        "seeds": SEEDS,
        "languages": LANGUAGES,
        "avg_character_3gram_recall": avg_by_lang,
        "detail": detail,
        "missing_lang_seed": missing,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote: {out_path}")
    # 简单打印一行 summary
    for lang in LANGUAGES:
        print(f"{lang}: {avg_by_lang[lang]} (n={detail[lang]['count']})")

if __name__ == "__main__":
    main()
